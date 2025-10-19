import gpmf
import json
from datetime import datetime, timedelta
import numpy as np
import math
import argparse
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def haversineDistance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def extractGpsData(videoPath: str):
    print(f'Extracting GPS data from "{videoPath}"', end='')
    outputJsonPath = videoPath.replace('.MP4', '.json')

    try:
        # Read the binary stream from the file
        gpmfStream = gpmf.io.extract_gpmf_stream(videoPath)
    except Exception as e:
        print(f"\nError reading GPMF stream from {videoPath}: {e}")
        return

    # Extract GPS low level data from the stream
    gpsBlocks = gpmf.gps.extract_gps_blocks(gpmfStream)
    if not gpsBlocks:
        print("\nNo GPS data found in the video.")
        return

    # Parse low level data into more usable format
    parsedGpsData = list(map(gpmf.gps.parse_gps_block, gpsBlocks))

    print(".", end='')

    # Prepare data for JSON output
    gpsPoints = []
    lastTimeIncrement = None
    for i, gpsBlock in enumerate(parsedGpsData):
        if not hasattr(gpsBlock, 'latitude') or gpsBlock.latitude is None:
            continue

        # Handle both array and scalar latitude values
        if isinstance(gpsBlock.latitude, (np.ndarray, list)):
            numSamples = len(gpsBlock.latitude)
            if numSamples == 0:
                continue
        else:
            # Handle case where latitude is a single value (scalar)
            numSamples = 1

        startTimeStr = str(gpsBlock.timestamp)
        startTime = datetime.fromisoformat(startTimeStr)

        timeIncrement = None
        
        # Calculate time increment based on the time difference between this block and the next
        if i + 1 < len(parsedGpsData):
            nextBlock = parsedGpsData[i+1]
            if hasattr(nextBlock, 'timestamp'):
                endTimeStr = str(nextBlock.timestamp)
                endTime = datetime.fromisoformat(endTimeStr)
                blockDuration = endTime - startTime
                if numSamples > 0:
                    timeIncrement = blockDuration / numSamples
                    lastTimeIncrement = timeIncrement  # Save for the last block
        elif lastTimeIncrement:
            # For the last block, use the same time increment as the previous block
            timeIncrement = lastTimeIncrement

        for j in range(numSamples):
            timestampToWrite = startTime
            if timeIncrement:
                timestampToWrite = startTime + timeIncrement * j

            timestampStr = timestampToWrite.isoformat() if isinstance(timestampToWrite, datetime) else str(timestampToWrite)

            # The fix and precision are per-block, not per-sample            # Access values safely, handling both array and scalar cases
            latitude = gpsBlock.latitude[j] if isinstance(gpsBlock.latitude, (np.ndarray, list)) else gpsBlock.latitude
            longitude = gpsBlock.longitude[j] if isinstance(gpsBlock.longitude, (np.ndarray, list)) else gpsBlock.longitude
            altitude = gpsBlock.altitude[j] if isinstance(gpsBlock.altitude, (np.ndarray, list)) else gpsBlock.altitude
            speed2d = gpsBlock.speed_2d[j] if isinstance(gpsBlock.speed_2d, (np.ndarray, list)) else gpsBlock.speed_2d
            speed3d = gpsBlock.speed_3d[j] if isinstance(gpsBlock.speed_3d, (np.ndarray, list)) else gpsBlock.speed_3d
            
            gpsPoints.append({
                "timestamp": timestampStr,
                "fixType": int(gpsBlock.fix) if hasattr(gpsBlock, 'fix') else 0,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "speed2d": speed2d,
                "speed3d": speed3d,
                "accuracy": gpsBlock.precision if hasattr(gpsBlock, 'precision') else 0
            })

    if not gpsPoints:
        print("\nCould not extract any valid GPS points.")
        return

    # --- Post-processing: Heading and Turn Rate Calculation ---    
    # Get the start time of the recording
    recordingStartTime = datetime.fromisoformat(gpsPoints[0]['timestamp'])    # First pass: Convert timestamps and find maxSpeed
    maxSpeed = 0
    for gpsPoint in gpsPoints:
        gpsPoint['timestamp'] = datetime.fromisoformat(gpsPoint['timestamp'])
        if gpsPoint['speed2d'] is not None and gpsPoint['speed2d'] > maxSpeed:
            maxSpeed = gpsPoint['speed2d']
    
    # Second pass: Calculate heading based on points in the past AND future
    for i in range(len(gpsPoints)):
        # If speed is less than 2.5 kph (0.69 m/s), use the last heading (can't turn if stationary)
        if  gpsPoints[i]['speed2d'] < 0.69:
            gpsPoints[i]['heading'] = gpsPoints[i-1].get('heading', 0) if i > 0 else 0
            continue
            
        # Set minimum distance threshold based on GPS accuracy
        minDistance = gpsPoints[i].get('accuracy', 2.0)*2 # Default to 4m if accuracy is not available
        
        # Look for a point in the past (approximately 1s in the past)
        pastTimestamp = gpsPoints[i]['timestamp'] - timedelta(seconds=1.0)
        pastHeading = None
        
        # Find a point approximately 1s in the past
        j = i - 1
        while j >= 0 and gpsPoints[j]['timestamp'] > pastTimestamp:
            j -= 1

        # If the point 1s in the past is not far enough away, continue looking further back
        pastPointFound = False
        while j >= 0:
            if gpsPoints[j]['latitude'] is not None:
                distance = haversineDistance(
                    gpsPoints[i]['latitude'], gpsPoints[i]['longitude'],
                    gpsPoints[j]['latitude'], gpsPoints[j]['longitude']
                )
                if distance >= minDistance:
                    pastPointFound = True
                    break
            j -= 1
        
        # Calculate heading from past point if found
        if pastPointFound and gpsPoints[j]['latitude'] is not None:
            lat1 = math.radians(gpsPoints[j]['latitude'])
            lon1 = math.radians(gpsPoints[j]['longitude'])
            lat2 = math.radians(gpsPoints[i]['latitude'])
            lon2 = math.radians(gpsPoints[i]['longitude'])

            if lat1 != lat2 or lon1 != lon2:
                diffLongitude = lon2 - lon1
                x = math.sin(diffLongitude) * math.cos(lat2)
                y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLongitude))
                initialBearing = math.atan2(x, y)
                compassBearing = (math.degrees(initialBearing) + 360) % 360
                pastHeading = compassBearing
        
        # Look for a point in the future (approximately 1s ahead)
        futureTimestamp = gpsPoints[i]['timestamp'] + timedelta(seconds=1.0)
        futureHeading = None
        
        # Find a point approximately 1s in the future
        k = i + 1
        while k < len(gpsPoints) and gpsPoints[k]['timestamp'] < futureTimestamp:
            k += 1

        # If the point 1s in the future is not far enough away, continue looking further ahead
        futurePointFound = False
        while k < len(gpsPoints):
            if gpsPoints[k]['latitude'] is not None:
                distance = haversineDistance(
                    gpsPoints[i]['latitude'], gpsPoints[i]['longitude'],
                    gpsPoints[k]['latitude'], gpsPoints[k]['longitude']
                )
                if distance >= minDistance:
                    futurePointFound = True
                    break
            k += 1
        
        # Calculate heading from current point to future point if found
        if futurePointFound and gpsPoints[k]['latitude'] is not None:
            lat1 = math.radians(gpsPoints[i]['latitude'])
            lon1 = math.radians(gpsPoints[i]['longitude'])
            lat2 = math.radians(gpsPoints[k]['latitude'])
            lon2 = math.radians(gpsPoints[k]['longitude'])

            if lat1 != lat2 or lon1 != lon2:
                diffLongitude = lon2 - lon1
                x = math.sin(diffLongitude) * math.cos(lat2)
                y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLongitude))
                initialBearing = math.atan2(x, y)
                compassBearing = (math.degrees(initialBearing) + 360) % 360
                futureHeading = compassBearing
        
        # Combine the past and future headings
        if pastHeading is not None and futureHeading is not None:
            # Get the average bearing, handling the circular nature of angles
            diff = futureHeading - pastHeading
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            gpsPoints[i]['heading'] = (pastHeading + (diff / 2)) % 360
        elif pastHeading is not None:
            gpsPoints[i]['heading'] = pastHeading
        elif futureHeading is not None:
            gpsPoints[i]['heading'] = futureHeading
        else:
            # If no suitable points found, use previous heading or default to 0
            gpsPoints[i]['heading'] = gpsPoints[i-1].get('heading', 0) if i > 0 else 0

    # Third pass: Calculate instantaneous turn rates and acceleration
    # Third pass: Calculate instantaneous turn rates and acceleration
    instantaneousTurnRates = [0] * len(gpsPoints)
    for i in range(1, len(gpsPoints)):
        if (gpsPoints[i]['timestamp'] - recordingStartTime).total_seconds() < 1.0:
            continue

        # Calculate turn rate
        # Calculate turn rate
        heading1 = gpsPoints[i-1]['heading']
        heading2 = gpsPoints[i]['heading']
        headingChange = heading2 - heading1
        if headingChange > 180: headingChange -= 360
        elif headingChange < -180: headingChange += 360
        
        # Time difference between consecutive points
        
        # Time difference between consecutive points
        timeDiff = (gpsPoints[i]['timestamp'] - gpsPoints[i-1]['timestamp']).total_seconds()
        
        
        if timeDiff > 0:
            # Calculate turn rate
            # Calculate turn rate
            instantaneousTurnRates[i] = headingChange / timeDiff
            
            # Calculate acceleration (m/s²) from speed difference
            speedDiff = gpsPoints[i]['speed2d'] - gpsPoints[i-1]['speed2d']
            gpsPoints[i]['acceleration'] = speedDiff / timeDiff
        else:
            # Default acceleration if time difference is zero
            gpsPoints[i]['acceleration'] = 0.0


    print(".", end='')    
    
    # Default acceleration for the first point
    gpsPoints[0]['acceleration'] = 0.0    # Fourth pass: Average turn rates and accelerations over a symmetric rolling window (+/- 0.5s)
            
            # Calculate acceleration (m/s²) from speed difference
            speedDiff = gpsPoints[i]['speed2d'] - gpsPoints[i-1]['speed2d']
            gpsPoints[i]['acceleration'] = speedDiff / timeDiff
        else:
            # Default acceleration if time difference is zero
            gpsPoints[i]['acceleration'] = 0.0


    print(".", end='')    
    
    # Default acceleration for the first point
    gpsPoints[0]['acceleration'] = 0.0    # Fourth pass: Average turn rates and accelerations over a symmetric rolling window (+/- 0.5s)
    averagedTurnRates = [0] * len(gpsPoints)
    averagedAccelerations = [0] * len(gpsPoints)
    
    averagedAccelerations = [0] * len(gpsPoints)
    
    for i in range(len(gpsPoints)):
        startTimeWindow = gpsPoints[i]['timestamp'] - timedelta(seconds=0.5)
        endTimeWindow = gpsPoints[i]['timestamp']
        
        # Collect points within the time window
        pointsInWindow = []
        for k in range(len(gpsPoints)):
            if startTimeWindow <= gpsPoints[k]['timestamp'] <= endTimeWindow:
                pointsInWindow.append(k)
        
        # Average turn rates in window
        ratesInWindow = [instantaneousTurnRates[k] for k in pointsInWindow]
        endTimeWindow = gpsPoints[i]['timestamp']
        
        # Collect points within the time window
        pointsInWindow = []
        for k in range(len(gpsPoints)):
            if startTimeWindow <= gpsPoints[k]['timestamp'] <= endTimeWindow:
                pointsInWindow.append(k)
        
        # Average turn rates in window
        ratesInWindow = [instantaneousTurnRates[k] for k in pointsInWindow]
        if ratesInWindow:
            averagedTurnRates[i] = sum(ratesInWindow) / len(ratesInWindow)

        # Average accelerations in window
        accelsInWindow = [gpsPoints[k].get('acceleration', 0.0) for k in pointsInWindow if k > 0]  # Skip first point
        if accelsInWindow:
            averagedAccelerations[i] = sum(accelsInWindow) / len(accelsInWindow)
            

        # Average accelerations in window
        accelsInWindow = [gpsPoints[k].get('acceleration', 0.0) for k in pointsInWindow if k > 0]  # Skip first point
        if accelsInWindow:
            averagedAccelerations[i] = sum(accelsInWindow) / len(accelsInWindow)
            
    print(".", end='\r')
    # Fifth pass: Apply additional smoothing to turn rate and acceleration, then add max speed
    # Fifth pass: Apply additional smoothing to turn rate and acceleration, then add max speed
    for i in range(len(gpsPoints)):
        # Apply further smoothing with a 3-point moving average
        # Apply further smoothing with a 3-point moving average
        if i > 0 and i < len(gpsPoints) - 1:
            # Smooth turn rate
            # Smooth turn rate
            gpsPoints[i]['turnRate'] = (averagedTurnRates[i-1] + averagedTurnRates[i] + averagedTurnRates[i+1]) / 3
            
            # Smooth acceleration
            smoothedAccel = (averagedAccelerations[i-1] + averagedAccelerations[i] + averagedAccelerations[i+1]) / 3
            
            # Apply additional clamping to acceleration to reduce extreme values
            gpsPoints[i]['acceleration'] = max(min(smoothedAccel, 3.0), -3.0)
            
            # Smooth acceleration
            smoothedAccel = (averagedAccelerations[i-1] + averagedAccelerations[i] + averagedAccelerations[i+1]) / 3
            
            # Apply additional clamping to acceleration to reduce extreme values
            gpsPoints[i]['acceleration'] = max(min(smoothedAccel, 3.0), -3.0)
        else:
            gpsPoints[i]['turnRate'] = averagedTurnRates[i]
            gpsPoints[i]['acceleration'] = max(min(averagedAccelerations[i], 3.0), -3.0)
        
            gpsPoints[i]['acceleration'] = max(min(averagedAccelerations[i], 3.0), -3.0)
        
        gpsPoints[i]['maxSpeed'] = maxSpeed
        gpsPoints[i]['timestamp'] = gpsPoints[i]['timestamp'].isoformat()


    # Write to JSON file
    with open(outputJsonPath, 'w') as f:
        json.dump(gpsPoints, f, indent=4, cls=NumpyEncoder)

    print(f'Extracting GPS data from "{videoPath}" Done.')

def main(path:str):
def main(path:str):

    if os.path.isdir(path):
        print(f"Processing all MP4 files in directory: {path}...\n")
        for file in os.listdir(path):
    if os.path.isdir(path):
        print(f"Processing all MP4 files in directory: {path}...\n")
        for file in os.listdir(path):
            if file.lower().endswith('.mp4'):
                videoPath = os.path.join(path, file)
                videoPath = os.path.join(path, file)
                extractGpsData(videoPath)
                print()
                
    elif os.path.isfile(path) and path.lower().endswith('.mp4'):
        extractGpsData(path)
    elif os.path.isfile(path) and path.lower().endswith('.mp4'):
        extractGpsData(path)
    else:
        print(f"Error: The path '{path}' is not a valid MP4 file or directory.")
        print(f"Error: The path '{path}' is not a valid MP4 file or directory.")

if __name__ == "__main__":
    path = r"D:\VS_Python_Project\Autopilot\Autopilot\Test_drive\2025.06.25"
    main(path)