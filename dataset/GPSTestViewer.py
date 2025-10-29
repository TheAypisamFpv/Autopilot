import cv2
import json
from datetime import datetime, timedelta
import os
import time
import math
import numpy as np

# Increase the read attempts limit for FFmpeg
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192'


def loadGpsData(jsonPath):
    print(f'Loading GPS data from "{jsonPath}"', end='')
    with open(jsonPath, 'r') as f:
        data = json.load(f)
    
    # Convert timestamps from string to datetime objects
    for point in data:
        point['timestamp'] = datetime.fromisoformat(point['timestamp'])

    print(" done.")
    return data

def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    """Draws text with a black outline."""
    # Draw the outline in black
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw the main text in the specified color
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def main(videoPath:str = None):
    jsonPath = videoPath.replace('.MP4', '.json')

    if not os.path.exists(videoPath):
        print(f"Error: Video file not found at {videoPath}")
        return
    if not os.path.exists(jsonPath):
        print(f"Error: JSON file not found at {jsonPath}")
        return

    # Load the data
    gpsData = loadGpsData(jsonPath)
    # Use the MSMF backend on Windows, which can be more stable
    cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Error: Could not open video with MSMF backend, trying default...")
        cap = cv2.VideoCapture(videoPath) # Fallback to default
        if not cap.isOpened():
            print("Error: Could not open video with any backend.")
            return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from video. Defaulting to 30.")
        fps = 30
    else:
        print(f"Video FPS: {fps}")
    
    desiredFrameTime = 1.0 / fps
        
    videoStartTime = gpsData[0]['timestamp'] # Approximate start time

    gpsDataIndex = 0
    perspMatrix = None

    i = 0
    while cap.isOpened():
        i += 1
        startTime = time.time()

        ret, frame = cap.read()
        if not ret:
            break        # --- Create largest centered square and resize to 640x640 ---
        
        if i % 2 == 0:
            continue
        
        height, width = frame.shape[:2]
        centerX = width // 2
        centerY = height // 2

        modelDimensions = (320, 320)  # Model expects 640x640 input
        
        # Determine the size of the largest possible centered square
        # This will be limited by the smallest dimension (height or width)
        cropSize = min(height, width)
        
        # Calculate the crop coordinates for the largest centered square
        xStart = centerX - cropSize // 2
        yStart = centerY - cropSize // 2
        
        # Crop the frame to get the largest possible square
        # croppedFrame = frame[yStart:yStart + cropSize, xStart:xStart + cropSize]
        
        # # Resize the square to 640x640
        # resizedFrame = cv2.resize(croppedFrame, modelDimensions, interpolation=cv2.INTER_AREA)
        resizedFrame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
        

        # Use the video's internal timestamp for more accurate synchronization
        currentVideoMs = cap.get(cv2.CAP_PROP_POS_MSEC)
        currentFrameTime = videoStartTime + timedelta(milliseconds=currentVideoMs)

        # Find the corresponding GPS data point
        while gpsDataIndex < len(gpsData) - 1 and gpsData[gpsDataIndex + 1]['timestamp'] <= currentFrameTime:
            gpsDataIndex += 1

        currentGpsPoint = gpsData[gpsDataIndex]

        # --- Overlay Text ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255) # White
        lineType = 1
        textX = 10
        textYStart = 30
        lineHeight = 25

        # Timestamp, GPS Fix and Precision
        fixType = currentGpsPoint.get('fixType', 'N/A')
        accuracy = currentGpsPoint.get('accuracy', 0)
        # Timestamp
        putTextWithOutline(resizedFrame, f"Time: {currentGpsPoint['timestamp'].strftime('%H:%M:%S.%f')[:-3]}", 
                (textX, textYStart), font, fontScale, fontColor, lineType)
        
        # GPS Fix 
        GPSColor = (0, 255, 0) if fixType == 3 else (0, 0, 255)  # Green for 3D fix, Red otherwise
        putTextWithOutline(resizedFrame, f"Fix: {fixType}", 
                (textX + 300, textYStart), font, fontScale, GPSColor, lineType)
        
        # Accuracy with color coding
        accuracyColor = (0, 255, 0) if accuracy <= 3 else (0, 0, 255)  # Green if ≤ 3, Red if > 3
        putTextWithOutline(resizedFrame, f"Accuracy: {accuracy}m", 
                (textX + 450, textYStart), font, fontScale, accuracyColor, lineType)
        # Latitude
        currentLat = currentGpsPoint['latitude']
        putTextWithOutline(resizedFrame, f"Latitude: {currentLat:.6f}", 
                    (textX, textYStart + lineHeight), font, fontScale, fontColor, lineType)
        # Longitude
        currentLon = currentGpsPoint['longitude']
        putTextWithOutline(resizedFrame, f"Longitude: {currentLon:.6f}", 
                    (textX, textYStart + 2 * lineHeight), font, fontScale, fontColor, lineType)
        # Altitude
        putTextWithOutline(resizedFrame, f"Altitude: {currentGpsPoint['altitude']:.1f} m", 
                    (textX, textYStart + 3 * lineHeight), font, fontScale, fontColor, lineType)
        # Speed 2D
        speed2dKph = currentGpsPoint['speed2d'] * 3.6
        putTextWithOutline(resizedFrame, f"Speed (2D): {currentGpsPoint['speed2d']:.1f} m/s ({speed2dKph:.1f} km/h)", 
                    (textX, textYStart + 4 * lineHeight), font, fontScale, fontColor, lineType)
        # Speed 3D
        speed3dKph = currentGpsPoint['speed3d'] * 3.6
        putTextWithOutline(resizedFrame, f"Speed (3D): {currentGpsPoint['speed3d']:.1f} m/s ({speed3dKph:.1f} km/h)", 
                    (textX, textYStart + 5 * lineHeight), font, fontScale, fontColor, lineType)

        # Acceleration
        acceleration = currentGpsPoint.get('acceleration', 0)
        accelerationKph2 = acceleration * 3.6  # Convert m/s^2 to km/h^2
        putTextWithOutline(resizedFrame, f"Acceleration: {acceleration:.1f} m.s-2 ({accelerationKph2:.1f} km.h-2)",
                    (textX, textYStart + 6 * lineHeight), font, fontScale, fontColor, lineType)

        # Heading
        heading = currentGpsPoint.get('heading', 0)
        putTextWithOutline(resizedFrame, f"Heading: {heading:.0f} deg",
                    (textX, textYStart + 7 * lineHeight), font, fontScale, fontColor, lineType)
        # Turn Rate
        turnRateDisplay = currentGpsPoint.get('turnRate', 0)
        putTextWithOutline(resizedFrame, f"Turn Rate: {turnRateDisplay:.0f} deg/s",
                    (textX, textYStart + 8 * lineHeight), font, fontScale, fontColor, lineType)
                      # --- Draw Future Path (Tentacle) on a Separate Frame and Warp it ---
        # Frame dimensions
        frameHeight, frameWidth, _ = resizedFrame.shape
        
        tentacleFrameHeight = frameHeight*3
        tentacleFrameWidth = frameWidth
        # Create a separate transparent frame for the tentacle/path
        tentacleFrame = np.zeros((tentacleFrameHeight, tentacleFrameWidth, 4), dtype=np.uint8)  # RGBA
        
        # Vector origin (bottom center)
        vectorWidth = 10
        originX = tentacleFrameWidth // 2
        originY = tentacleFrameHeight - vectorWidth
        
        # Number of future points to find (1 point -> 0.25 second into the future)
        numPoints = 12  # 3 seconds into the future
        secondsPerPoint = 3/numPoints
        
        # Scale factor for distance to pixels conversion
        scaleFactorPixelsPerMeter = 10
        
        # Get actual future GPS points from the data
        futureGpsPoints = []
        
        # Get the target time intervals
        targetTimes = []
        for i in range(1, numPoints + 1):
            targetTimes.append(currentFrameTime + timedelta(seconds=i * secondsPerPoint))
            
        # Find the closest GPS data points to our target times
        for targetTime in targetTimes:
            # Find the index of the GPS point closest to the target time
            closestIdx = gpsDataIndex
            minTimeDiff = abs((gpsData[closestIdx]['timestamp'] - targetTime).total_seconds())
            
            # Search forward in the GPS data
            searchIdx = gpsDataIndex + 1
            while searchIdx < len(gpsData):
                timeDiff = abs((gpsData[searchIdx]['timestamp'] - targetTime).total_seconds())
                if timeDiff < minTimeDiff:
                    minTimeDiff = timeDiff
                    closestIdx = searchIdx
                if gpsData[searchIdx]['timestamp'] > targetTime:
                    break
                searchIdx += 1
            
            # Only use points that are within 2 seconds of the target time
            if minTimeDiff <= 2.0:
                futureGpsPoints.append(gpsData[closestIdx])
        
        # Convert the future GPS points to relative distances and then to screen coordinates
        # aligned with current heading
        points = [(originX, originY)]  # Start with current position
        
        # Earth's radius in meters (approximate)
        earthRadius = 6371000
        
        for point in futureGpsPoints:
            # Calculate the distance and bearing from current position to the future point
            
            # Convert lat/lon to radians
            lat1 = math.radians(currentLat)
            lon1 = math.radians(currentLon)
            lat2 = math.radians(point['latitude'])
            lon2 = math.radians(point['longitude'])
            
            # Calculate distance (in meters)
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = earthRadius * c  # Distance in meters

            if distance <= 1:
                distance = max(0, 3*distance-2)
            
            # Calculate bearing to the future point
            x = math.sin(dlon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
            bearing = math.atan2(x, y)
            bearing = math.degrees(bearing)
            bearing = (bearing + 360) % 360  # Normalize to 0-360 degrees
            
            # Calculate the bearing relative to the current heading
            relativeBearing = bearing - heading
            relativeBearing = (relativeBearing + 360) % 360  # Normalize
            relativeBearingRad = math.radians(relativeBearing)
            
            # Forward is up (-y on screen), right is +x on screen
            screenX = originX + distance * scaleFactorPixelsPerMeter * math.sin(relativeBearingRad)
            screenY = originY - distance * scaleFactorPixelsPerMeter * math.cos(relativeBearingRad)
            points.append((int(screenX), int(screenY)))
            
        # Draw the connected segments on the tentacle frame with transparency
        for i in range(len(points) - 1):
            if i < len(futureGpsPoints):
                # Get speed from the GPS point and normalize it
                speed = futureGpsPoints[i]['speed2d']
                maxSpeed = currentGpsPoint.get('maxSpeed', 30)  # Use 30 m/s (~108 km/h) as default max speed
                
                # Normalize speed between 0 and 1
                normalizedSpeed = min(1.0, max(0.0, speed / maxSpeed))
                
                # Create color based on speed (blue->red gradient with alpha for transparency)
                # blue = int(255 * (1 - normalizedSpeed))
                # red = int(255 * normalizedSpeed)
                alpha = 255  # Transparency level (0-255)
                color = (251, 152, 52, alpha)  # BGRA format
                color = (152, 152, 152, alpha)  # BGRA format
                
                # Draw the line on the transparent tentacle frame
                cv2.line(tentacleFrame, points[i], points[i+1], color, vectorWidth, cv2.LINE_AA)
            
        # Define perspective transform points for road warping effect
        # Source points - the tentacle as drawn in 2D space

        verticalShift = -0.02 * tentacleFrameHeight  # Shift the tentacle up by 2% of the frame height

        tentacleWidth = 0.05 * tentacleFrameWidth
        tentacleStartHeight = 1 * tentacleFrameHeight
        bottomMargin = 0.0 * tentacleFrameHeight
        topMargin = 0.18 * tentacleFrameHeight

        srcPts = np.float32([
            [frameWidth/2 - tentacleWidth, tentacleStartHeight - bottomMargin + verticalShift],   # Bottom left
            [frameWidth/2 + tentacleWidth, tentacleStartHeight - bottomMargin + verticalShift],   # Bottom right
            [frameWidth/2 + tentacleWidth, tentacleStartHeight - topMargin + verticalShift],  # Top right
            [frameWidth/2 - tentacleWidth, tentacleStartHeight - topMargin + verticalShift]   # Top left
        ])

        # draw the source points on the resized frame for debugging
        for pt in srcPts:
            cv2.circle(tentacleFrame, tuple(pt.astype(int)), 5, (255, 0, 0), -1)


        topWidth = frameWidth * 0.04 # 15% of the frame width
        topHeight = frameHeight * 0.66 # higher is lower on the screen

        # margins
        bottomHorizontalMargin = -0.15 * frameWidth  # 5% of the frame width
        bottomVerticalMargin = -0.05 * frameHeight  # 3% of the frame height

        horizontalOffset = 0.01 * frameWidth  # 5% of the frame width

        # Destination points - how we want to warp it to look like it's on the road
        dstPts = np.float32([
            [bottomHorizontalMargin, frameHeight-bottomVerticalMargin],   # Bottom left
            [frameWidth-bottomHorizontalMargin, frameHeight-bottomVerticalMargin],  # Bottom right
            [frameWidth/2+topWidth + horizontalOffset, topHeight],    # Top right
            [frameWidth/2-topWidth + horizontalOffset, topHeight]  # Top left
        ])

        #/2
        resizedTentacleFrame = cv2.resize(tentacleFrame, (tentacleFrameWidth//2, tentacleFrameHeight//2), interpolation=cv2.INTER_AREA)
        cv2.imshow('Tentacle Path', resizedTentacleFrame)
        
        # draw the destination points on the resized frame for debugging
        for pt in dstPts:
            cv2.circle(resizedFrame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
        
        # Calculate the perspective transform matrix
        if perspMatrix is None:
            print("Calculating perspective transform matrix...")
            perspMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
        

        warpedTentacle = cv2.warpPerspective(tentacleFrame, perspMatrix, (frameWidth, frameHeight))

        # Extract BGR and alpha channels
        bgr = warpedTentacle[:, :, 0:3]
        alpha = warpedTentacle[:, :, 3] / 255.0  # Normalize alpha to 0-1
        
        # Expand alpha to 3 channels
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Blend the warped tentacle with the original frame
        resizedFrame = (resizedFrame * (1 - alpha) + bgr * alpha).astype(np.uint8)

        # --- Display Frame ---
        cv2.imshow('GPS Overlay', resizedFrame)

        # --- Timing and Exit ---
        # Calculate the time taken to process the frame and determine the necessary delay to maintain real-time playback.
        processingTime = time.time() - startTime
        waitTime = max(0, desiredFrameTime - processingTime - 0.003)
        
        time.sleep(max(0, waitTime))  # Sleep to maintain real-time playback


        print(f"frameTime: {processingTime:.3f} | waitTime: {waitTime:.3f} | FinalFrameTime: {processingTime + waitTime:.3f} | intendedFrameTime: {desiredFrameTime:.3f}", end='\r')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # videoPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GOPR5986_combined.MP4"
    videoPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP095986.MP4"
    main(videoPath)
