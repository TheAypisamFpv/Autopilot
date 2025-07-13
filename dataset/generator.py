import random
import cv2
import json
from matplotlib.dates import FR
import numpy as np
import os
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

# Set environment variable to increase read attempts for potential network issues with large video files
# os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192'

def loadGpsData(jsonPath: str) -> List[Dict[str, Any]]:
    """
    Load GPS data from a JSON file.

    Args:
        jsonPath: Path to the JSON file containing GPS data

    Returns:
        List of dictionaries containing GPS data points
    """
    print(f'Loading GPS data from "{jsonPath}"... ', end='')
    try:
        with open(jsonPath, 'r') as f:
            data = json.load(f)
        
        # Convert timestamps from string to datetime objects
        for point in data:
            point['timestamp'] = datetime.fromisoformat(point['timestamp'])
        
        print("done.")
        return data
    except Exception as e:
        print(f"\nError loading GPS data: {e}")
        return []

def findFutureGpsPoints(gpsData: List[Dict[str, Any]], currentFrameTime: datetime, duration: float = 3.0, interval: float = 0.5) -> List[Dict[str, Any]]:
    """
    Find future GPS points based on the current frame time and specified duration and interval.

    Args:
        gpsData: List of GPS data points
        currentFrameTime: Timestamp of the current frame
        duration: Duration in seconds for trajectory calculation (default: 3.0)
        interval: Time interval between trajectory points in seconds (default: 0.5)

    Returns:
        List of future GPS data points
    """
    # Number of future points to find
    numPoints = int(duration / interval)
    
    # Generate target times
    targetTimes = []
    for i in range(1, numPoints + 1):
        targetTimes.append(currentFrameTime + timedelta(seconds=i * interval))
    
    # Find the actual GPS points at those future times
    futureGpsPoints = []
    
    # Find the current GPS data index to start searching from
    startIdx = 0
    for i, point in enumerate(gpsData):
        if point['timestamp'] >= currentFrameTime:
            startIdx = i
            break
    
    # Find the closest GPS data points to our target times
    for targetTime in targetTimes:
        # Start with the closest known point
        closestIdx = startIdx
        minTimeDiff = abs((gpsData[closestIdx]['timestamp'] - targetTime).total_seconds())
        
        # Search forward in the GPS data
        searchIdx = startIdx + 1
        while searchIdx < len(gpsData):
            timeDiff = abs((gpsData[searchIdx]['timestamp'] - targetTime).total_seconds())
            if timeDiff < minTimeDiff:
                minTimeDiff = timeDiff
                closestIdx = searchIdx
            # If we've gone past the target time by more than 2 seconds, stop searching
            if gpsData[searchIdx]['timestamp'] > targetTime + timedelta(seconds=2):
                break
            searchIdx += 1

        # Only use points that are within 0.5 seconds of the target time
        if minTimeDiff <= 0.5:
            futureGpsPoints.append(gpsData[closestIdx])
        else:
            futureGpsPoints.append(None)
            
    return futureGpsPoints

def isGpsDataValid(gpsPoint: Dict[str, Any], gpsData: List[Dict[str, Any]], currentFrameTime: datetime) -> bool:
    """
    Validate the GPS data point and its future trajectory.

    Args:
        gpsPoint: Current GPS data point to validate
        gpsData: List of all GPS data points
        currentFrameTime: Timestamp of the current frame

    Returns:
        A list containing:
        - Boolean indicating if the GPS data is valid
        - Fix type of the GPS point
        - Accuracy of the GPS point
    """
    # Check current point
    if gpsPoint is None:
        return [False, 0, float('inf')]
    
    Fix = gpsPoint.get('fixType', 0)
    Accuracy = round(gpsPoint.get('accuracy', float('inf')), 1)  # Round accuracy to 1 decimal place
    if not (Fix == 3 and Accuracy < 3):
        return [False, Fix, Accuracy]

    # Find future points
    futureGpsPoints = findFutureGpsPoints(gpsData, currentFrameTime)

    # If there are no future points found, we can't validate the future path.
    if not futureGpsPoints:
        return [False, Fix, Accuracy]

    # Check if at least one future point is valid
    pointValidity = []
    for point in futureGpsPoints:
        if point is None:
            pointValidity.append([False, 0, float('inf')])
            continue
        
        # Check fixType and accuracy for each future point
        Fix = point.get('fixType', 0)
        Accuracy = point.get('accuracy', float('inf'))
        if Fix == 3 and Accuracy < 3:
            pointValidity.append([True, Fix, Accuracy])
        else:
            pointValidity.append([False, Fix, Accuracy])
            
    # check if all future points are valid, and average the fixType and accuracy
    avgFixType = round(sum(validPoint[1] for validPoint in pointValidity) / len(pointValidity)) # rounded to the nearest integer
    avgAccuracy = round(sum(validPoint[2] for validPoint in pointValidity) / len(pointValidity), 1) # rounded to 1 decimal place

    allValid = all(validPoint[0] for validPoint in pointValidity)
    if allValid:
        return [True, avgFixType, avgAccuracy]
    else:
        return [False, avgFixType, avgAccuracy]

def findGpsPointAtTime(gpsData: List[Dict[str, Any]], targetTime: datetime) -> Tuple[Dict[str, Any], int]:
    """
    Find the GPS data point closest to the target time.

    Args:
        gpsData: List of GPS data points
        targetTime: Target timestamp to find

    Returns:
        A tuple containing:
        - GPS data point closest to the target time (or None)
        - Index of the data point (or -1)
    """
    if not gpsData:
        return None, -1
    
    closestIdx = 0
    minTimeDiff = abs((gpsData[0]['timestamp'] - targetTime).total_seconds())
    
    for i, point in enumerate(gpsData):
        timeDiff = abs((point['timestamp'] - targetTime).total_seconds())
        if timeDiff < minTimeDiff:
            minTimeDiff = timeDiff
            closestIdx = i
    
    # Only return the point if it's within 0.5 seconds of the target time
    if minTimeDiff <= 0.5:
        return gpsData[closestIdx], closestIdx
    
    return None, -1

def calculateFutureTrajectory(gpsData: List[Dict[str, Any]], currentPoint: Dict[str, Any], 
                             currentFrameTime: datetime,
                             duration: float, interval: float) -> List[Dict[str, Any]]:
    """
    Calculate future trajectory vectors based on GPS data and current position.
    
    Args:
        gpsData: List of GPS data points
        currentPoint: Current GPS data point
        currentFrameTime: Timestamp of the current frame
        duration: Duration in seconds for trajectory calculation
        interval: Time interval between trajectory points in seconds
        
    Returns:
        List of dictionaries containing trajectory vectors
    """
    # Earth's radius in meters
    earthRadius = 6371000
    
    # Current position
    currentLat = currentPoint['latitude']
    currentLon = currentPoint['longitude']
    currentHeading = currentPoint.get('heading', 0)
    
    # Find the actual GPS points at those future times
    futureGpsPoints = findFutureGpsPoints(gpsData, currentFrameTime, duration, interval)
    
    # Process the future points to calculate trajectory vectors
    trajectoryVectors = []
    
    for futurePoint in futureGpsPoints:
        if futurePoint is None:
            trajectoryVectors.append(None)
            continue
        
        # Convert lat/lon to radians
        lat1 = math.radians(currentLat)
        lon1 = math.radians(currentLon)
        lat2 = math.radians(futurePoint['latitude'])
        lon2 = math.radians(futurePoint['longitude'])

        currentLat = futurePoint['latitude']
        currentLon = futurePoint['longitude']
        
        # Calculate distance (in meters)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = earthRadius * c  # Distance in meters
        
        # For very short distances, apply a small correction like in GPSTestViewer
        # under 1m of displacment, this fonction is applied:
        # -log10(1-0.9x)x
        if distance <= 1:
            distance = -math.log10(1 - 0.9 * distance) * distance
        
        # Calculate bearing to the future point
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to 0-360 degrees
        
        # Calculate the bearing relative to the current heading
        relativeBearing = bearing - currentHeading
        relativeBearing = (relativeBearing + 360) % 360  # Normalize
        
        # Convert to -180 to +180 format
        if relativeBearing > 180:
            relativeBearing -= 360
            
        # Calculate time difference from current point
        timeDiff = (futurePoint['timestamp'] - currentFrameTime).total_seconds()
        
        # Create a vector with x (lateral) and y (longitudinal) components
        # x is positive to the right, y is positive forward
        xComponent = distance * math.sin(math.radians(relativeBearing))
        yComponent = distance * math.cos(math.radians(relativeBearing))
        
        trajectoryVectors.append({
            'time': timeDiff,  # Actual time difference in seconds
            'x': xComponent,  # meters, lateral displacement
            'y': yComponent,  # meters, longitudinal displacement
        })
    
    return trajectoryVectors


def getRandomizeFrameParams(deltaExpo: float = 0.2, deltaGamma: float = 0.2,
                     deltaBrightness: float = 0.2, deltaContrast: float = 0.2) -> np.ndarray:
    """
    Randomize frame parameters such as exposure, gamma, brightness, and contrast.

    Args:
        deltaExpo: Maximum exposure adjustment factor (default: 0.2)
        deltaGamma: Maximum gamma adjustment factor (default: 0.2)
        deltaBrightness: Maximum brightness adjustment factor (default: 0.2)
        deltaContrast: Maximum contrast adjustment factor (default: 0.2)

    Returns:
        frameParams: Dictionary containing randomized parameters for the frame
    """
    # Random exposure adjustment
    expoFactor = 1 + random.uniform(-deltaExpo, deltaExpo)
    # Random gamma adjustment
    gammaFactor = 1 + random.uniform(-deltaGamma, deltaGamma)
    # Random brightness adjustment
    brightnessFactor = random.uniform(-deltaBrightness, deltaBrightness) * 255
    # Random contrast adjustment
    contrastFactor = 1 + random.uniform(-deltaContrast, deltaContrast)
    # Clip values to ensure they are within valid range

    # frameParams
    frameParams = {
        'expoFactor': expoFactor,
        'gammaFactor': gammaFactor,
        'brightnessFactor': brightnessFactor,
        'contrastFactor': contrastFactor
    }

    return frameParams


def applyFrameParams(frame: np.ndarray, frameParams: Dict[str, float]) -> np.ndarray:
    """
    Apply randomized frame parameters to the input frame.
    
    Args:
        frame: Input frame to be processed
        frameParams: Dictionary containing randomized parameters for the frame

    Returns:
        Processed frame with applied parameters
    """
    if frame is None:
        return None
    
    # Apply exposure adjustment
    frame = cv2.convertScaleAbs(frame, alpha=frameParams['expoFactor'], beta=0)
    
    # Apply gamma correction
    gamma = frameParams['gammaFactor']
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)], dtype=np.uint8)
    frame = cv2.LUT(frame, table)
    
    # Apply brightness and contrast adjustments
    brightness = frameParams['brightnessFactor']
    contrast = frameParams['contrastFactor']
    
    # Adjust brightness and contrast
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    return frame


def saveDatasetItem(outputDir: str, index: int, prevFrame: np.ndarray, currentFrame: np.ndarray, 
                   frameParams: Dict[str, float], vectors: List[Dict[str, Any]], speed: float, acceleration: float, turnRate: float):
    """
    Save dataset item (frames and metadata)
    
    Args:
        outputDir: Output directory for the dataset
        index: Index of the dataset item
        prevFrame: Previous frame
        currentFrame: Current frame
        frameParams: Dictionary containing randomized parameters for the frame
        vectors: List of trajectory vectors
        speed: Current speed in m/s
        acceleration: Current acceleration in m/s²
        turnRate: Current turn rate in deg/s
    """
    # Create directories if they don't exist
    imagesDir = os.path.join(outputDir, "images")
    os.makedirs(imagesDir, exist_ok=True)

    labelsDir = os.path.join(outputDir, "labels")
    os.makedirs(labelsDir, exist_ok=True)
    
    # Save images
    prevImagePath = os.path.join(imagesDir, f"{index:06d}_prev.png")
    currentImagePath = os.path.join(imagesDir, f"{index:06d}_current.png")

    # Apply frame parameters to the frames
    prevFrame = applyFrameParams(prevFrame, frameParams)
    currentFrame = applyFrameParams(currentFrame, frameParams)

    
    cv2.imwrite(prevImagePath, prevFrame)
    cv2.imwrite(currentImagePath, currentFrame)
    
    # Filter out None vectors
    validVectors = [v for v in vectors if v is not None]
    
    # Create text file with metadata
    metadataPath = os.path.join(labelsDir, f"{index:06d}.txt")

    with open(metadataPath, 'w') as f:
        # Write vectors
        f.write("vectors : ")
        if validVectors:
            vectorsStr = " ".join([f"{v['x']:.3f},{v['y']:.3f}" for v in validVectors])
            f.write(vectorsStr)
        else:
            f.write("None")
        f.write("\n")
        
        # Write speed, acceleration, and turn rate
        f.write(f"speed : {speed:.3f}\n")
        f.write(f"acceleration : {acceleration:.3f}\n")
        f.write(f"turnRate : {turnRate:.3f}\n")


def visualizeFutureTrajectory(frame: np.ndarray, previousFrame: np.ndarray, vectors: List[Dict[str, Any]], speed: float, acceleration: float, turnRate: float, frameParams: Dict[str, float]):
    """
    Visualize the future trajectory on the given frame.

    Args:
        frame: The current video frame.
        previousFrame: The previous video frame.
        vectors: List of future trajectory vectors.
        speed: Current speed in m/s.
        acceleration: Current acceleration in m/s².
        turnRate: Current turn rate in deg/s.
        frameParams: Dictionary containing randomized parameters for the frame.
    """

    if not frame.all() or not previousFrame.all():
        print("No frame or previous frame to visualize.")
        return

    if not vectors:
        print("No vectors to visualize.")
        return

    # Apply frame parameters
    currentFrame = frame
    previousFrame = previousFrame

    currentFrame = applyFrameParams(currentFrame, frameParams)
    previousFrame = applyFrameParams(previousFrame, frameParams)

    originPoint = (currentFrame.shape[1] // 2, currentFrame.shape[0] -5)  # Center of the frame

    vecToPixel = 3
    vectorThickness = 2

    colorBlend = 255 / len(vectors)

    cv2.line(currentFrame, originPoint, (originPoint[0]+100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"0.00s",
                    (originPoint[0] + 10, originPoint[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(currentFrame, f"speed = {speed:.1f} m.s-1",
                    (10, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"acc = {acceleration:.1f} m.s-2",
                    (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"turnRate = {turnRate:.1f} deg.s-1",
                    (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for i, vector in enumerate(vectors):
        if vector is None:
            continue
        
        # Calculate the end point of the vector
        endX = int(originPoint[0] + vector['x'] * vecToPixel)
        endY = int(originPoint[1] - vector['y'] * vecToPixel)

        # Draw the vector
        cv2.line(currentFrame, originPoint, (endX, endY), (255 - i*colorBlend, 0, i*colorBlend), vectorThickness, cv2.LINE_AA)
        originPoint = (endX, endY)  # Update origin for the next vector
        
        # Alternate between left and right side for the black line and text
        if i % 2 != 0:
            # Draw on right
            cv2.line(currentFrame, originPoint, (originPoint[0]+100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(currentFrame, f"{vector['time']:.2f}s",
                        (originPoint[0] + 10, originPoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Draw on left
            cv2.line(currentFrame, originPoint, (originPoint[0]-100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(currentFrame, f"{vector['time']:.2f}s",
                        (originPoint[0] - 100, originPoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # show both image (previous on the left, current on the right) on the same window
    combinedFrame = np.hstack((previousFrame, currentFrame))
    cv2.imshow("Future Trajectory Visualization", combinedFrame)
    cv2.waitKey(1)


def generateDataset(videoPath: str, outputDir: str, frameInterval: int, startIndex: int, vectorsNumbers: int, vectorTimeWindow: float, temporalContextTimeWindow: float, DEBUGVIZ: bool = False):
    """
    Generate dataset by processing video frames and corresponding GPS data
    
    Args:
        videoPath: Path to the input video file
        outputDir: Directory to save the generated dataset
        frameInterval: Interval in seconds to skip frames (0 for no skipping)
        startIndex: Starting index for dataset items
        vectorsNumbers: Number of future trajectory vectors to generate
        vectorTimeWindow: Time window in seconds for each vector
        temporalContextTimeWindow: Time window in seconds for temporal context
        DEBUGVIZ: Whether to visualize the future trajectory (default: False)
    """
    jsonPath = videoPath.replace('.MP4', '.json')
    
    if not os.path.exists(videoPath):
        print(f"Error: Video file not found at {videoPath}")
        return startIndex
    if not os.path.exists(jsonPath):
        print(f"Error: JSON file not found at {jsonPath}")
        return startIndex

    
    # Load GPS data
    gpsData = loadGpsData(jsonPath)
    bestGpsFixType = 3
    maxGpsAccuracy = 3.0
    if not gpsData:
        print("Error: No GPS data found or failed to load")
        return startIndex
    
    # Create output directory
    os.makedirs(outputDir, exist_ok=True)
    
    # Use the MSMF backend on Windows, which can be more stable
    cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print("Error: Could not open video with MSMF backend, trying default...")
        cap = cv2.VideoCapture(videoPath)  # Fallback to default
        if not cap.isOpened():
            print("Error: Could not open video with any backend.")
            return startIndex
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from video. Defaulting to 30.")
        fps = 30
    else:
        print(f"Video FPS: {fps}")

    videoFrameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    videoDuration = videoFrameCount / fps
    
    # Get approximate start time of the video
    videoStartTime = gpsData[0]['timestamp']
    
    frameCount = 0
    datasetIndex = startIndex  # Start index for dataset items
    frameBuffer = []  # Buffer to store the last 1s of images
    frameBufferSize = max(int(fps * temporalContextTimeWindow), 2)  # Number of frames to keep in the buffer (at least 2 frames)

    gpsDataIndex = 0 # Start from the beginning of the GPS data

    while cap.isOpened():
        currentVideoMs = cap.get(cv2.CAP_PROP_POS_MSEC)
        if (currentVideoMs / 1000) >= (videoDuration - vectorTimeWindow):
            print(f"\nStopping generation {vectorTimeWindow}s before the end of the video.")
            break

        ret, frame = cap.read()

        # Scale image down to 270p (480x270)
        newHeight = 270
        newWidth = int(frame.shape[1] * (newHeight / frame.shape[0]))
        processedFrame = cv2.resize(frame, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

        frameParams = getRandomizeFrameParams(deltaExpo=0.15, deltaGamma=0.15, deltaBrightness=0.15, deltaContrast=0.15)

        if not ret:
            break
        
        # Skip frames based on the specified interval
        if frameInterval > 0:
            if frameCount % (fps * frameInterval) != 0:
                frameBuffer = []
                frameCount += 1
                print(f"Skipping frame {frameCount:06d}                    ", end='\r')
                continue
        
        # Get the current video timestamp
        currentVideoMs = cap.get(cv2.CAP_PROP_POS_MSEC)
        currentFrameTime = videoStartTime + timedelta(milliseconds=currentVideoMs)
        
        # Find corresponding GPS data point
        currentGpsPoint, currentGpsIndex = findGpsPointAtTime(gpsData, currentFrameTime)
        
        # If no GPS data found for this frame or data is invalid, skip
        GPDDataInfo = isGpsDataValid(currentGpsPoint, gpsData, currentFrameTime)
        if not currentGpsPoint or not GPDDataInfo[0]:
            # Search for the next valid GPS point from the current position
            nextValidGpsIndex = None
            # Start searching from the last known good point, or the start.
            searchStartIndex = gpsDataIndex
            if currentGpsIndex > gpsDataIndex:
                searchStartIndex = currentGpsIndex + 1

            for i in range(searchStartIndex, len(gpsData)):
                point = gpsData[i]
                # Quick check for basic validity before the more expensive check
                if point.get('fixType', 0) == bestGpsFixType and point.get('accuracy', float('inf')) < maxGpsAccuracy:
                    # Full check for this point and its future trajectory
                    if isGpsDataValid(point, gpsData, point['timestamp'])[0]:
                        nextValidGpsIndex = i
                        break

            if nextValidGpsIndex is not None:
                # A valid GPS point was found, so we jump the video to that point in time
                nextValidGpsPoint = gpsData[nextValidGpsIndex]
                timeToJumpMs = (nextValidGpsPoint['timestamp'] - videoStartTime).total_seconds() * 1000
                
                if timeToJumpMs >= videoDuration * 1000:
                    print("\nNext valid GPS point is past the end of the video. Stopping.")
                    break

                print(f"Invalid GPS data at {currentFrameTime.strftime('%H:%M:%S')}. Skipping to next valid point at {nextValidGpsPoint['timestamp'].strftime('%H:%M:%S')}...{' '*20}")
                cap.set(cv2.CAP_PROP_POS_MSEC, timeToJumpMs)
                gpsDataIndex = nextValidGpsIndex
                frameBuffer = []  # Clear buffer after a jump
                frameCount += 1 # Increment frame count as we are skipping frames
                continue # Continue to the next loop iteration to read the new frame
            else:
                # No more valid GPS data found in the rest of the file
                print("\nNo more valid GPS data found in the video. Stopping.")
                break

        print(f"Processing frame {frameCount:06d} - {datasetIndex:06d}{' '*60}", end='\r')
        
        if currentGpsIndex != -1:
            gpsDataIndex = currentGpsIndex
        
        # If the frame buffer is not full, just append the processed frame
        if len(frameBuffer) < frameBufferSize:
            frameBuffer.append(processedFrame)
            frameCount += 1
            continue

        previousFrame = frameBuffer[0]

        # Calculate future trajectory
        interval = vectorTimeWindow / vectorsNumbers
        futureTrajectory = calculateFutureTrajectory(gpsData, currentGpsPoint, currentFrameTime, duration=vectorTimeWindow, interval=interval)

        # Get the acceleration in the GPS
        acceleration = round(currentGpsPoint.get('acceleration', 0.0), 1)

        if DEBUGVIZ:
            visualizeFutureTrajectory(processedFrame, previousFrame, futureTrajectory, currentGpsPoint['speed2d'], acceleration, currentGpsPoint.get('turnRate', 0), frameParams)

        # Save dataset item
        saveDatasetItem(
            outputDir, 
            datasetIndex, 
            previousFrame, 
            processedFrame, 
            frameParams,
            futureTrajectory,
            currentGpsPoint['speed2d'],
            acceleration,
            currentGpsPoint.get('turnRate', 0)
        )
        
        # Update for next iteration
        frameBuffer.append(processedFrame)
        frameBuffer = frameBuffer[-frameBufferSize:]  # Keep only the last second of frames
        datasetIndex += 1
        
        frameCount += 1
    
    cap.release()
    print(f"Dataset generation complete. Generated {datasetIndex - startIndex} items. {' '*10}\n")

    return datasetIndex  # Return the last index used for further processing if needed


def main(Path: str, outputDir: str, frameInterval: int, vectorsNumbers: int, vectorTimeWindow: float, temporalContextTimeWindow: float, manualStartIndex = False, DEBUGVIZ:bool = False):
    """
    Main function to generate dataset from a video file or directory of video files.
    Args:
    - Path (str): Path to the video file or directory.
    - outputDir (str): Directory to save the generated dataset.
    - frameInterval (int): Interval between each frame sample, in seconds.
    - vectorsNumbers (int): Number of vectors to generate for each frame.
    - vectorTimeWindow (float): Time window for future trajectory prediction, in seconds.
    - temporalContextTimeWindow (float): Time window for temporal context, in seconds.
    - manualStartIndex (int, optional): Starting index for dataset items, can be adjusted if resuming from a previous run.
    - DEBUGVIZ (bool, optional): Flag to enable debug visualization.
    """
    startIndex = manualStartIndex if manualStartIndex is not False else 0
    # check if the provided path is a video file or a directory
    if os.path.isfile(Path):
        # If it's a file, process it directly
        startIndex = generateDataset(Path, outputDir, frameInterval, startIndex, vectorsNumbers, vectorTimeWindow, temporalContextTimeWindow, DEBUGVIZ=DEBUGVIZ)
    elif os.path.isdir(Path):
        # If it's a directory, process all MP4 files in it
        for filename in os.listdir(Path):
            if filename.endswith('.MP4'):
                videoPath = os.path.join(Path, filename)
                startIndex = generateDataset(videoPath, outputDir, frameInterval, startIndex, vectorsNumbers, vectorTimeWindow, temporalContextTimeWindow, DEBUGVIZ=DEBUGVIZ)
    else:
        print(f"Error: {Path} is neither a file nor a directory.")

    print(f"Total dataset items generated: {startIndex}\n")
    


if __name__ == "__main__":
    # Hardcoded values - no arguments or parser
    videoPath = r"D:\VS_Python_Project\Autopilot\Autopilot\Test_drive\2025.06.25"
    outputDir = r"F:\VS_Python_Project\Autopilot\Autopilot\dataset\output"
    frameInterval = 0 # interval between each frame sample, in seconds
    vectorsNumbers = 12
    vectorTimeWindow = 3.0 # seconds

    temporalContextTimeWindow = 0.1 # seconds

    manualStartIndex = 389376  # Starting index for dataset items, can be adjusted if resuming from a previous run

    debugViz = True
    
    main(videoPath, outputDir, frameInterval, vectorsNumbers, vectorTimeWindow, temporalContextTimeWindow, manualStartIndex, DEBUGVIZ=debugViz)
