import random
import cv2
import json
from matplotlib.dates import FR
import numpy as np
import os
import math
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp
import time

# Set environment variable to increase read attempts for potential network issues with large video files
# os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192'


def buildNonUniformTimeOffsets(vectorCount: int, totalTime: float) -> List[float]:
    if vectorCount == 12 and abs(totalTime - 3.0) < 1e-6:
        timeDeltas = [0.1] * 6 + [0.25] * 4 + [0.7] * 2
    else:
        step = totalTime / max(1, vectorCount)
        timeDeltas = [step] * vectorCount

    timeOffsets = []
    running = 0.0
    for delta in timeDeltas:
        running += float(delta)
        timeOffsets.append(running)

    if timeOffsets:
        scale = totalTime / max(1e-6, timeOffsets[-1])
        if abs(scale - 1.0) > 1e-6:
            timeOffsets = [t * scale for t in timeOffsets]

    return [round(t, 4) for t in timeOffsets]

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


def loadEgomotionParquet(parquetPath: str) -> Dict[str, np.ndarray]:
    """
    Load egomotion data from a parquet file into numpy arrays.

    Args:
        parquetPath: Path to the egomotion parquet file

    Returns:
        Dictionary of numpy arrays for each egomotion column
    """
    table = pq.read_table(parquetPath)
    data = {name: table.column(name).to_numpy() for name in table.schema.names}
    return data


def loadCameraTimestampsParquet(parquetPath: str) -> np.ndarray:
    """
    Load camera frame timestamps from parquet.

    Args:
        parquetPath: Path to the camera timestamps parquet

    Returns:
        Numpy array where index = frame_index and value = timestamp (int64)
    """
    table = pq.read_table(parquetPath)
    frame_index = table.column('frame_index').to_numpy()
    timestamp = table.column('timestamp').to_numpy()

    max_index = int(frame_index.max()) if len(frame_index) else -1
    timestamps_by_frame = np.full(max_index + 1, np.iinfo(np.int64).min, dtype=np.int64)
    timestamps_by_frame[frame_index] = timestamp
    return timestamps_by_frame


def getYawFromQuaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Compute yaw (heading) in radians from a quaternion.
    """
    sinyCosp = 2.0 * (qw * qz + qx * qy)
    cosyCosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sinyCosp, cosyCosp)



def findClosestIndex(timestamps: np.ndarray, target: int) -> int:
    """
    Find index of closest timestamp to target using binary search.
    """
    if len(timestamps) == 0:
        return -1
    idx = int(np.searchsorted(timestamps, target))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1

    before = timestamps[idx - 1]
    after = timestamps[idx]
    if abs(target - before) <= abs(after - target):
        return idx - 1
    return idx


def interpolateEgomotionState(egoData: Dict[str, np.ndarray], targetTimeUs: int) -> Dict[str, float]:
    """
    Linearly interpolate egomotion state at a target timestamp.
    Quaternion is interpolated with normalized lerp (nlerp).
    """
    timestamps = egoData['timestamp']
    if len(timestamps) < 2:
        return None

    idx = int(np.searchsorted(timestamps, targetTimeUs))
    if idx <= 0 or idx >= len(timestamps):
        return None

    t0 = int(timestamps[idx - 1])
    t1 = int(timestamps[idx])
    if t1 == t0:
        alpha = 0.0
    else:
        alpha = (targetTimeUs - t0) / float(t1 - t0)

    def lerp(a: float, b: float) -> float:
        return (1.0 - alpha) * a + alpha * b

    q0 = np.array([
        float(egoData['qx'][idx - 1]),
        float(egoData['qy'][idx - 1]),
        float(egoData['qz'][idx - 1]),
        float(egoData['qw'][idx - 1]),
    ])
    q1 = np.array([
        float(egoData['qx'][idx]),
        float(egoData['qy'][idx]),
        float(egoData['qz'][idx]),
        float(egoData['qw'][idx]),
    ])

    q = (1.0 - alpha) * q0 + alpha * q1
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm
    else:
        q = q0

    return {
        'timestamp': float(targetTimeUs),
        'x': lerp(float(egoData['x'][idx - 1]), float(egoData['x'][idx])),
        'y': lerp(float(egoData['y'][idx - 1]), float(egoData['y'][idx])),
        'z': lerp(float(egoData['z'][idx - 1]), float(egoData['z'][idx])),
        'vx': lerp(float(egoData['vx'][idx - 1]), float(egoData['vx'][idx])),
        'vy': lerp(float(egoData['vy'][idx - 1]), float(egoData['vy'][idx])),
        'vz': lerp(float(egoData['vz'][idx - 1]), float(egoData['vz'][idx])),
        'ax': lerp(float(egoData['ax'][idx - 1]), float(egoData['ax'][idx])),
        'ay': lerp(float(egoData['ay'][idx - 1]), float(egoData['ay'][idx])),
        'az': lerp(float(egoData['az'][idx - 1]), float(egoData['az'][idx])),
        'curvature': lerp(float(egoData['curvature'][idx - 1]), float(egoData['curvature'][idx])),
        'qx': float(q[0]),
        'qy': float(q[1]),
        'qz': float(q[2]),
        'qw': float(q[3]),
    }


def calculateFutureTrajectoryEgomotion(
    egoData: Dict[str, np.ndarray],
    currentTimeUs: int,
    targetTimesUs: List[int]
) -> List[Dict[str, Any]]:
    """
    Calculate future trajectory vectors from egomotion data.
    Vectors are incremental displacements between successive future points.
    """
    trajectoryVectors: List[Dict[str, Any]] = []

    currentState = interpolateEgomotionState(egoData, int(currentTimeUs))
    if currentState is None:
        return [None for _ in targetTimesUs]

    yaw = getYawFromQuaternion(currentState['qx'], currentState['qy'], currentState['qz'], currentState['qw'])
    currentX = currentState['x']
    currentY = currentState['y']
    current_ts = int(currentTimeUs)
    prevTimeUs = int(currentTimeUs)

    for targetTime in targetTimesUs:
        futureState = interpolateEgomotionState(egoData, int(targetTime))
        if futureState is None:
            trajectoryVectors.append(None)
            prevTimeUs = int(targetTime)
            continue

        futureX = futureState['x']
        futureY = futureState['y']

        dxWorld = futureX - currentX
        dyWorld = futureY - currentY

        forward = dxWorld * math.cos(yaw) + dyWorld * math.sin(yaw)
        right = dxWorld * math.sin(yaw) - dyWorld * math.cos(yaw)

        timeDiff = (int(targetTime) - current_ts) / 1e6

        trajectoryVectors.append({
            'time': timeDiff,
            'x': right,  # lateral displacement (right)
            'y': forward,  # longitudinal displacement (forward)
        })

        currentX = futureX
        currentY = futureY
        prevTimeUs = int(targetTime)

    return trajectoryVectors

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


def findFutureGpsPointsFromOffsets(gpsData: List[Dict[str, Any]], currentFrameTime: datetime, timeOffsets: List[float]) -> List[Dict[str, Any]]:
    if not gpsData:
        return []

    targetTimes = [currentFrameTime + timedelta(seconds=float(offset)) for offset in timeOffsets]

    startIdx = 0
    for i, point in enumerate(gpsData):
        if point['timestamp'] >= currentFrameTime:
            startIdx = i
            break

    futureGpsPoints = []
    for targetTime in targetTimes:
        closestIdx = startIdx
        minTimeDiff = abs((gpsData[closestIdx]['timestamp'] - targetTime).total_seconds())

        searchIdx = startIdx + 1
        while searchIdx < len(gpsData):
            timeDiff = abs((gpsData[searchIdx]['timestamp'] - targetTime).total_seconds())
            if timeDiff < minTimeDiff:
                minTimeDiff = timeDiff
                closestIdx = searchIdx
                searchIdx += 1
            else:
                break

        if minTimeDiff <= 0.5:
            futureGpsPoints.append(gpsData[closestIdx])
            startIdx = closestIdx
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


def calculateFutureTrajectoryFromOffsets(
    gpsData: List[Dict[str, Any]],
    currentPoint: Dict[str, Any],
    currentFrameTime: datetime,
    timeOffsets: List[float]
) -> List[Dict[str, Any]]:
    """
    Calculate future trajectory vectors based on GPS data and non-uniform time offsets.
    """
    earthRadius = 6371000
    currentLat = currentPoint['latitude']
    currentLon = currentPoint['longitude']
    currentHeading = currentPoint.get('heading', 0)

    futureGpsPoints = findFutureGpsPointsFromOffsets(gpsData, currentFrameTime, timeOffsets)

    trajectoryVectors = []

    for idx, futurePoint in enumerate(futureGpsPoints):
        if futurePoint is None:
            trajectoryVectors.append(None)
            continue

        lat1 = math.radians(currentLat)
        lon1 = math.radians(currentLon)
        lat2 = math.radians(futurePoint['latitude'])
        lon2 = math.radians(futurePoint['longitude'])

        currentLat = futurePoint['latitude']
        currentLon = futurePoint['longitude']

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earthRadius * c

        if distance <= 1:
            distance = -math.log10(1 - 0.9 * distance) * distance

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        bearing = (bearing + 360) % 360

        relativeBearing = (bearing - currentHeading + 360) % 360
        if relativeBearing > 180:
            relativeBearing -= 360

        timeDiff = float(timeOffsets[idx])
        xComponent = distance * math.sin(math.radians(relativeBearing))
        yComponent = distance * math.cos(math.radians(relativeBearing))

        trajectoryVectors.append({
            'time': timeDiff,
            'x': xComponent,
            'y': yComponent,
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
                   frameParams: Dict[str, float], vectors: List[Dict[str, Any]], speed: float, acceleration: float, turnRate: float,
                   saveImages: bool = True, videoPath: str = None, prevFrameIndex: int = None, currentFrameIndex: int = None):
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
        turnRate: Current turn rate in rad/s
    """
    labelsDir = os.path.join(outputDir, "labels")
    os.makedirs(labelsDir, exist_ok=True)
    
    if saveImages:
        # Create directories if they don't exist
        imagesDir = os.path.join(outputDir, "images")
        os.makedirs(imagesDir, exist_ok=True)

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
    if videoPath is not None and currentFrameIndex is not None:
        videoBase = os.path.basename(videoPath)
        videoUuid = videoBase.split(".camera_front_wide_120fov.mp4")[0]
        if videoUuid == videoBase:
            videoUuid = os.path.splitext(videoBase)[0]
        metadataFilename = f"{videoUuid}_{currentFrameIndex:06d}.txt"
    else:
        metadataFilename = f"{index:06d}.txt"

    metadataPath = os.path.join(labelsDir, metadataFilename)

    with open(metadataPath, 'w') as f:
        # Write vectors
        f.write("vectors : ")
        if validVectors:
            vectorsStr = " ".join([f"{v['x']:.3f},{v['y']:.3f}" for v in validVectors])
            f.write(vectorsStr)
        else:
            f.write("None")
        f.write("\n")

        f.write("vectorTimes : ")
        if validVectors:
            timeStr = " ".join([f"{v['time']:.3f}" for v in validVectors])
            f.write(timeStr)
        else:
            f.write("None")
        f.write("\n")
        
        # Write speed, acceleration, and turn rate
        f.write(f"speed : {speed:.3f}\n")
        f.write(f"acceleration : {acceleration:.3f}\n")
        f.write(f"turnRate : {turnRate:.3f}\n")

        # Optional video reference (labels-only mode)
        if videoPath is not None and prevFrameIndex is not None and currentFrameIndex is not None:
            f.write(f"video : {videoPath}\n")
            f.write(f"prevFrameIndex : {prevFrameIndex}\n")
            f.write(f"frameIndex : {currentFrameIndex}\n")


def visualizeFutureTrajectory(
    frame: np.ndarray,
    previousFrame: np.ndarray,
    vectors: List[Dict[str, Any]],
    speed: float,
    acceleration: float,
    turnRate: float,
    frameParams: Dict[str, float],
    worldPos: Tuple[float, float, float] = None,
    worldVel: Tuple[float, float, float] = None,
    windowName: str = "Future Trajectory Visualization"
):
    """
    Visualize the future trajectory on the given frame.

    Args:
        frame: The current video frame.
        previousFrame: The previous video frame.
        vectors: List of future trajectory vectors.
        speed: Current speed in m/s.
        acceleration: Current acceleration in m/s².
        turnRate: Current turn rate in rad/s.
        frameParams: Dictionary containing randomized parameters for the frame.
        worldPos: Optional world position (x, y, z) in meters.
        worldVel: Optional world velocity (vx, vy, vz) in m/s.
    """

    if frame is None or previousFrame is None:
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

    vecToPixel = 5
    vectorThickness = 2

    colorBlend = 255 / len(vectors)

    textColor = (255, 255, 255)
    cv2.line(currentFrame, originPoint, (originPoint[0]+100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"0.00s",
                    (originPoint[0] + 10, originPoint[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)

    cv2.putText(currentFrame, f"speed = {speed:.1f} m.s-1",
                    (10, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"acc = {acceleration:.1f} m.s-2",
                    (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)
    cv2.putText(currentFrame, f"turnRate = {turnRate:.1f} rad.s-1",
                    (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)

    infoY = 75
    if worldPos is not None:
        cv2.putText(currentFrame, f"pos = ({worldPos[0]:.2f}, {worldPos[1]:.2f}, {worldPos[2]:.2f}) m",
                        (10, infoY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)
        infoY += 20

    if worldVel is not None:
        cv2.putText(currentFrame, f"vel = ({worldVel[0]:.2f}, {worldVel[1]:.2f}, {worldVel[2]:.2f}) m/s",
                        (10, infoY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)

    vectorLengths = []
    for i, vector in enumerate(vectors):
        if vector is None:
            continue
        
        # Calculate the end point of the vector
        endX = int(originPoint[0] + vector['x'] * vecToPixel)
        endY = int(originPoint[1] - vector['y'] * vecToPixel)

        vectorLength = math.sqrt(vector['x'] * vector['x'] + vector['y'] * vector['y'])
        vectorLengths.append(vectorLength)

        # Draw the vector
        cv2.line(currentFrame, originPoint, (endX, endY), (255 - i*colorBlend, 0, i*colorBlend), vectorThickness, cv2.LINE_AA)
        originPoint = (endX, endY)  # Update origin for the next vector
        
        # Alternate between left and right side for the black line and text
        if i % 2 != 0:
            # Draw on right
            cv2.line(currentFrame, originPoint, (originPoint[0]+100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(currentFrame, f"{vector['time']:.2f}s",
                        (originPoint[0] + 10, originPoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)
        else:
            # Draw on left
            cv2.line(currentFrame, originPoint, (originPoint[0]-100, originPoint[1]), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(currentFrame, f"{vector['time']:.2f}s",
                        (originPoint[0] - 100, originPoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)

    if vectorLengths:
        reversedLengths = list(reversed(vectorLengths))
        startY = currentFrame.shape[0] - 20 - (len(reversedLengths) * 14)
        totalCount = len(reversedLengths)
        for idx, length in enumerate(reversedLengths, start=1):
            labelIndex = totalCount - idx + 1
            cv2.putText(currentFrame, f"L{labelIndex}: {length:.2f}m",
                        (10, startY + (idx - 1) * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, textColor, 1, cv2.LINE_AA)

    # show both image (previous on the left, current on the right) on the same window
    combinedFrame = np.hstack((previousFrame, currentFrame))
    cv2.imshow(windowName, combinedFrame)
    cv2.waitKey(1)


def generateDataset(
    videoPath: str,
    outputDir: str,
    imageSize: Tuple[int, int],
    frameInterval: int,
    startIndex: int,
    vectorsNumbers: int,
    vectorTimeWindow: float,
    vectorTimeOffsets: Optional[List[float]],
    temporalContextTimeWindow: float,
    DEBUGVIZ: bool = False
    ):
    """
    Generate dataset by processing video frames and corresponding GPS data
    
    Args:
        videoPath: Path to the input video file
        outputDir: Directory to save the generated dataset
        imageSize: Tuple[int, int], Size to resize frames to (width, height)
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

        # Scale image down to specified size
        processedFrame = cv2.resize(frame, imageSize, interpolation=cv2.INTER_AREA)

        frameParams = getRandomizeFrameParams(deltaExpo=0.1, deltaGamma=0.1, deltaBrightness=0.1, deltaContrast=0.1)

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
            nextValidGpsIndex = -1
            # Start searching from the last known good point, or the start.
            searchStartIndex = gpsDataIndex
            if currentGpsIndex > gpsDataIndex:
                searchStartIndex = currentGpsIndex + 1

            for i in range(searchStartIndex, len(gpsData)):
                point = gpsData[i]
                # Quick check for basic validity before the more expensive check
                if point.get('fixType', 0) == 3 and point.get('accuracy', float('inf')) < 3:
                    # Full check for this point and its future trajectory
                    if isGpsDataValid(point, gpsData, point['timestamp'])[0]:
                        nextValidGpsIndex = i
                        break
            
            if nextValidGpsIndex != -1:
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
        timeOffsets = vectorTimeOffsets if vectorTimeOffsets else buildNonUniformTimeOffsets(vectorsNumbers, vectorTimeWindow)
        futureTrajectory = calculateFutureTrajectoryFromOffsets(gpsData, currentGpsPoint, currentFrameTime, timeOffsets)

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


def generateDatasetNvidiaClip(
    clipUuid: str,
    cameraDir: str,
    egomotionDir: str,
    outputDir: str,
    imageSize: Tuple[int, int],
    frameInterval: int,
    startIndex: int,
    vectorsNumbers: int,
    vectorTimeWindow: float,
    vectorTimeOffsets: Optional[List[float]],
    temporalContextTimeWindow: float,
    DEBUGVIZ: bool = False,
    labelsOnly: bool = True,
    exportPngs: bool = False,
    indexCounter: Optional[Any] = None,
    indexLock: Optional[Any] = None,
    progressDict: Optional[Any] = None,
    progressModulo: int = 10,
    verbose: bool = True,
    statusDict: Optional[Any] = None
) -> int:
    """
    Generate dataset from NVIDIA egomotion and camera clips.

    Args:
        clipUuid: Clip UUID used in filenames
        cameraDir: Directory containing camera mp4 and timestamps parquet
        egomotionDir: Directory containing egomotion parquet
        outputDir: Directory to save the generated dataset
        imageSize: Size to resize frames to (width, height)
        frameInterval: Interval in seconds to skip frames
        startIndex: Starting index for dataset items
        vectorsNumbers: Number of future trajectory vectors to generate
        vectorTimeWindow: Time window in seconds for each vector
        temporalContextTimeWindow: Time window in seconds for temporal context
        DEBUGVIZ: Whether to visualize the future trajectory
        labelsOnly: Whether to save labels without images
        exportPngs: Whether to save PNGs alongside labels when labelsOnly is True

    Returns:
        Updated dataset index
    """
    cameraVideoPath = os.path.join(cameraDir, f"{clipUuid}.camera_front_wide_120fov.mp4")
    cameraTimestampsPath = os.path.join(cameraDir, f"{clipUuid}.camera_front_wide_120fov.timestamps.parquet")
    egomotionPath = os.path.join(egomotionDir, f"{clipUuid}.egomotion.parquet")

    if not os.path.exists(cameraVideoPath) or not os.path.exists(cameraTimestampsPath) or not os.path.exists(egomotionPath):
        if verbose:
            print(f"Skipping {clipUuid}: missing camera or egomotion files.")
        return startIndex

    egoData = loadEgomotionParquet(egomotionPath)
    frameTimestamps = loadCameraTimestampsParquet(cameraTimestampsPath)

    if len(frameTimestamps) == 0:
        if verbose:
            print(f"Skipping {clipUuid}: empty camera timestamps.")
        return startIndex

    cap = cv2.VideoCapture(cameraVideoPath, cv2.CAP_MSMF)
    if not cap.isOpened():
        if verbose:
            print("Error: Could not open video with MSMF backend, trying default...")
        cap = cv2.VideoCapture(cameraVideoPath)
        if not cap.isOpened():
            if verbose:
                print(f"Error: Could not open video for clip {clipUuid}.")
            return startIndex

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        if verbose:
            print("Warning: Could not get FPS from video. Defaulting to 30.")
        fps = 30
    elif verbose:
        print(f"Video FPS: {fps}")

    frameCount = 0
    datasetIndex = startIndex
    generatedCount = 0
    frameBuffer: List[np.ndarray] = []
    frameIndexBuffer: List[int] = []
    frameBufferSize = max(int(fps * temporalContextTimeWindow), 2)

    maxTimestampUs = int(frameTimestamps.max())
    vectorTimeWindowUs = int(vectorTimeWindow * 1e6)
    skipModulo = int(round(fps * frameInterval)) if frameInterval > 0 else 0

    frameIndex = 0
    if progressDict is not None:
        progressDict[clipUuid] = 0
    if statusDict is not None:
        statusDict[clipUuid] = "active"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frameIndex >= len(frameTimestamps):
            break

        currentFrameTimestampUs = int(frameTimestamps[frameIndex])
        if currentFrameTimestampUs == np.iinfo(np.int64).min:
            frameIndex += 1
            frameCount += 1
            continue

        if currentFrameTimestampUs >= (maxTimestampUs - vectorTimeWindowUs):
            if verbose:
                print(f"\nStopping generation {vectorTimeWindow}s before the end of the clip {clipUuid}.")
            break

        processedFrame = cv2.resize(frame, imageSize, interpolation=cv2.INTER_AREA)
        frameParams = getRandomizeFrameParams(deltaExpo=0, deltaGamma=0, deltaBrightness=0, deltaContrast=0)

        if skipModulo > 0 and frameCount % skipModulo != 0:
            frameBuffer = []
            frameIndexBuffer = []
            frameCount += 1
            frameIndex += 1
            if verbose:
                print(f"Skipping frame {frameCount:06d}                    ", end='\r')
            continue

        currentState = interpolateEgomotionState(egoData, currentFrameTimestampUs)
        if currentState is None:
            frameIndex += 1
            frameCount += 1
            continue

        if verbose:
            print(f"Processing frame {frameCount:06d} - {datasetIndex:06d}{' '*60}", end='\r')
        if progressDict is not None and (frameCount % progressModulo == 0):
            progressDict[clipUuid] = frameCount

        if len(frameBuffer) < frameBufferSize:
            frameBuffer.append(processedFrame)
            frameIndexBuffer.append(frameIndex)
            frameCount += 1
            frameIndex += 1
            continue

        previousFrame = frameBuffer[0]
        prevFrameIndex = frameIndexBuffer[0]

        timeOffsets = vectorTimeOffsets if vectorTimeOffsets else buildNonUniformTimeOffsets(vectorsNumbers, vectorTimeWindow)
        targetTimesUs = [currentFrameTimestampUs + int(offset * 1e6) for offset in timeOffsets]

        futureTrajectory = calculateFutureTrajectoryEgomotion(egoData, currentFrameTimestampUs, targetTimesUs)

        vx = float(currentState['vx'])
        vy = float(currentState['vy'])
        speed = math.sqrt(vx * vx + vy * vy)

        ax = float(currentState['ax'])
        ay = float(currentState['ay'])
        acceleration = math.sqrt(ax * ax + ay * ay)

        curvature = float(currentState['curvature'])
        # FIXED: yaw-rate unit mismatch between train and inference
        # Use SI units: yaw-rate in radians per second (rad/s)
        turnRate = curvature * speed  # rad/s (SI units, consistent with inference)

        if DEBUGVIZ:
            workerIdentity = mp.current_process()._identity
            workerSuffix = f"_{workerIdentity[0] - 1}" if workerIdentity else ""
            visualizeFutureTrajectory(
                processedFrame,
                previousFrame,
                futureTrajectory,
                speed,
                acceleration,
                turnRate,
                frameParams,
                worldPos=(currentState['x'], currentState['y'], currentState['z']),
                worldVel=(currentState['vx'], currentState['vy'], currentState['vz']),
                windowName=f"Future Trajectory Visualization{workerSuffix}"
            )

        if indexCounter is not None and indexLock is not None:
            with indexLock:
                datasetIndex = int(indexCounter.value)
                indexCounter.value += 1
        saveDatasetItem(
            outputDir,
            datasetIndex,
            previousFrame,
            processedFrame,
            frameParams,
            futureTrajectory,
            speed,
            acceleration,
            turnRate,
            saveImages=(not labelsOnly) or exportPngs,
            videoPath=cameraVideoPath if labelsOnly else None,
            prevFrameIndex=prevFrameIndex if labelsOnly else None,
            currentFrameIndex=frameIndex if labelsOnly else None
        )

        frameBuffer.append(processedFrame)
        frameIndexBuffer.append(frameIndex)
        frameBuffer = frameBuffer[-frameBufferSize:]
        frameIndexBuffer = frameIndexBuffer[-frameBufferSize:]
        if indexCounter is None:
            datasetIndex += 1
        generatedCount += 1

        frameCount += 1
        frameIndex += 1

    cap.release()
    if progressDict is not None:
        progressDict[clipUuid] = frameCount
    if statusDict is not None:
        statusDict[clipUuid] = "done"
    if verbose:
        print(f"Dataset generation complete for {clipUuid}. Generated {generatedCount} items. {' '*10}\n")

    return datasetIndex


def processNvidiaClip(args: Tuple[Any, ...]) -> int:
    (
        clipUuid,
        cameraDir,
        egomotionDir,
        outputDir,
        imageSize,
        frameInterval,
        startIndex,
        vectorsNumbers,
        vectorTimeWindow,
        vectorTimeOffsets,
        temporalContextTimeWindow,
        DEBUGVIZ,
        labelsOnly,
        exportPngs,
        indexCounter,
        indexLock,
        progressDict,
        statusDict,
        processedClips,
        processedLock,
        verbose
    ) = args

    resultIndex = generateDatasetNvidiaClip(
        clipUuid,
        cameraDir,
        egomotionDir,
        outputDir,
        imageSize,
        frameInterval,
        startIndex,
        vectorsNumbers,
        vectorTimeWindow,
        vectorTimeOffsets,
        temporalContextTimeWindow,
        DEBUGVIZ=DEBUGVIZ,
        labelsOnly=labelsOnly,
        exportPngs=exportPngs,
        indexCounter=indexCounter,
        indexLock=indexLock,
        progressDict=progressDict,
        statusDict=statusDict,
        verbose=verbose
    )

    with processedLock:
        processedClips.value += 1

    return resultIndex


def main(
    Path: str,
    outputDir: str,
    imageSize: Tuple[int, int],
    frameInterval: int,
    vectorsNumbers: int,
    vectorTimeWindow: float,
    vectorTimeOffsets: Optional[List[float]],
    temporalContextTimeWindow: float,
    manualStartIndex = False,
    DEBUGVIZ:bool = False,
    labelsOnly: bool = True,
    exportPngs: bool = False,
    numWorkers: int = 0
    ):
    """
    Main function to generate dataset from a video file or directory of video files.
    Args:
    - Path (str): Path to the video file or directory.
    - outputDir (str): Directory to save the generated dataset.
    - imageSize (Tuple[int, int]): Size to resize frames to (width, height).
    - frameInterval (int): Interval between each frame sample, in seconds.
    - vectorsNumbers (int): Number of vectors to generate for each frame.
    - vectorTimeWindow (float): Time window for future trajectory prediction, in seconds.
    - temporalContextTimeWindow (float): Time window for temporal context, in seconds.
    - manualStartIndex (int, optional): Starting index for dataset items, can be adjusted if resuming from a previous run.
    - DEBUGVIZ (bool, optional): Flag to enable debug visualization.
    - labelsOnly (bool, optional): Save labels without images.
    - exportPngs (bool, optional): Save PNGs alongside labels when labelsOnly is True.
    """
    outputDir = os.path.join(outputDir, f"output_NVIDIA_{vectorsNumbers}_{vectorTimeWindow}_{temporalContextTimeWindow}_framesize{imageSize[0]}x{imageSize[1]}")
    startIndex = manualStartIndex if manualStartIndex is not False else 0

    # if that directory exists, add a number "(x)" at the end until a non-existing directory is found
    counter = 1
    while os.path.exists(outputDir):
        outputDir = os.path.join(outputDir + f"({counter})")
        counter += 1

    print(f"Output Directory: {outputDir}\n")

    os.makedirs(outputDir, exist_ok=True)

    # check if the provided path is a video file or a directory
    if os.path.isfile(Path):
        # If it's a file, process it directly with the legacy GPS pipeline
        startIndex = generateDataset(Path, outputDir, imageSize, frameInterval, startIndex, vectorsNumbers, vectorTimeWindow, vectorTimeOffsets, temporalContextTimeWindow, DEBUGVIZ=DEBUGVIZ)
    elif os.path.isdir(Path):
        cameraDir = os.path.join(Path, "camera", "camera_front_wide_120fov")
        egomotionDir = os.path.join(Path, "labels", "egomotion")

        if os.path.isdir(cameraDir) and os.path.isdir(egomotionDir):
            # NVIDIA dataset structure
            mp4Files = [f for f in os.listdir(cameraDir) if f.endswith(".camera_front_wide_120fov.mp4")]
            mp4Files.sort()

            clipUuids = [f.split(".camera_front_wide_120fov.mp4")[0] for f in mp4Files]
            totalClips = len(clipUuids)
            if totalClips == 0:
                print("No clips found to process.")
                return

            workerCount = numWorkers if numWorkers and numWorkers > 0 else max((os.cpu_count() or 2) - 1, 1)

            if workerCount > 1:
                manager = mp.Manager()
                progressDict = manager.dict()
                statusDict = manager.dict()
                indexCounter = manager.Value('i', startIndex)
                indexLock = manager.Lock()
                processedClips = manager.Value('i', 0)
                processedLock = manager.Lock()

                with mp.Pool(processes=workerCount) as pool:
                    argsList = [(
                        clipUuid,
                        cameraDir,
                        egomotionDir,
                        outputDir,
                        imageSize,
                        frameInterval,
                        startIndex,
                        vectorsNumbers,
                        vectorTimeWindow,
                        vectorTimeOffsets,
                        temporalContextTimeWindow,
                        DEBUGVIZ,
                        labelsOnly,
                        exportPngs,
                        indexCounter,
                        indexLock,
                        progressDict,
                        statusDict,
                        processedClips,
                        processedLock,
                        False
                    ) for clipUuid in clipUuids]

                    results = [pool.apply_async(processNvidiaClip, (args,)) for args in argsList]

                    startTime = time.time()
                    initialLabels = startIndex
                    lastLineLen = 0
                    while True:
                        doneCount = sum(1 for r in results if r.ready())

                        activeClips = [k for k, v in list(statusDict.items()) if v == "active"]
                        activeClips.sort()
                        lineParts = [f"Video {clipUuid[:8]}: {progressDict.get(clipUuid, 0):04d}" for clipUuid in activeClips]
                        if doneCount < totalClips:
                            missingSlots = max(workerCount - len(lineParts), 0)
                            if missingSlots:
                                lineParts.extend(["Video loading......."] * missingSlots)

                        labelsTotal = int(indexCounter.value)
                        processed = int(processedClips.value)
                        elapsed = max(time.time() - startTime, 1e-6)
                        totalFps = (labelsTotal - initialLabels) / elapsed
                        videosPerHour = processed * 3600.0 / elapsed

                        line = " - ".join(lineParts)
                        line += f" | Total labels: {labelsTotal:06d}"
                        line += f" - Videos: {processed}/{totalClips}"
                        line += f" - FPS: {totalFps:.1f}"
                        line += f" - VPH: {videosPerHour:.1f}"

                        pad = max(lastLineLen - len(line), 0)
                        print(line + (" " * pad), end='\r', flush=True)
                        lastLineLen = len(line)

                        if doneCount == totalClips:
                            break
                        time.sleep(0.1)

                    for r in results:
                        r.get()

                startIndex = int(indexCounter.value)
                print("".ljust(max(lastLineLen, 120)), end='\r')
            else:
                for clipUuid in clipUuids:
                    startIndex = generateDatasetNvidiaClip(
                        clipUuid,
                        cameraDir,
                        egomotionDir,
                        outputDir,
                        imageSize,
                        frameInterval,
                        startIndex,
                        vectorsNumbers,
                        vectorTimeWindow,
                        vectorTimeOffsets,
                        temporalContextTimeWindow,
                        DEBUGVIZ=DEBUGVIZ,
                        labelsOnly=labelsOnly,
                        exportPngs=exportPngs,
                        verbose=True
                    )
        else:
            # Legacy dataset structure: recursively process all MP4 files
            for root, dirs, files in os.walk(Path):
                for filename in files:
                    if filename.endswith('.MP4'):
                        videoPath = os.path.join(root, filename)
                        startIndex = generateDataset(videoPath, outputDir, imageSize, frameInterval, startIndex, vectorsNumbers, vectorTimeWindow, vectorTimeOffsets, temporalContextTimeWindow, DEBUGVIZ=DEBUGVIZ)
    else:
        print(f"Error: {Path} is neither a file nor a directory.")

    print(f"Total dataset items generated: {startIndex}\n")



if __name__ == "__main__":
    videoPath = r"F:\Projects\Autopilot\nvidia_dataset"
    outputDir = r"D:\VS_Python_Project\Autopilot\NVIDIA_Dataset_output"
    frameInterval = 0 # interval between each frame sample, in seconds
    imageSize = (640, 360)  # Width, Height
    
    vectorTimeOffsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.85, 1.1, 1.35, 1.6, 2.3, 3.0]
    
    
    
    if vectorTimeOffsets:
        vectorsNumbers = len(vectorTimeOffsets)
        vectorTimeWindow = float(vectorTimeOffsets[-1])
    else:
        vectorsNumbers = 12
        vectorTimeWindow = 3.0 # seconds

    temporalContextTimeWindow = 0.1 # difference in seconds between the current frame and the previous frame used for temporal context.

    manualStartIndex = 0  # Starting index for dataset items, can be adjusted if resuming from a previous run

    debugViz = False
    numWorkers = 3
    exportPngs = True
    
    main(
        videoPath,
        outputDir,
        imageSize,
        frameInterval,
        vectorsNumbers,
        vectorTimeWindow,
        vectorTimeOffsets,
        temporalContextTimeWindow,
        manualStartIndex,
        DEBUGVIZ=debugViz,
        labelsOnly=True,
        exportPngs=exportPngs,
        numWorkers=numWorkers
    )
