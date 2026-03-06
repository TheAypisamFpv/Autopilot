import os
import json
import random
import time
from datetime import datetime
from typing import List, Tuple
import sys
import math
import statistics
import glob
import pandas as pd
import concurrent.futures
import threading

# Add the parent directory to sys.path to enable absolute import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import progressBar


def processLabel(index: int, labelFilename: str, labelsDir: str, lateralThreshold: float) -> Tuple[str, str]:
    labelPath = os.path.join(labelsDir, labelFilename)
    indexStr = os.path.splitext(labelFilename)[0]

    try:
        with open(labelPath, 'r') as f:
            lines = f.readlines()

        vectorsLine = next((line for line in lines if line.startswith("vectors : ")), None)
        vectorTimesLine = next((line for line in lines if line.startswith("vectorTimes : ")), None)
        if not vectorsLine or "None" in vectorsLine:
            return indexStr, 'skip'

        vectors = []
        vectorsPart = vectorsLine.split(" : ")[1].strip()
        for pair in vectorsPart.split():
            xStr, yStr = pair.split(",")
            vectors.append((float(xStr), float(yStr)))

        if not vectors:
            return indexStr, 'skip'

        firstDx, firstDy = vectors[0]
        firstDistance = math.hypot(firstDx, firstDy)
        if vectorTimesLine and "None" not in vectorTimesLine:
            try:
                timeParts = vectorTimesLine.split(" : ")[1].strip().split()
                firstTime = float(timeParts[0]) if timeParts else 0.0
            except (ValueError, IndexError):
                firstTime = 0.0
        else:
            firstTime = 3.0 / max(1, len(vectors))

        if firstTime > 0:
            firstSpeed = firstDistance / firstTime
            if firstSpeed < (1.0 / 3.6):
                return indexStr, 'still'

        # Compute angles in degrees
        angles = [math.degrees(math.atan2(dy, dx)) for dx, dy in vectors]
        angleDiffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)] if len(angles) > 1 else []

        # Constant turn check
        isConstantTurn = False
        if angleDiffs:
            meanDiff = statistics.mean(angleDiffs)
            stdDiff = statistics.stdev(angleDiffs) if len(angleDiffs) > 1 else 0.0
            minDiff = min(angleDiffs)
            maxDiff = max(angleDiffs)
            isConstantTurn = minDiff >= meanDiff - stdDiff and maxDiff <= meanDiff + stdDiff
        else:
            isConstantTurn = True  # Single vector

        # Total angle
        totalAngle = angles[-1] - angles[0] if angles else 0.0

        # Max lateral
        cumulativeLateral = 0.0
        maxLateralDeviation = 0.0
        for dx, _ in vectors:
            cumulativeLateral += dx
            maxLateralDeviation = max(maxLateralDeviation, abs(cumulativeLateral))

        # Classify
        isStraight = (maxLateralDeviation <= lateralThreshold) or (
            maxLateralDeviation > lateralThreshold and isConstantTurn and abs(totalAngle) <= 10.0
        )

        if isStraight:
            return indexStr, 'straight'
        else:
            if totalAngle < -10.0:
                return indexStr, 'right'
            elif totalAngle > 10.0:
                return indexStr, 'left'
            else:
                return indexStr, 's'

    except Exception:
        return indexStr, 'skip'


def loadCalibrationData(calibrationRoot, clipUuid, cameraName):
    intrinsics = None
    extrinsics = None
    vehicleDims = None
    try:
        cameraIntrinsicsPath = os.path.join(calibrationRoot, "camera_intrinsics")
        for pqFile in glob.glob(os.path.join(cameraIntrinsicsPath, "*.parquet")):
            df = pd.read_parquet(pqFile)
            try:
                # Extract UUID part for calibration lookup
                uuidPart = clipUuid.split('_')[0]
                intrinsics = df.loc[(uuidPart, cameraName)].to_dict()
                break
            except KeyError:
                continue

        sensorExtrinsicsPath = os.path.join(calibrationRoot, "sensor_extrinsics")
        for pqFile in glob.glob(os.path.join(sensorExtrinsicsPath, "*.parquet")):
            df = pd.read_parquet(pqFile)
            try:
                # Extract UUID part for calibration lookup
                uuidPart = clipUuid.split('_')[0]
                extrinsics = df.loc[(uuidPart, cameraName)].to_dict()
                break
            except KeyError:
                continue

        vehicleDimensionsPath = os.path.join(calibrationRoot, "vehicle_dimensions")
        for pqFile in glob.glob(os.path.join(vehicleDimensionsPath, "*.parquet")):
            df = pd.read_parquet(pqFile)
            try:
                # Extract UUID part for vehicle dimensions lookup
                uuidPart = clipUuid.split('_')[0]
                vehicleDims = df.loc[uuidPart].to_dict()
                break
            except KeyError:
                continue
    except Exception:
        pass  # Ignore errors
    return intrinsics, extrinsics, vehicleDims


def loadWidth(clipUuid: str, calibrationRoot: str, cameraName: str) -> float:
    try:
        vehicleDims = loadCalibrationData(calibrationRoot, clipUuid, cameraName)[2]
        return vehicleDims.get('width', 2.0) if vehicleDims else 2.0
    except Exception:
        return 2.0


def loadVehicleWidthMap(calibrationRoot: str) -> dict:
    vehicleDimensionsPath = os.path.join(calibrationRoot, "vehicle_dimensions")
    if not os.path.isdir(vehicleDimensionsPath):
        return {}

    widthMap = {}
    pqFiles = glob.glob(os.path.join(vehicleDimensionsPath, "*.parquet"))
    totalFiles = len(pqFiles)
    if totalFiles == 0:
        return {}
    for i, pqFile in enumerate(pqFiles):
        df = pd.read_parquet(pqFile)
        if 'width' in df.columns:
            for clipUuid, width in df['width'].items():
                widthMap[str(clipUuid)] = float(width)
        if i % 69 == 0 or i == totalFiles - 1:
            completion = (i + 1) / totalFiles if totalFiles else 1.0
            print(f"{progressBar.getProgressBar(completion, int(i))}Loading width map", end='\r')
    if totalFiles > 0:
        print()
    return widthMap


def balanceDataset(
    datasetDir: str,
    calibrationRoot: str,
    seed: int = 42,
    threadWorkers: int = 8
) -> None:
    """
    Balance dataset 50/50 straight vs turns (right, left, S-shaped).
    Outputs only the list of label indices to keep in balanced.json.
    """
    labelsDir = os.path.join(datasetDir, "labels")
    if not os.path.exists(labelsDir):
        raise FileNotFoundError(f"Labels directory not found: {labelsDir}")

    # Collect all clipUuids first to compute average lateralThreshold
    print("Collecting clip UUIDs...")
    clipUuids = []
    allFiles = os.listdir(labelsDir)
    totalFilesForUuid = len(allFiles)
    for i, labelFilename in enumerate(allFiles):
        if labelFilename.endswith(".txt"):
            index = os.path.splitext(labelFilename)[0]
            clipUuids.append(index)
            
        if i % 42069 == 0 or i == totalFilesForUuid - 1:
            completion = (i + 1) / totalFilesForUuid
            print(f"{progressBar.getProgressBar(completion, int(i))}Collecting UUIDs", end='\r')
    
    print(f"\nCollected {len(clipUuids)} UUIDs from {totalFilesForUuid} files.")

    # Compute average vehicle width
    print("\n\nLoading vehicle calibrations...")
    vehicleDimensionsPath = os.path.join(calibrationRoot, "vehicle_dimensions")
    if not os.path.isdir(vehicleDimensionsPath):
        print(f"Calibration folder not found: {vehicleDimensionsPath}")
    else:
        print(f"Using calibration folder: {vehicleDimensionsPath}")
    
    totalUuids = len(clipUuids)
    vehicleWidths = [2.0] * totalUuids  # Default
    widthMap = loadVehicleWidthMap(calibrationRoot)

    if widthMap:
        for i, clipUuid in enumerate(clipUuids):
            # Extract UUID part (before underscore) for calibration lookup
            uuidPart = clipUuid.split('_')[0]
            vehicleWidths[i] = widthMap.get(uuidPart, 2.0)
            if i % 42069 == 0 or i == totalUuids - 1:
                completion = (i + 1) / totalUuids
                print(f"{progressBar.getProgressBar(completion, int(i))}Loading calibrations", end='\r')
        print()  # Newline
    else:
        print(f"No calibration files found in {vehicleDimensionsPath}. Using default width 2.0m for all clips.")

    averageWidth = statistics.mean(vehicleWidths) if vehicleWidths else 2.0
    lateralThreshold = averageWidth / 2.0
    print(f"\nLoaded calibrations for {len(vehicleWidths)} clips. Average width: {averageWidth:.2f}m, Lateral threshold: {lateralThreshold:.2f}m")

    print("\nClassifying label files...")

    straightIndices: List[str] = []
    rightTurnIndices: List[str] = []
    leftTurnIndices: List[str] = []
    sTurnIndices: List[str] = []
    stillIndices: List[str] = []

    # Get all label files
    labelFiles = [f for f in os.listdir(labelsDir) if f.endswith('.txt')]
    totalFiles = len(labelFiles)

    print(f"\nProcessing label files with {threadWorkers} threads...")
    startTime = time.time()
    processedCount = 0
    lock = threading.Lock()
    stopEvent = threading.Event()
    threadIdLock = threading.Lock()
    threadIdMap = {}
    nextThreadId = 1

    def getThreadId():
        nonlocal nextThreadId
        threadName = threading.current_thread().name
        with threadIdLock:
            threadId = threadIdMap.get(threadName)
            if threadId is None:
                threadId = nextThreadId
                if threadId > threadWorkers:
                    threadId = ((threadId - 1) % threadWorkers) + 1
                threadIdMap[threadName] = threadId
                nextThreadId += 1
            return threadId

    def progressUpdater():
        while not stopEvent.is_set():
            with lock:
                completion = processedCount / totalFiles if totalFiles else 1.0
                elapsed = time.time() - startTime
                if processedCount == 0:
                    etaStr = "waiting for first completion..."
                else:
                    rate = processedCount / elapsed if elapsed > 0 else 0.0
                    remaining = (totalFiles - processedCount) / rate if rate > 0 else 0.0
                    etaFinish = datetime.fromtimestamp(time.time() + remaining)
                    etaStr = etaFinish.strftime("%Y-%m-%d %H:%M:%S")

                print(f"{progressBar.getProgressBar(completion, processedCount//1000)}Processing files ({processedCount}/{totalFiles}) ETA: {etaStr}                ", end='\r', flush=True)
            time.sleep(0.01)  # Update every 0.01 seconds

    def processLabelWithThreadInfo(fileIndex: int, labelFilename: str):
        threadId = getThreadId()
        # print(f"Thread {threadId} start: {labelFilename}")
        result = processLabel(fileIndex, labelFilename, labelsDir, lateralThreshold)
        # print(f"Thread {threadId} end: {labelFilename}")
        return result

    # Start progress updater thread
    progressThread = threading.Thread(target=progressUpdater)
    progressThread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=threadWorkers) as executor:
        futures = {}
        fileIter = iter(enumerate(labelFiles))
        maxQueue = max(threadWorkers * 2, 1)

        def submitNext():
            try:
                i, f = next(fileIter)
            except StopIteration:
                return False
            future = executor.submit(processLabelWithThreadInfo, i, f)
            futures[future] = (i, f)
            return True

        for _ in range(maxQueue):
            if not submitNext():
                break

        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                i, f = futures.pop(future)
                index, category = future.result()
                with lock:
                    processedCount += 1
                if category == 'straight':
                    straightIndices.append(index)
                elif category == 'right':
                    rightTurnIndices.append(index)
                elif category == 'left':
                    leftTurnIndices.append(index)
                elif category == 's':
                    sTurnIndices.append(index)
                elif category == 'still':
                    stillIndices.append(index)
                submitNext()

    # Stop progress updater
    stopEvent.set()
    progressThread.join()
    # Final progress print
    completion = processedCount / totalFiles
    print(f"{progressBar.getProgressBar(completion, int(processedCount))}Processing files ({processedCount}/{totalFiles}) ETA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", end='\r')
    print()  # Newline

    print(f"Processed {processedCount} files.")


    print("\n\nBalancing dataset...")
    random.seed(seed)
    targetRight = min(len(rightTurnIndices), len(leftTurnIndices))
    targetLeft = targetRight
    targetS = len(sTurnIndices)
    targetTurns = targetRight + targetLeft + targetS
    targetStraight = min(len(straightIndices), targetTurns)

    selectedStraight = random.sample(straightIndices, targetStraight)
    selectedRight = random.sample(rightTurnIndices, targetRight)
    selectedLeft = random.sample(leftTurnIndices, targetLeft)
    selectedS = random.sample(sTurnIndices, targetS)

    if stillIndices:
        random.shuffle(stillIndices)
        for i, idx in enumerate(stillIndices):
            bucket = i % 4
            if bucket == 0:
                selectedStraight.append(idx)
            elif bucket == 1:
                selectedRight.append(idx)
            elif bucket == 2:
                selectedLeft.append(idx)
            else:
                selectedS.append(idx)

    def sortLabelIndex(labelIndex: str):
        uuidPart, _, framePart = labelIndex.rpartition('_')
        if uuidPart and framePart.isdigit():
            return (uuidPart, int(framePart))
        return (labelIndex, 0)

    selectedIndices = sorted(selectedStraight + selectedRight + selectedLeft + selectedS, key=sortLabelIndex)
    selectedStraight = sorted(selectedStraight, key=sortLabelIndex)
    selectedRight = sorted(selectedRight, key=sortLabelIndex)
    selectedLeft = sorted(selectedLeft, key=sortLabelIndex)
    selectedS = sorted(selectedS, key=sortLabelIndex)
    stillIndices = sorted(stillIndices, key=sortLabelIndex)

    print("\nSaving balanced.json...")
    jsonPath = os.path.join(datasetDir, "balanced.json")
    with open(jsonPath, 'w') as outputFile:
        json.dump({
            "keep": selectedIndices,
            "straight": selectedStraight,
            "right": selectedRight,
            "left": selectedLeft,
            "s": selectedS,
            "still": stillIndices,
        }, outputFile, indent=2)

    print(f"Balanced: {len(selectedIndices)} samples ({targetStraight} straight + {targetRight} right + {targetLeft} left + {targetS} S-turns + {len(stillIndices)} still)")
    print(f"Saved: {jsonPath}")


if __name__ == "__main__":
    datasetDir = r"F:\Projects\Autopilot\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360(1)"
    calibrationRoot = r"F:\Projects\Autopilot\nvidia_dataset\calibration"

    threadWorkers = 16
    seed = 42
    
    balanceDataset(datasetDir, calibrationRoot, seed, threadWorkers)