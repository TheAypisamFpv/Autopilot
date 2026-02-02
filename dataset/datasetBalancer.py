import os
import json
import random
import time
from datetime import datetime
from typing import List
import sys
import math
import statistics
import glob
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

# Add the parent directory to sys.path to enable absolute import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import progressBar


def loadCalibrationData(calibrationRoot, clipUuid, cameraName):
    cameraIntrinsicsPath = os.path.join(calibrationRoot, "camera_intrinsics")
    sensorExtrinsicsPath = os.path.join(calibrationRoot, "sensor_extrinsics")
    vehicleDimensionsPath = os.path.join(calibrationRoot, "vehicle_dimensions")

    intrinsics = None
    extrinsics = None
    vehicleDims = None

    for pqFile in glob.glob(os.path.join(cameraIntrinsicsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            intrinsics = df.loc[(clipUuid, cameraName)].to_dict()
            break
        except KeyError:
            continue

    for pqFile in glob.glob(os.path.join(sensorExtrinsicsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            extrinsics = df.loc[(clipUuid, cameraName)].to_dict()
            break
        except KeyError:
            continue

    for pqFile in glob.glob(os.path.join(vehicleDimensionsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            vehicleDims = df.loc[clipUuid].to_dict()
            break
        except KeyError:
            continue

    return intrinsics, extrinsics, vehicleDims


def balanceDataset(
    datasetDir: str,
    calibrationRoot: str,
    seed: int = 42
) -> None:
    """
    Balance dataset 50/50 straight vs turns (right, left, S-shaped).
    Outputs only the list of label indices to keep in balanced.json.
    """
    labelsDir = os.path.join(datasetDir, "labels")
    if not os.path.exists(labelsDir):
        raise FileNotFoundError(f"Labels directory not found: {labelsDir}")

    # Collect all clipUuids first to compute average lateralThreshold
    clipUuids = []
    for labelFilename in os.listdir(labelsDir):
        if labelFilename.endswith(".txt"):
            index = os.path.splitext(labelFilename)[0]
            clipUuids.append(index)

    # Compute average vehicle width
    vehicleWidths = []
    cameraName = "camera_front_wide_120fov"
    for clipUuid in clipUuids:
        try:
            _, _, vehicleDims = loadCalibrationData(calibrationRoot, clipUuid, cameraName)
            width = vehicleDims.get('width', 2.0) if vehicleDims else 2.0
            vehicleWidths.append(width)
        except Exception:
            vehicleWidths.append(2.0)  # Always fallback to 2.0m

    averageWidth = statistics.mean(vehicleWidths) if vehicleWidths else 2.0
    lateralThreshold = averageWidth / 2.0

    straightIndices: List[str] = []
    rightTurnIndices: List[str] = []
    leftTurnIndices: List[str] = []
    sTurnIndices: List[str] = []

    print("Analyzing dataset for balancing...")
    i = 0
    wi = 0
    dirLen = len(os.listdir(labelsDir))
    startTime = time.time()
    for labelFilename in os.listdir(labelsDir):
        i += 1
        completion = i / dirLen
        if i % 100 == 0 or completion == 1.0:
            wi += 1
            elapsed = time.time() - startTime
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = (dirLen - i) / rate if rate > 0 else 0.0
            etaFinish = datetime.fromtimestamp(time.time() + remaining)
            etaTime = etaFinish.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{progressBar.getProgressBar(completion, wi)}Parsing vectors ETA: {etaTime}", end='\r')

        if not labelFilename.endswith(".txt"):
            print(f"Skipping non-txt file: {labelFilename}", end='\r')
            continue

        labelPath = os.path.join(labelsDir, labelFilename)
        index = os.path.splitext(labelFilename)[0]

        try:
            with open(labelPath, 'r') as f:
                lines = f.readlines()

            vectorsLine = next((line for line in lines if line.startswith("vectors : ")), None)
            if not vectorsLine or "None" in vectorsLine:
                print(f"Skipping {labelFilename}: No valid vectors found", end='\r')
                continue

            vectors = []
            vectorsPart = vectorsLine.split(" : ")[1].strip()
            for pair in vectorsPart.split():
                xStr, yStr = pair.split(",")
                vectors.append((float(xStr), float(yStr)))

            if not vectors:
                continue

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
                straightIndices.append(index)
            else:
                if totalAngle < -10.0:
                    rightTurnIndices.append(index)
                elif totalAngle > 10.0:
                    leftTurnIndices.append(index)
                else:
                    sTurnIndices.append(index)

        except Exception as exception:
            print(f"Failed to parse {labelPath}: {exception}")
            continue

    print("\nBalancing dataset...")
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

    selectedIndices = sorted(selectedStraight + selectedRight + selectedLeft + selectedS, key=int)

    print("Saving balanced.json...")
    jsonPath = os.path.join(datasetDir, "balanced.json")
    with open(jsonPath, 'w') as outputFile:
        json.dump({"keep": selectedIndices}, outputFile, indent=2)

    print(f"Balanced: {len(selectedIndices)} samples ({targetStraight} straight + {targetRight} right + {targetLeft} left + {targetS} S-turns)")
    print(f"Saved: {jsonPath}")


if __name__ == "__main__":
    datasetDir = r"F:\Projects\Autopilot\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360"
    calibrationRoot = r"F:\Projects\Autopilot\calibration"
    balanceDataset(datasetDir, calibrationRoot)