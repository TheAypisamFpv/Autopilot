import os
import json
import random
from typing import List
import sys

# Add the parent directory to sys.path to enable absolute import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import progressBar


def balanceDataset(
    datasetDir: str,
    lateralThreshold: float = 1.0,
    seed: int = 42
) -> None:
    """
    Balance dataset 50/50 straight vs curves.
    Outputs only the list of label indices to keep in balanced.json.
    S-curves are counted as curves.
    """
    labelsDir = os.path.join(datasetDir, "labels")
    if not os.path.exists(labelsDir):
        raise FileNotFoundError(f"Labels directory not found: {labelsDir}")

    straightIndices: List[str] = []
    curveIndices: List[str] = []

    print("Analyzing dataset for balancing...")
    i = 0
    wi = 0
    dirLen = len(os.listdir(labelsDir))
    for labelFilename in os.listdir(labelsDir):
        i += 1
        completion = i / dirLen
        if i % 10 == 0 or completion == 1.0:
            wi += 1
            print(progressBar.getProgressBar(completion, wi), end='\r')

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

            xComponents = []
            vectorsPart = vectorsLine.split(" : ")[1].strip()
            for pair in vectorsPart.split():
                xStr, _ = pair.split(",")
                xComponents.append(float(xStr))

            # Track cumulative lateral position
            cumulativeLateral = 0.0
            maxLateralDeviation = 0.0
            for dx in xComponents:
                cumulativeLateral += dx
                maxLateralDeviation = max(maxLateralDeviation, abs(cumulativeLateral))

            if maxLateralDeviation > lateralThreshold:
                curveIndices.append(index)
            else:
                straightIndices.append(index)

        except Exception as exception:
            print(f"Failed to parse {labelPath}: {exception}")
            continue

    print("\nBalancing dataset...")
    random.seed(seed)
    targetCount = min(len(straightIndices), len(curveIndices))
    selectedStraight = random.sample(straightIndices, targetCount)
    selectedCurves = random.sample(curveIndices, targetCount)

    selectedIndices = sorted(selectedStraight + selectedCurves, key=int)

    print("Saving balanced.json...")
    jsonPath = os.path.join(datasetDir, "balanced.json")
    with open(jsonPath, 'w') as outputFile:
        json.dump({"keep": selectedIndices}, outputFile, indent=2)

    balancedDir = datasetDir + "_balanced"
    os.rename(datasetDir, balancedDir)

    print(f"Balanced: {len(selectedIndices)} samples ({targetCount} straight + {targetCount} curves)")
    print(f"Saved: {jsonPath}")
    print(f"Renamed folder to: {balancedDir}")


if __name__ == "__main__":
    datasetDir = r"D:\VS_Python_Project\Autopilot\Autopilot\dataset\output_12_3.0_0.1_framesize640x360"
    lateralThreshold = 1.0  # meters
    balanceDataset(datasetDir, lateralThreshold)