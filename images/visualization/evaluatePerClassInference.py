import json
import os
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if projectRoot not in sys.path:
    sys.path.insert(0, projectRoot)

from model.CreateModel import TrajectoryModel
from model.progressBar import getProgressBar
from model.train import DrivingDataset


runName = "run23"
runDir = os.path.join(projectRoot, "training", runName)
trainingParamsPath = os.path.join(runDir, "training_params.json")
checkpointPath = os.path.join(runDir, "best_model.pth")

# Set this explicitly if needed. If None, datasetDir is read from training_params.json.
datasetDir = None
balancedJsonPath = None

samplesPerClass = 1000
classOrder = ["straight", "right", "left", "s", "still"]

batchSize = 24
numWorkers = 0
randomSeed = 42
deviceOverride = None

outputDir = os.path.dirname(__file__)

def getOutputPaths(modeSuffix):
    baseName = f"{runName}_{modeSuffix}_perClass"
    return {
        "csv": os.path.join(outputDir, f"{baseName}Metrics.csv"),
        "figure": os.path.join(outputDir, f"{baseName}AdeFde.png"),
        "json": os.path.join(outputDir, f"{baseName}Results.json"),
    }


def ensureFileExists(filePath):
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Missing file: {filePath}")


def loadJsonFile(filePath):
    ensureFileExists(filePath)
    with open(filePath, "r", encoding="utf-8") as fileHandle:
        return json.load(fileHandle)


def resolveDevice():
    if deviceOverride:
        return torch.device(deviceOverride)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def buildEvalTransform():
    return transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def buildModelFromParams(modelParams, device):
    predSteps = int(modelParams.get("predSteps", 12))
    vectorTimes = modelParams.get("vectorTimes", None)

    model = TrajectoryModel(
        featDim=int(modelParams.get("featDim", 384)),
        hiddenDim=int(modelParams.get("hiddenDim", 640)),
        predSteps=predSteps,
        intervalSeconds=float(modelParams.get("intervalSeconds", 0.25)),
        vectorTimes=vectorTimes,
        baseChannels=int(modelParams.get("baseChannels", 48)),
        numHeads=int(modelParams.get("numHeads", 8)),
        numLayers=int(modelParams.get("numLayers", 3)),
    ).to(device)

    checkpointData = torch.load(checkpointPath, map_location=device)
    if isinstance(checkpointData, dict) and "model_state_dict" in checkpointData:
        stateDict = checkpointData["model_state_dict"]
    else:
        stateDict = checkpointData

    model.load_state_dict(stateDict, strict=False)
    model.eval()
    return model, predSteps


def computeKinematicPositions(predKinematic, vectorTimesBatch):
    if vectorTimesBatch.dim() == 1:
        vectorTimesBatch = vectorTimesBatch.unsqueeze(0)

    deltaTimes = vectorTimesBatch.clone()
    if vectorTimesBatch.size(1) > 1:
        deltaTimes[:, 1:] = vectorTimesBatch[:, 1:] - vectorTimesBatch[:, :-1]
    deltaTimes = torch.clamp(deltaTimes, min=1e-3)

    batchSizeLocal, predStepsLocal, _ = predKinematic.shape

    yawValues = torch.zeros(batchSizeLocal, device=predKinematic.device, dtype=predKinematic.dtype)
    posXValues = torch.zeros(batchSizeLocal, device=predKinematic.device, dtype=predKinematic.dtype)
    posYValues = torch.zeros(batchSizeLocal, device=predKinematic.device, dtype=predKinematic.dtype)

    integratedPositions = []
    for stepIndex in range(predStepsLocal):
        stepDt = deltaTimes[:, stepIndex]
        speedValues = predKinematic[:, stepIndex, 0]
        yawRateValues = predKinematic[:, stepIndex, 1]

        yawValues = yawValues + yawRateValues * stepDt
        posXValues = posXValues + speedValues * torch.sin(yawValues) * stepDt
        posYValues = posYValues + speedValues * torch.cos(yawValues) * stepDt
        integratedPositions.append(torch.stack([posXValues, posYValues], dim=-1))

    return torch.stack(integratedPositions, dim=1)


def selectClassIndices(datasetSamples, balancedData):
    sampleIndexMap = {sampleId: sampleIndex for sampleIndex, sampleId in enumerate(datasetSamples)}

    rng = random.Random(randomSeed)
    selectedIndices = []
    indexToClass = {}
    requestedCountByClass = {}

    for className in classOrder:
        classSampleIds = list(balancedData.get(className, []))
        if len(classSampleIds) > samplesPerClass:
            classSampleIds = rng.sample(classSampleIds, samplesPerClass)

        classIndices = []
        for sampleId in classSampleIds:
            sampleIndex = sampleIndexMap.get(sampleId)
            if sampleIndex is None:
                continue
            classIndices.append(sampleIndex)

        requestedCountByClass[className] = len(classIndices)

        for sampleIndex in classIndices:
            selectedIndices.append(sampleIndex)
            indexToClass[sampleIndex] = className

    return selectedIndices, indexToClass, requestedCountByClass


def evaluatePerClass(model, evalLoader, selectedIndices, indexToClass, device, useEgoHistory=True):
    metricSums = {
        className: {"count": 0, "adeSum": 0.0, "fdeSum": 0.0}
        for className in classOrder
    }

    totalEvalSamples = len(selectedIndices)
    totalBatches = len(evalLoader)
    cursor = 0
    evaluatedSamples = 0
    startTime = time.perf_counter()
    print()
    with torch.no_grad():
        for batchIndex, (prevImage, currentImage, dynamicData, labels, vectorTimesBatch, egoHistory) in enumerate(evalLoader, start=1):
            del dynamicData

            prevImage = prevImage.to(device)
            currentImage = currentImage.to(device)
            labels = labels.to(device)
            vectorTimesBatch = vectorTimesBatch.to(device)

            egoHistoryInput = egoHistory.to(device) if useEgoHistory else None

            predKinematic, _, _ = model(
                currentImage,
                prevImage,
                teacherForcing=False,
                tfRatio=0.0,
                egoHistory=egoHistoryInput,
                vectorTimes=vectorTimesBatch,
                returnAttn=False,
            )

            predPositions = computeKinematicPositions(predKinematic, vectorTimesBatch)
            gtPositions = torch.cumsum(labels, dim=1)

            distanceTensor = torch.norm(predPositions - gtPositions, dim=-1)
            adePerSample = distanceTensor.mean(dim=1).detach().cpu().numpy()
            fdePerSample = distanceTensor[:, -1].detach().cpu().numpy()

            batchCount = len(adePerSample)
            evaluatedSamples += batchCount
            completion = min(1.0, evaluatedSamples / max(1, totalEvalSamples))
            elapsed = time.perf_counter() - startTime
            eta = (elapsed / completion - elapsed) if completion > 0 else 0.0
            finishTime = datetime.fromtimestamp(time.time() + eta)
            etaTime = finishTime.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{getProgressBar(completion, wheelIndex=batchIndex, maxbarLength=60)}"
                f"Evaluating batch {batchIndex}/{totalBatches} ({evaluatedSamples}/{totalEvalSamples}) - ETA: {etaTime}",
                end="\r",
                flush=True,
            )

            batchGlobalIndices = selectedIndices[cursor:cursor + batchCount]
            cursor += batchCount

            for localIndex, sampleIndex in enumerate(batchGlobalIndices):
                className = indexToClass.get(sampleIndex)
                if className not in metricSums:
                    continue

                adeValue = float(adePerSample[localIndex])
                fdeValue = float(fdePerSample[localIndex])
                if not np.isfinite(adeValue) or not np.isfinite(fdeValue):
                    continue

                metricSums[className]["count"] += 1
                metricSums[className]["adeSum"] += adeValue
                metricSums[className]["fdeSum"] += fdeValue

    print()
    return metricSums


def buildResultsDataframe(metricSums, requestedCountByClass):
    resultRows = []
    for className in classOrder:
        classCount = metricSums[className]["count"]
        if classCount > 0:
            adeMean = metricSums[className]["adeSum"] / classCount
            fdeMean = metricSums[className]["fdeSum"] / classCount
        else:
            adeMean = np.nan
            fdeMean = np.nan

        resultRows.append(
            {
                "className": className,
                "requestedSamples": requestedCountByClass.get(className, 0),
                "evaluatedSamples": classCount,
                "adeMeters": adeMean,
                "fdeMeters": fdeMean,
            }
        )

    return pd.DataFrame(resultRows)


def plotPerClassBars(resultDf, figurePath, modeLabel):
    classNames = resultDf["className"].tolist()
    adeValues = resultDf["adeMeters"].to_numpy(dtype=float)
    fdeValues = resultDf["fdeMeters"].to_numpy(dtype=float)
    sampleCounts = resultDf["evaluatedSamples"].to_numpy(dtype=int)

    xPositions = np.arange(len(classNames), dtype=float)
    barWidth = 0.36

    plotAdeValues = np.nan_to_num(adeValues, nan=0.0)
    plotFdeValues = np.nan_to_num(fdeValues, nan=0.0)

    plt.figure(figsize=(11, 6))
    plt.bar(xPositions - barWidth / 2.0, plotAdeValues, width=barWidth, label="ADE")
    plt.bar(xPositions + barWidth / 2.0, plotFdeValues, width=barWidth, label="FDE")

    for idx, countValue in enumerate(sampleCounts):
        yValue = max(plotAdeValues[idx], plotFdeValues[idx])
        plt.text(xPositions[idx], yValue + 0.02, f"n={countValue}", ha="center", va="bottom", fontsize=9)

    plt.xticks(xPositions, classNames)
    plt.ylabel("Error (meters)")
    plt.title(
        f"Per-class inference metrics ({runName})\n"
        f"Mode: {modeLabel} - Target {samplesPerClass} samples per class"
    )
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figurePath, dpi=220)
    plt.close()


def serializeResultsToJson(resultDf, jsonPath, modeLabel):
    outputData = {
        "runName": runName,
        "mode": modeLabel,
        "targetSamplesPerClass": samplesPerClass,
        "classOrder": classOrder,
        "results": resultDf.to_dict(orient="records"),
    }
    with open(jsonPath, "w", encoding="utf-8") as jsonFile:
        json.dump(outputData, jsonFile, indent=2)


def main():
    print("Loading training parameters...", end="\r")
    trainingParams = loadJsonFile(trainingParamsPath)
    print("Training parameters loaded.    ")

    effectiveDatasetDir = datasetDir if datasetDir else trainingParams.get("datasetDir")
    if not effectiveDatasetDir:
        raise ValueError("datasetDir is not set. Define datasetDir in this script.")
    if not os.path.exists(effectiveDatasetDir):
        raise FileNotFoundError(
            "Dataset path does not exist on this machine. "
            f"Update datasetDir in this script. Current value: {effectiveDatasetDir}"
        )

    effectiveBalancedJsonPath = (
        balancedJsonPath
        if balancedJsonPath
        else os.path.join(effectiveDatasetDir, "balanced.json")
    )

    ensureFileExists(effectiveBalancedJsonPath)
    ensureFileExists(checkpointPath)

    print("Loading balancing data...", end="\r")
    balancedData = loadJsonFile(effectiveBalancedJsonPath)
    print("Balancing data loaded.   ")
    device = resolveDevice()

    print(f"Using device: {device}")
    print(f"Effective dataset directory: {effectiveDatasetDir}")
    print(f"Balanced JSON path: {effectiveBalancedJsonPath}")
    print("Building model and dataset...", end="\r")
    model, predSteps = buildModelFromParams(trainingParams, device)
    print("Model and dataset ready.     ")
    
    print(f"Evaluating on {len(balancedData.get(classOrder[0], []))} samples per class (if available)...")
    evalTransform = buildEvalTransform()
    
    print("Creating evaluation dataset and loader...", end="\r")
    evalDataset = DrivingDataset(
        effectiveDatasetDir,
        transform=evalTransform,
        maxSize=None,
        predSteps=predSteps,
        verbose=True,
        loadEgoHistory=True,
    )
    print("Evaluation dataset and loader ready.     ")

    selectedIndices, indexToClass, requestedCountByClass = selectClassIndices(evalDataset.samples, balancedData)
    if len(selectedIndices) == 0:
        raise RuntimeError("No samples selected for evaluation. Check class keys and dataset alignment.")

    print(f"Selected {len(selectedIndices)} samples for evaluation across {len(requestedCountByClass)} classes.")

    variants = [
        ("withEgo", True),
        ("withoutEgo", False),
    ]

    for modeName, useEgoHistory in variants:
        print(f"\nPreparing evaluation dataset for mode: {modeName}...")
        evalDatasetMode = evalDataset if useEgoHistory else DrivingDataset(
            effectiveDatasetDir,
            transform=evalTransform,
            maxSize=None,
            predSteps=predSteps,
            verbose=True,
            loadEgoHistory=False,
        )

        evalSubset = Subset(evalDatasetMode, selectedIndices)
        evalLoader = DataLoader(
            evalSubset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            pin_memory=(device.type == "cuda"),
        )

        print(f"Starting per-class evaluation ({modeName})...")
        metricSums = evaluatePerClass(model, evalLoader, selectedIndices, indexToClass, device, useEgoHistory=useEgoHistory)
        resultDf = buildResultsDataframe(metricSums, requestedCountByClass)

        paths = getOutputPaths(modeName)
        resultDf.to_csv(paths["csv"], index=False)
        serializeResultsToJson(resultDf, paths["json"], modeName)
        plotPerClassBars(resultDf, paths["figure"], modeName)

        print(f"Per-class evaluation complete ({modeName}).")
        print(f"CSV: {paths['csv']}")
        print(f"JSON: {paths['json']}")
        print(f"Figure: {paths['figure']}")
        print(resultDf.to_string(index=False))


if __name__ == "__main__":
    main()
