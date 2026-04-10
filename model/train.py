import os
# Keep FFmpeg decoding conservative by default for training stability.
# If needed, users can override this env var before launching training.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads=2;video_codec=h264")
import time
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torch.amp import GradScaler, autocast
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import multiprocessing
from collections import deque


def computeCleanSubepochSchedule(totalTrainSamples: int, requestedSamplesPerSub: int, minFullEpochs: int = 5):
    """
    Compute a clean sub-epoch schedule.

    Attempts to choose numSubEpochs and an adjusted samplesPerSub so that
    numSubEpochs * samplesPerSub ~= desiredFullEpochs * totalTrainSamples
    where desiredFullEpochs is at least minFullEpochs. The function returns
    (actualSamplesPerSub, numSubEpochs, desiredFullEpochs) and prints a
    short summary of the chosen schedule.
    """
    print(f"Computing clean sub-epoch schedule: totalTrainSamples={totalTrainSamples}, requestedSamplesPerSub={requestedSamplesPerSub}, minFullEpochs={minFullEpochs}")
    if totalTrainSamples <= 0:
        return 0, 0, 0

    requested = int(requestedSamplesPerSub) if requestedSamplesPerSub else totalTrainSamples
    best = None
    # Search a small range of candidate full-epoch counts starting at min_full_epochs
    for desired in range(minFullEpochs, minFullEpochs + 21):
        neededTotal = desired * totalTrainSamples
        numSubs = max(1, math.ceil(neededTotal / max(1, requested)))
        adjusted = max(1, int(round(neededTotal / numSubs)))
        error = abs(adjusted - requested)
        if best is None or error < best[0] or (error == best[0] and numSubs < best[1]):
            best = (error, numSubs, adjusted, desired)

    if best is None:
        return requested, 1, minFullEpochs

    _, numSubEpochs, actualSamplesPerSub, desiredFullEpochs = best
    print(
        f"Clean sub-epoch schedule: totalSamples={totalTrainSamples}, requestedPerSub={requested}, "
        f"actualPerSub={actualSamplesPerSub}, numSubEpochs={numSubEpochs}, desiredFullEpochs={desiredFullEpochs}"
    )
    return actualSamplesPerSub, numSubEpochs, desiredFullEpochs

try:
    from .CreateModel import TrajectoryModel
    from .progressBar import getProgressBar
except ImportError:
    from CreateModel import TrajectoryModel
    from progressBar import getProgressBar

# Set multiprocessing start method to 'spawn' for Windows compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Enable expandable segments for better memory management when supported.
if torch.cuda.is_available() and os.name != "nt" and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DrivingDataset(Dataset):
    _inMemoryEgoHistoryCache = {}
    """
    Dataset for trajectory prediction.
    Loads paired images (previous and current) and trajectory vectors.
    Optionally filters samples using 'balanced.json' if present.
    """

    def __init__(self, datasetDir, transform=None, maxSize=None, predSteps=6, dtype=torch.float32, verbose=True, loadEgoHistory=True):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, "labels")
        self.imagesDir = os.path.join(datasetDir, "images")
        self.transform = transform
        self.predSteps = predSteps
        self.dtype = dtype
        self.maxSize = maxSize
        self.labelsOnly = False
        self.verbose = verbose
        self.loadEgoHistory = loadEgoHistory
        
        self._videoCaches = {}          # videoPath -> (cap, lastFrameIndex)
        self._maxCachedVideos = 32      # LRU limit to prevent memory explosion

        if not os.path.exists(self.imagesDir):
            self.labelsOnly = True
        else:
            try:
                imageFiles = [f for f in os.listdir(self.imagesDir) if f.endswith(".png")]
                if len(imageFiles) == 0:
                    self.labelsOnly = True
            except FileNotFoundError:
                self.labelsOnly = True

        # Load all label indices
        allSamples = sorted([f.split(".")[0] for f in os.listdir(self.labelsDir) if f.endswith(".txt")])

        # Check for balanced.json
        balancedJsonPath = os.path.join(datasetDir, "balanced.json")
        if os.path.exists(balancedJsonPath):
            if self.verbose:
                print(f"Found balanced.json - using filtered dataset ({balancedJsonPath})")
            with open(balancedJsonPath, 'r') as f:
                balancedData = json.load(f)
            
            keepIndices = set(balancedData.get("keep", []))
            self.samples = [idx for idx in allSamples if idx in keepIndices]
            totalSamples = len(self.samples)
            effectiveSamples = min(totalSamples, self.maxSize) if self.maxSize is not None else totalSamples
            if self.verbose:
                print(f"Filtered to {effectiveSamples} / {totalSamples} samples based on balanced.json")
        else:
            self.samples = allSamples
            totalSamples = len(self.samples)
            effectiveSamples = min(totalSamples, self.maxSize) if self.maxSize is not None else totalSamples
            if self.verbose:
                print(f"No balanced.json found - using {effectiveSamples} / {totalSamples} samples")

        if self.maxSize is not None:
            self.samples = self.samples[:self.maxSize]

        if self.verbose:
            print(f"Active samples: {len(self.samples)}")

        self.egoHistoryArray = self._loadOrBuildEgoHistoryCache() if self.loadEgoHistory else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["egoHistoryArray"] = None
        state["verbose"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.samples)

    def _getEgoHistoryCachePath(self):
        cacheFilename = f"egoHistoryCache_pred{self.predSteps}_count{len(self.samples)}.npz"
        return os.path.join(self.datasetDir, cacheFilename)

    def _getCacheAnchors(self):
        if not self.samples:
            return np.array([], dtype=np.str_)

        anchorIndices = sorted({
            0,
            min(31, len(self.samples) - 1),
            min(127, len(self.samples) - 1),
            len(self.samples) - 1,
        })
        return np.array([self.samples[i] for i in anchorIndices], dtype=np.str_)

    def _loadOrBuildEgoHistoryCache(self):
        cachePath = self._getEgoHistoryCachePath()
        cacheAnchors = self._getCacheAnchors()
        inMemoryCache = DrivingDataset._inMemoryEgoHistoryCache.get(cachePath)
        if inMemoryCache is not None:
            egoHistory, cachedAnchors = inMemoryCache
            if (
                egoHistory.ndim == 3
                and egoHistory.shape[0] == len(self.samples)
                and egoHistory.shape[1:] == (8, 3)
                and egoHistory.dtype == np.float32
                and np.array_equal(cachedAnchors.astype(np.str_), cacheAnchors)
            ):
                if self.verbose:
                    print(f"Loaded ego history cache from memory: {cachePath}")
                return egoHistory

        if os.path.exists(cachePath):
            try:
                with np.load(cachePath, allow_pickle=False) as cacheData:
                    egoHistory = cacheData["egoHistory"]
                    cachedAnchors = cacheData["anchors"]

                if (
                    egoHistory.ndim == 3
                    and egoHistory.shape[0] == len(self.samples)
                    and egoHistory.shape[1:] == (8, 3)
                    and egoHistory.dtype == np.float32
                    and np.array_equal(cachedAnchors.astype(np.str_), cacheAnchors)
                ):
                    DrivingDataset._inMemoryEgoHistoryCache[cachePath] = (egoHistory, cachedAnchors)
                    if self.verbose:
                        print(f"Loaded ego history cache: {cachePath}")
                    return egoHistory

                if self.verbose:
                    print("Ego history cache mismatch. Rebuilding cache...")
            except Exception:
                if self.verbose:
                    print("Failed to read ego history cache. Rebuilding cache...")

        egoHistory = self._buildEgoHistoryMap()
        DrivingDataset._inMemoryEgoHistoryCache[cachePath] = (egoHistory, cacheAnchors)
        try:
            np.savez(cachePath, egoHistory=egoHistory, anchors=cacheAnchors)
            if self.verbose:
                print(f"Saved ego history cache: {cachePath}")
        except Exception:
            if self.verbose:
                print("Could not save ego history cache. Continuing without persistent cache.")

        return egoHistory

    def _ensureEgoHistoryArrayLoaded(self):
        if not self.loadEgoHistory:
            return
        if self.egoHistoryArray is None:
            self.egoHistoryArray = self._loadOrBuildEgoHistoryCache()

    def _parseLabelFile(self, labelPath):
        vectors = None
        vectorTimes = None
        speed = 0.0
        acceleration = 0.0
        turnRate = 0.0
        videoPath = None
        prevFrameIndex = None
        frameIndex = None

        with open(labelPath, "r") as f:
            for line in f:
                if line.startswith("vectors"):
                    vectorsLine = line.strip().split(" : ")[1]
                    if vectorsLine == "None":
                        vectors = np.zeros((self.predSteps, 2), dtype=np.float32)
                    else:
                        vectorsList = [list(map(float, v.split(","))) for v in vectorsLine.split(" ")]
                        while len(vectorsList) < self.predSteps:
                            vectorsList.append([0.0, 0.0])
                        vectors = np.array(vectorsList[:self.predSteps], dtype=np.float32)
                elif line.startswith("vectorTimes"):
                    timesLine = line.strip().split(" : ")[1]
                    if timesLine == "None":
                        vectorTimes = np.zeros((self.predSteps,), dtype=np.float32)
                    else:
                        timesList = [float(v) for v in timesLine.split(" ") if v]
                        while len(timesList) < self.predSteps:
                            timesList.append(0.0)
                        vectorTimes = np.array(timesList[:self.predSteps], dtype=np.float32)
                elif line.startswith("speed"):
                    speed = float(line.strip().split(" : ")[1])
                elif line.startswith("acceleration"):
                    acceleration = float(line.strip().split(" : ")[1])
                elif line.startswith("turnRate"):
                    turnRate = float(line.strip().split(" : ")[1])
                elif line.startswith("video"):
                    videoPath = line.strip().split(" : ", 1)[1]
                    """temporary FIX for folder renaming"""
                    if "Autopiot" in videoPath:
                        videoPath = videoPath.replace("Autopiot", "Autopilot")
                elif line.startswith("prevFrameIndex"):
                    prevFrameIndex = int(line.strip().split(" : ")[1])
                elif line.startswith("frameIndex"):
                    frameIndex = int(line.strip().split(" : ")[1])

        if vectors is None:
            vectors = np.zeros((self.predSteps, 2), dtype=np.float32)

        if vectorTimes is None:
            vectorTimes = np.zeros((self.predSteps,), dtype=np.float32)

        return vectors, vectorTimes, speed, acceleration, turnRate, videoPath, prevFrameIndex, frameIndex

    def _parseSampleId(self, sampleId):
        clipKey, separator, framePart = sampleId.rpartition("_")
        if separator and framePart.isdigit():
            return clipKey, int(framePart)
        return sampleId, -1

    def _loadDynamicState(self, sampleId):
        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        speed = 0.0
        acceleration = 0.0
        turnRate = 0.0
        foundSpeed = False
        foundAcceleration = False
        foundTurnRate = False

        with open(labelPath, "r") as f:
            for line in f:
                if not foundSpeed and line.startswith("speed"):
                    speed = float(line.strip().split(" : ")[1])
                    foundSpeed = True
                elif not foundAcceleration and line.startswith("acceleration"):
                    acceleration = float(line.strip().split(" : ")[1])
                    foundAcceleration = True
                elif not foundTurnRate and line.startswith("turnRate"):
                    turnRate = float(line.strip().split(" : ")[1])
                    foundTurnRate = True

                if foundSpeed and foundAcceleration and foundTurnRate:
                    break

        return np.array([speed, acceleration, turnRate], dtype=np.float32)

    def _buildEgoHistoryMap(self):
        totalSamples = len(self.samples)
        egoHistoryArray = np.zeros((totalSamples, 8, 3), dtype=np.float32)
        clipGroups = {}
        progressModulo = 1000

        if self.verbose:
            print("\nPreparing clip groups for ego history...")

        for sampleIndex, sampleId in enumerate(self.samples):
            clipKey, frameOrder = self._parseSampleId(sampleId)
            clipGroups.setdefault(clipKey, []).append((frameOrder, sampleIndex))

            if self.verbose and ((sampleIndex + 1) % progressModulo == 0 or sampleIndex + 1 == totalSamples):
                completion = (sampleIndex + 1) / max(1, totalSamples)
                print(
                    f"{getProgressBar(completion, wheelIndex=sampleIndex, maxbarLength=60)}"
                    f"Grouping samples for ego history ({sampleIndex + 1}/{totalSamples})",
                    end='\r',
                    flush=True,
                )

        if self.verbose:
            print()
            print(f"\nBuilding ego history for {len(clipGroups)} clips...")

        processedSamples = 0

        for clipKey, entries in clipGroups.items():
            if any(frameOrder >= 0 for frameOrder, _ in entries):
                sortedEntries = sorted(entries, key=lambda item: item[0])
            else:
                sortedEntries = sorted(entries, key=lambda item: item[1])

            stateHistory = deque(maxlen=8)
            for _, sampleIndex in sortedEntries:
                sampleId = self.samples[sampleIndex]
                currentState = self._loadDynamicState(sampleId)
                stateHistory.append(currentState)

                statesList = list(stateHistory)
                if len(statesList) < 8:
                    firstState = statesList[0]
                    statesList = [firstState] * (8 - len(statesList)) + statesList

                egoHistoryArray[sampleIndex] = np.asarray(statesList, dtype=np.float32)

                processedSamples += 1
                if self.verbose and (processedSamples % progressModulo == 0 or processedSamples == totalSamples):
                    completion = processedSamples / max(1, totalSamples)
                    print(
                        f"{getProgressBar(completion, wheelIndex=processedSamples, maxbarLength=60)}"
                        f"Computing ego history ({processedSamples}/{totalSamples})",
                        end='\r',
                        flush=True,
                    )

        if self.verbose:
            print()

        return egoHistoryArray

    def _readFramesFromVideo(self, videoPath, prevFrameIndex, frameIndex):
        """Robust video frame reader with backend fallback and safe retries."""
        backendCandidates = [
            cv2.CAP_FFMPEG,
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            None,
        ]

        lastError = None
        for backend in backendCandidates:
            cap = cv2.VideoCapture(videoPath) if backend is None else cv2.VideoCapture(videoPath, backend)
            if not cap.isOpened():
                cap.release()
                continue

            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, prevFrameIndex)
                ret_prev, prevFrame = cap.read()

                cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                ret_curr, currFrame = cap.read()

                if not ret_prev or not ret_curr or prevFrame is None or currFrame is None:
                    lastError = RuntimeError(
                        f"Failed to read frames from video: {videoPath} ({prevFrameIndex}, {frameIndex})"
                    )
                    continue

                prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2RGB)
                currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
                prevFrame = cv2.resize(prevFrame, (640, 360))
                currFrame = cv2.resize(currFrame, (640, 360))
                return Image.fromarray(prevFrame), Image.fromarray(currFrame)

            finally:
                cap.release()

        if lastError is not None:
            raise lastError
        raise RuntimeError(f"Could not open video: {videoPath}")

    def __getitem__(self, idx):
        sampleId = self.samples[idx]
        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        vectors, vectorTimes, speed, acceleration, turnRate, videoPath, prevFrameIndex, frameIndex = self._parseLabelFile(labelPath)

        if self.labelsOnly and videoPath is not None and prevFrameIndex is not None and frameIndex is not None:
            prevImage, currentImage = self._readFramesFromVideo(videoPath, prevFrameIndex, frameIndex)
            if prevImage is None or currentImage is None:
                raise RuntimeError(f"Failed to read frames from video: {videoPath} ({prevFrameIndex}, {frameIndex})")
        else:
            prevImgPath = os.path.join(self.imagesDir, f"{sampleId}_prev.png")
            currentImgPath = os.path.join(self.imagesDir, f"{sampleId}_current.png")
            prevImage = Image.open(prevImgPath).convert("RGB")
            currentImage = Image.open(currentImgPath).convert("RGB")

        if self.transform:
            prevImage = self.transform(prevImage)
            currentImage = self.transform(currentImage)

        dynamicData = torch.tensor([speed, acceleration, turnRate], dtype=self.dtype)
        egoHistory = np.tile(np.array([speed, acceleration, turnRate], dtype=np.float32), (8, 1))
        if self.loadEgoHistory:
            self._ensureEgoHistoryArrayLoaded()
            if self.egoHistoryArray is not None and idx < len(self.egoHistoryArray):
                egoHistory = self.egoHistoryArray[idx]

        vectors = np.nan_to_num(vectors, nan=0.0, posinf=200.0, neginf=-200.0)
        vectorTimes = np.nan_to_num(vectorTimes, nan=0.1, posinf=3.0, neginf=0.1)
        vectorTimes = np.clip(vectorTimes, 1e-3, 10.0)
        egoHistory = np.nan_to_num(egoHistory, nan=0.0, posinf=100.0, neginf=-100.0)

        dynamicData = torch.nan_to_num(dynamicData, nan=0.0, posinf=100.0, neginf=-100.0)
        vectorsTensor = torch.from_numpy(vectors).to(self.dtype)
        vectorTimesTensor = torch.from_numpy(vectorTimes).to(self.dtype)
        egoHistoryTensor = torch.from_numpy(egoHistory).to(self.dtype)

        return prevImage, currentImage, dynamicData, vectorsTensor, vectorTimesTensor, egoHistoryTensor


def getRunDir(baseDir="training"):
    """
    Returns a unique directory path for a new training run.
    Creates the base directory and new run folder if needed.
    """
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)
    runNum = 1
    while os.path.exists(os.path.join(baseDir, f"run{runNum}")):
        runNum += 1
    runDir = os.path.join(baseDir, f"run{runNum}")
    os.makedirs(runDir)
    return runDir


def plotHistory(history, savePath):
    """Plots training and validation loss history."""
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(savePath, dpi=400)
    plt.close()


def trajectoryLossWithWeights(pred, target, perStepWeights, reduction="mean", delta=1.0):
    """
    Computes weighted Huber loss across predicted trajectory waypoints.

    Args:
        pred (Tensor): Predicted waypoints (B, T, 2)
        target (Tensor): Ground-truth waypoints (B, T, 2)
        per_step_weights (Tensor): Weight for each time step
        reduction (str): 'mean', 'sum', or 'none'
        delta (float): Huber delta

    Returns:
        Tensor: Weighted loss value
    """
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=delta)
    raw = loss_fn(pred, target).mean(dim=-1)
    weights = perStepWeights.to(raw.device).view(1, -1)
    weighted = raw * weights
    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    return weighted


def buildDeltaTimes(vectorTimes):
    if vectorTimes.dim() == 1:
        vectorTimes = vectorTimes.unsqueeze(0)
    deltaTimes = vectorTimes.clone()
    if vectorTimes.size(1) > 1:
        deltaTimes[:, 1:] = vectorTimes[:, 1:] - vectorTimes[:, :-1]
    deltaTimes = torch.clamp(deltaTimes, min=1e-3)
    return deltaTimes


def buildKinematicTargets(gtVectors, vectorTimes):
    deltaTimes = buildDeltaTimes(vectorTimes)
    lateralStep = gtVectors[:, :, 0]
    longitudinalStep = gtVectors[:, :, 1]

    signedLongitudinalSpeed = longitudinalStep / deltaTimes
    headings = torch.atan2(lateralStep, longitudinalStep + 1e-9)
    previousHeading = torch.zeros_like(headings)
    if headings.size(1) > 1:
        previousHeading[:, 1:] = headings[:, :-1]
    headingDelta = torch.atan2(torch.sin(headings - previousHeading), torch.cos(headings - previousHeading))
    yawRate = headingDelta / deltaTimes

    gtKinematic = torch.stack([signedLongitudinalSpeed, yawRate], dim=-1)
    gtPositions = torch.cumsum(gtVectors, dim=1)
    return gtKinematic, gtPositions


def kinematicIntegration(preds, vectorTimes):
    deltaTimes = buildDeltaTimes(vectorTimes)
    batchSize, predSteps, _ = preds.shape

    yaw = torch.zeros(batchSize, device=preds.device, dtype=preds.dtype)
    posX = torch.zeros(batchSize, device=preds.device, dtype=preds.dtype)
    posY = torch.zeros(batchSize, device=preds.device, dtype=preds.dtype)
    positions = []

    for stepIndex in range(predSteps):
        stepDt = deltaTimes[:, stepIndex]
        speed = preds[:, stepIndex, 0]
        yawRate = preds[:, stepIndex, 1]

        yaw = yaw + yawRate * stepDt
        posX = posX + speed * torch.sin(yaw) * stepDt
        posY = posY + speed * torch.cos(yaw) * stepDt
        positions.append(torch.stack([posX, posY], dim=-1))

    return torch.stack(positions, dim=1)


def smoothnessKinematicLoss(preds, gtVectors, gtPositions, perStepWeights, vectorTimes):
    mainLoss = trajectoryLossWithWeights(preds, gtVectors, perStepWeights, reduction="mean")
    integratedPositions = kinematicIntegration(preds, vectorTimes)
    integrationLoss = nn.SmoothL1Loss()(integratedPositions, gtPositions)

    jerkSpeed = torch.diff(torch.diff(preds[:, :, 0], dim=1), dim=1).abs().mean()
    jerkYaw = torch.diff(torch.diff(preds[:, :, 1], dim=1), dim=1).abs().mean()
    jerkLoss = 0.08 * (jerkSpeed + jerkYaw)

    return mainLoss + 0.45 * integrationLoss + jerkLoss


def adeFde(pred, target):
    """
    Computes ADE (Average Displacement Error) and FDE (Final Displacement Error).

    Args:
        pred (Tensor): Predicted waypoints (B, T, 2)
        target (Tensor): Ground-truth waypoints (B, T, 2)

    Returns:
        tuple(float, float): ADE, FDE
    """
    diff = pred - target
    dists = torch.norm(diff, dim=-1)
    ade = dists.mean(dim=1).mean().item()
    fde = dists[:, -1].mean().item()
    return ade, fde


def inferVectorTimesFromLabels(labelsDir):
    """
    Infer non-uniform vector times from the first label file that contains vectorTimes.
    Returns None if no valid vectorTimes are found.
    """
    if not os.path.isdir(labelsDir):
        return None

    labelFiles = sorted([f for f in os.listdir(labelsDir) if f.endswith(".txt")])
    for name in labelFiles:
        labelPath = os.path.join(labelsDir, name)
        try:
            with open(labelPath, "r") as f:
                for line in f:
                    if line.startswith("vectorTimes"):
                        timesLine = line.strip().split(" : ")[1]
                        if timesLine == "None":
                            break
                        timesList = [float(v) for v in timesLine.split(" ") if v]
                        if timesList:
                            return timesList
                        break
        except Exception:
            continue

    return None


def trainModel(
    datasetDir,
    numEpochs=60,
    batchSize=32,
    learningRate=3e-4,
    trainValSplit=0.8,
    patience=12,
    datasetMaxSize=None,
    gradAccumSteps=2,
    trainSamplesPerEpoch=None,
    valSamplesPerEpoch=None,
    seed=42,
    splitSeed=None,
    useAuxDyn=False,
    featDim=256,
    hiddenDim=256,
    baseChannels=32,
    numHeads=4,
    numLayers=2,
    predSteps=None,
    intervalSeconds=None,
    deviceOverride=None,
    resumeModelPath=None,
    stopAfterSubepochCallback=None,
):
    """
    Main training function for TrajectoryModel.

    Handles data loading, training, validation, early stopping, and saving checkpoints.

    args:
        datasetDir (str): Path to dataset directory.
        numEpochs (int): Number of training epochs.
        batchSize (int): Batch size for training.
        learningRate (float): Initial learning rate.
        trainValSplit (float): Proportion of data for training vs validation.
        patience (int): Early stopping patience.
        datasetMaxSize (int or None): Max number of samples to load from dataset.
        gradAccumSteps (int): Gradient accumulation steps.
        trainSamplesPerEpoch (int or None): Samples per epoch from train split.
        valSamplesPerEpoch (int or None): Samples per epoch from val split.
        seed (int): Base seed for reproducibility.
        splitSeed (int or None): Seed for train/val split (defaults to seed).
        useAuxDyn (bool): Whether to use auxiliary dynamics head.
        featDim (int): Feature dimension for model.
        hiddenDim (int): Hidden dimension for model.
        baseChannels (int): Base channel count for the backbone.
        numHeads (int): Number of attention heads in the decoder.
        numLayers (int): Number of recurrent planner layers.
        predSteps (int or None): Number of prediction steps (inferred from labels if None).
        intervalSeconds (float or None): Time interval between prediction steps (inferred from labels if None).
        deviceOverride (str or None): Device to use ('cpu' or 'cuda'), or None for auto-detect.
        resumeModelPath (str or None): Path to a previously trained model to resume training from.
        stopAfterSubepochCallback (callable or None): Optional callback called after each sub-epoch.
            Signature: callback(subEpochIndex, numSubEpochs) -> bool.
            If it returns True, training exits gracefully after checkpoint/history updates.
    """
    print()

    vectorTimes = None

    if resumeModelPath:
        print(f"Resuming training from model: '{resumeModelPath}'...")
        runDir = os.path.dirname(resumeModelPath)
        paramsPath = os.path.join(runDir, "training_params.json")
        historyPath = os.path.join(runDir, "training_history.csv")
        
        if not os.path.exists(paramsPath) or not os.path.exists(historyPath):
            raise FileNotFoundError(f"Cannot resume: missing 'training_params.json' or 'training_history.csv' in '{runDir}'")
        
        with open(paramsPath, "r") as f:
            loadedParams = json.load(f)
        # Set parameters from loaded, keeping passed values if missing
        if "datasetDir" in loadedParams:
            datasetDir = loadedParams["datasetDir"]
        if "numEpochs" in loadedParams:
            numEpochs = loadedParams["numEpochs"]
        if "batchSize" in loadedParams:
            batchSize = loadedParams["batchSize"]
        if "learningRate" in loadedParams:
            learningRate = loadedParams["learningRate"]
        if "trainValSplit" in loadedParams:
            trainValSplit = loadedParams["trainValSplit"]
        if "patience" in loadedParams:
            patience = loadedParams["patience"]
        if "datasetMaxSize" in loadedParams:
            datasetMaxSize = loadedParams["datasetMaxSize"]
        if "gradAccumSteps" in loadedParams:
            gradAccumSteps = loadedParams["gradAccumSteps"]
        if "trainSamplesPerEpoch" in loadedParams:
            trainSamplesPerEpoch = loadedParams["trainSamplesPerEpoch"]
        if "valSamplesPerEpoch" in loadedParams:
            valSamplesPerEpoch = loadedParams["valSamplesPerEpoch"]
        if "seed" in loadedParams:
            seed = loadedParams["seed"]
        if "splitSeed" in loadedParams:
            splitSeed = loadedParams["splitSeed"]
        if "useAuxDyn" in loadedParams:
            useAuxDyn = loadedParams["useAuxDyn"]
        if "featDim" in loadedParams:
            featDim = loadedParams["featDim"]
        if "hiddenDim" in loadedParams:
            hiddenDim = loadedParams["hiddenDim"]
        if "baseChannels" in loadedParams:
            baseChannels = loadedParams["baseChannels"]
        if "numHeads" in loadedParams:
            numHeads = loadedParams["numHeads"]
        if "numLayers" in loadedParams:
            numLayers = loadedParams["numLayers"]
        if "predSteps" in loadedParams:
            predSteps = loadedParams["predSteps"]
        if "intervalSeconds" in loadedParams:
            intervalSeconds = loadedParams["intervalSeconds"]
        if "vectorTimes" in loadedParams:
            vectorTimes = loadedParams["vectorTimes"]
        if "deviceOverride" in loadedParams:
            deviceOverride = loadedParams["deviceOverride"]

        historyDf = pd.read_csv(historyPath)
        history = historyDf.to_dict(orient='list')
        startEpoch = len(history["train_loss"])
        print(f"Resuming from epoch {startEpoch}")
    else:
        runDir = getRunDir()
        history = {"train_loss": [], "val_loss": [], "train_ADE": [], "val_ADE": [], "train_FDE": [], "val_FDE": []}
        startEpoch = 0

    if splitSeed is None:
        splitSeed = seed

    # Set fixed random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not os.path.exists(datasetDir):
        raise FileNotFoundError(f"Dataset directory '{datasetDir}' does not exist.")

    inferred = inferVectorTimesFromLabels(os.path.join(datasetDir, "labels"))
    if inferred:
        vectorTimes = inferred
        print(f"Using vectorTimes from labels: {vectorTimes}")
        predSteps = len(vectorTimes)
        intervalSeconds = float(vectorTimes[0]) if vectorTimes else intervalSeconds
    else:
        if predSteps is None:
            predSteps = 12
        if intervalSeconds is None:
            intervalSeconds = 0.25

    device = (
        torch.device(deviceOverride)
        if deviceOverride
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    useAmp = device.type == "cuda"
    ampDtype = torch.bfloat16 if (useAmp and torch.cuda.is_bf16_supported()) else torch.float16
    useGradScaler = useAmp and ampDtype == torch.float16
    if useAmp:
        print(f"AMP enabled for speed/stability: dtype={ampDtype}, grad_scaler={useGradScaler}")
    scaler = GradScaler(device='cuda', enabled=useGradScaler)

    trainTransform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valTransform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading dataset and preparing train/val split...")
    print("Loading base dataset for indexing...")
    baseDataset = DrivingDataset(datasetDir, transform=None, maxSize=datasetMaxSize, predSteps=predSteps, verbose=True, loadEgoHistory=False)
    print("Creating train dataset...")
    trainDataset = DrivingDataset(datasetDir, transform=trainTransform, maxSize=datasetMaxSize, predSteps=predSteps, verbose=False, loadEgoHistory=True)
    print("Creating validation dataset...")
    valDataset = DrivingDataset(datasetDir, transform=valTransform, maxSize=datasetMaxSize, predSteps=predSteps, verbose=False, loadEgoHistory=True)
    totalSamples = len(baseDataset)

    labelsOnlyMode = bool(trainDataset.labelsOnly or valDataset.labelsOnly)
    if labelsOnlyMode:
        # Video-decoding mode can spike VRAM/shared memory on Windows when using multiple workers
        # and hardware decoders. Force a conservative setup.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads=2;video_codec=h264"
        numWorkers = 0 if os.name == "nt" else 1
        pinMemory = False
        print(
            "Video-decoding dataset mode detected (no images folder). "
            f"Using num_workers={numWorkers}, pin_memory={pinMemory}, "
            "FFmpeg CPU decode to avoid decoder VRAM spikes."
        )
    else:
        numWorkers = 2 if os.name == "nt" else 4
        pinMemory = device.type == "cuda"

    balancedPath = os.path.join(datasetDir, "balanced.json")
    balancedData = None
    if os.path.exists(balancedPath):
        with open(balancedPath, "r") as f:
            balancedData = json.load(f)

    classBuckets = {
        "straight": balancedData.get("straight", []) if balancedData else [],
        "right": balancedData.get("right", []) if balancedData else [],
        "left": balancedData.get("left", []) if balancedData else [],
        "s": balancedData.get("s", []) if balancedData else [],
    }

    print("Mapping samples to class buckets for train/val split...")
    sampleIndexMap = {sampleId: idx for idx, sampleId in enumerate(baseDataset.samples)}
    classIndexBuckets = {key: [] for key in classBuckets}
    indexToClass = {}
    for className, sampleIds in classBuckets.items():
        for sampleId in sampleIds:
            idx = sampleIndexMap.get(sampleId)
            if idx is not None:
                classIndexBuckets[className].append(idx)
                indexToClass[idx] = className

    # FIXED: clip-aware train/val split respecting trainValSplit (clip-level grouping)
    # Group samples by clip key first, then split clips according to trainValSplit.
    clipGroups = {}
    for idx, sampleId in enumerate(baseDataset.samples):
        clipKey, _ = baseDataset._parseSampleId(sampleId)
        clipGroups.setdefault(clipKey, []).append(idx)

    clipKeys = list(clipGroups.keys())
    totalClips = len(clipKeys)
    # Ensure deterministic shuffling for split reproducibility
    rng = random.Random(splitSeed if splitSeed is not None else seed)
    rng.shuffle(clipKeys)

    print("Performing clip-aware train/val split...")
    if totalClips <= 1:
        trainClipKeys = clipKeys
        valClipKeys = []
    else:
        numTrainClips = max(1, min(totalClips - 1, int(round(totalClips * float(trainValSplit)))))

        # Compute per-class totals (only for samples present in indexToClass)
        classTotals = {k: 0 for k in classIndexBuckets}
        for idx, cls in indexToClass.items():
            if cls in classTotals:
                classTotals[cls] += 1

        desiredTrainClassCounts = {k: int(round(v * float(trainValSplit))) for k, v in classTotals.items()}

        # Precompute clip-level class counts
        clipClassCounts = {}
        for ck, idxs in clipGroups.items():
            counts = {k: 0 for k in classIndexBuckets}
            for ii in idxs:
                cls = indexToClass.get(ii)
                if cls in counts:
                    counts[cls] += 1
            clipClassCounts[ck] = counts

        # Greedy selection of clips to match desired class counts as closely as possible
        selectedTrainClips = set()
        currentTrainCounts = {k: 0 for k in classIndexBuckets}

        for _ in range(numTrainClips):
            bestClip = None
            bestScore = -1
            for ck in clipKeys:
                if ck in selectedTrainClips:
                    continue
                counts = clipClassCounts.get(ck, {})
                # Score = how many of the remaining desired class counts this clip would cover
                score = 0
                for cls in counts:
                    need = max(0, desiredTrainClassCounts.get(cls, 0) - currentTrainCounts.get(cls, 0))
                    score += min(counts.get(cls, 0), need)
                # Tie-breaker: random
                if score > bestScore or (score == bestScore and (bestClip is None or rng.random() < 0.5)):
                    bestScore = score
                    bestClip = ck

            if bestClip is None:
                remaining = [ck for ck in clipKeys if ck not in selectedTrainClips]
                if not remaining:
                    break
                bestClip = rng.choice(remaining)

            selectedTrainClips.add(bestClip)
            for cls in currentTrainCounts:
                currentTrainCounts[cls] += clipClassCounts.get(bestClip, {}).get(cls, 0)

        # If we didn't reach the desired number of train clips (rare), fill randomly
        remainingClips = [ck for ck in clipKeys if ck not in selectedTrainClips]
        while len(selectedTrainClips) < numTrainClips and remainingClips:
            pick = rng.choice(remainingClips)
            selectedTrainClips.add(pick)
            remainingClips.remove(pick)

        trainClipKeys = [ck for ck in clipKeys if ck in selectedTrainClips]
        valClipKeys = [ck for ck in clipKeys if ck not in selectedTrainClips]

    print("Building train/val index lists from clip partitions...")
    # Build train/val index lists from clip partitions
    trainIndices = []
    valIndices = []
    for ck in trainClipKeys:
        trainIndices.extend(clipGroups.get(ck, []))
    for ck in valClipKeys:
        valIndices.extend(clipGroups.get(ck, []))

    # Fallback: ensure we have at least one validation sample
    if len(valIndices) == 0 and len(trainIndices) > 1:
        # move one clip from train to val
        movedClip = trainClipKeys[-1]
        trainClipKeys = trainClipKeys[:-1]
        valClipKeys = [movedClip] + valClipKeys
        trainIndices = []
        valIndices = []
        for ck in trainClipKeys:
            trainIndices.extend(clipGroups.get(ck, []))
        for ck in valClipKeys:
            valIndices.extend(clipGroups.get(ck, []))

    valIndices = sorted(valIndices)
    trainIndices = sorted(trainIndices)

    # Compute clean sub-epoch schedule and adjust trainSamplesPerEpoch accordingly
    totalTrainSamples = len(trainIndices)
    actualTrainSamplesPerSub, numSubEpochs, desiredFullEpochs = computeCleanSubepochSchedule(
        totalTrainSamples, trainSamplesPerEpoch, minFullEpochs=5
    )
    # Update trainSamplesPerEpoch to the adjusted clean value
    trainSamplesPerEpoch = actualTrainSamplesPerSub

    def getTrainSubsetSize():
        if trainSamplesPerEpoch is None:
            return len(trainIndices)

        return max(1, min(int(trainSamplesPerEpoch), len(trainIndices)))

    def getValSubsetSize():
        if valSamplesPerEpoch is None:
            return len(valIndices)

        return max(1, min(int(valSamplesPerEpoch), len(valIndices)))

    def getTrainSubsetIndices(epoch):
        if trainSamplesPerEpoch is None or trainSamplesPerEpoch >= len(trainIndices):
            return list(trainIndices)

        subsetSize = getTrainSubsetSize()
        rng = random.Random(seed + epoch)
        return rng.sample(trainIndices, subsetSize)

    def getValSubsetIndices(epoch):
        if valSamplesPerEpoch is None or valSamplesPerEpoch >= len(valIndices):
            return list(valIndices)

        subsetSize = getValSubsetSize()
        return list(valIndices[:subsetSize])

    def buildTrainLoader(trainSubsetIndices):
        subset = Subset(trainDataset, trainSubsetIndices)
        loaderKwargs = {
            "batch_size": batchSize,
            "shuffle": True,  # FIXED: enable DataLoader shuffling for training
            "num_workers": numWorkers,
            "pin_memory": pinMemory,
        }
        if numWorkers > 0:
            loaderKwargs["persistent_workers"] = os.name != "nt"
            loaderKwargs["prefetch_factor"] = 2
        return DataLoader(subset, **loaderKwargs)

    def buildValLoader(valSubsetIndices):
        subset = Subset(valDataset, valSubsetIndices)
        loaderKwargs = {
            "batch_size": batchSize,
            "shuffle": False,
            "num_workers": numWorkers,
            "pin_memory": pinMemory,
        }
        if numWorkers > 0:
            loaderKwargs["persistent_workers"] = os.name != "nt"
            loaderKwargs["prefetch_factor"] = 2
        return DataLoader(subset, **loaderKwargs)

    def computeClassCounts(indices):
        counts = {"straight": 0, "right": 0, "left": 0, "s": 0}
        for idx in indices:
            className = indexToClass.get(idx)
            if className in counts:
                counts[className] += 1
        return counts

    model = TrajectoryModel(
        featDim=featDim,
        hiddenDim=hiddenDim,
        baseChannels=baseChannels,
        numHeads=numHeads,
        numLayers=numLayers,
        predSteps=predSteps,
        useAuxDyn=useAuxDyn,
        intervalSeconds=intervalSeconds,
        vectorTimes=vectorTimes,
    ).to(device)

    if resumeModelPath:
        model.load_state_dict(torch.load(resumeModelPath))
        print(f"Loaded model state from '{resumeModelPath}'")

    try:
        modelName = model.name
    except AttributeError:
        modelName = "I Guess We'll Never Know"
    
    # Export training parameters to JSON
    params = {
        "datasetDir": datasetDir,
        "numEpochs": numEpochs,
        "batchSize": batchSize,
        "learningRate": learningRate,
        "trainValSplit": trainValSplit,
        "patience": patience,
        "datasetMaxSize": datasetMaxSize,
        "gradAccumSteps": gradAccumSteps,
        "trainSamplesPerEpoch": trainSamplesPerEpoch,
        "valSamplesPerEpoch": valSamplesPerEpoch,
        "seed": seed,
        "splitSeed": splitSeed,
        "useAuxDyn": useAuxDyn,
        "featDim": featDim,
        "hiddenDim": hiddenDim,
        "baseChannels": baseChannels,
        "numHeads": numHeads,
        "numLayers": numLayers,
        "predSteps": predSteps,
        "intervalSeconds": intervalSeconds,
        "vectorTimes": model.vectorTimes,
        "deviceOverride": deviceOverride,
        "modelName": modelName,
        "actualTrainSamplesPerSub": actualTrainSamplesPerSub,
        "numSubEpochs": numSubEpochs,
        "desiredFullEpochs": desiredFullEpochs,
    }
    with open(os.path.join(runDir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Stable default optimizer for the attention-based planner
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)

    if resumeModelPath:
        # Set initial_lr for scheduler compatibility when resuming
        for group in optimizer.param_groups:
            group['initial_lr'] = learningRate

    trainSubsetSize = getTrainSubsetSize()
    batchesPerEpoch = math.ceil(trainSubsetSize / batchSize)
    totalSteps = math.ceil((batchesPerEpoch * numEpochs) / max(1, gradAccumSteps))
    warmupSteps = min(2000, max(50, int(0.05 * totalSteps)))  # Longer warmup helps the planner settle early

    def lrLambda(step):
        if step < warmupSteps:
            return step / max(1, warmupSteps)
        progress = (step - warmupSteps) / max(1, totalSteps - warmupSteps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    stepsPerEpoch = math.ceil(batchesPerEpoch / max(1, gradAccumSteps))
    globalStep = startEpoch * stepsPerEpoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lrLambda, last_epoch=globalStep - 1)

    modelVectorTimes = model.vectorTimes
    if modelVectorTimes and len(modelVectorTimes) == predSteps:
        vectorTimesTensor = torch.tensor(modelVectorTimes, dtype=torch.float32)
        maxT = float(vectorTimesTensor.max()) if len(modelVectorTimes) else 1.0
        perStepWeightsBase = torch.exp(-vectorTimesTensor / max(maxT, 1e-6)) * 1.5 + 0.3
    else:
        perStepWeightsBase = torch.tensor([1.6, 1.3, 1.0, 0.8, 0.6, 0.5] + [0.4] * (predSteps - 6), dtype=torch.float32)
    
    auxDynWeight = 0.02
    teacherForcingStart = 0.9
    teacherForcingEnd = 0.0
    tfDecayEpochs = 30

    # Non-finite safety policy: tolerate occasional bad batches but stop sustained instability.
    if labelsOnlyMode and device.type == "cuda":
        nonFiniteFractionLimit = 0.03
        nonFiniteMinLimit = 6
    else:
        nonFiniteFractionLimit = 0.05
        nonFiniteMinLimit = 8
    maxConsecutiveNonFiniteTrainBatches = 3

    bestValLoss = float("inf")
    epochsNoImprove = 0
    # Use the creation time of the saved training params as the overall training start
    # when resuming, so the displayed `Time: HH:mm:ss` reflects total time since
    # the original run started. Fall back to now if unavailable.
    paramsPath = os.path.join(runDir, "training_params.json")
    if resumeModelPath and os.path.exists(paramsPath):
        try:
            trainingStartTime = os.path.getctime(paramsPath)
        except Exception:
            trainingStartTime = time.time()
    else:
        trainingStartTime = time.time()

    if resumeModelPath:
        bestValLoss = min(history["val_loss"])
        epochsNoImprove = len(history["val_loss"]) - history["val_loss"].index(bestValLoss) - 1

    stoppedByScheduler = False
    stoppedByNonFinite = False

    for subEpoch in range(startEpoch, numSubEpochs):
        trainSubsetIndices = getTrainSubsetIndices(subEpoch)
        valSubsetIndices = getValSubsetIndices(subEpoch)
        trainLoader = buildTrainLoader(trainSubsetIndices)
        numTrainBatches = max(1, len(trainLoader))
        maxNonFiniteTrainBatchesPerSubepoch = max(
            nonFiniteMinLimit,
            int(math.ceil(nonFiniteFractionLimit * numTrainBatches)),
        )

        model.train()
        runningLoss, runningADE, runningFDE = 0.0, 0.0, 0.0
        epochStartTime = time.time()

        trainCounts = computeClassCounts(trainSubsetIndices)
        valCounts = computeClassCounts(valSubsetIndices)
        print(
            f"Class distribution - Train: straight={trainCounts['straight']} right={trainCounts['right']} "
            f"left={trainCounts['left']} s={trainCounts['s']} | "
            f"Val: straight={valCounts['straight']} right={valCounts['right']} "
            f"left={valCounts['left']} s={valCounts['s']}"
        )

        # Map sub-epoch index to full epoch count for scheduling/TF decay
        if subEpoch < tfDecayEpochs:
            tfRatio = teacherForcingStart - (subEpoch / tfDecayEpochs) * (teacherForcingStart - teacherForcingEnd)
        else:
            tfRatio = teacherForcingEnd

        fullEpoch = (subEpoch * trainSamplesPerEpoch) // max(1, totalTrainSamples)

        print(f"\n###### Sub-epoch {subEpoch + 1}/{numSubEpochs} (Full epoch {fullEpoch + 1}/{desiredFullEpochs}) - TF Ratio: {tfRatio:.2f} ######")
        print(
            "Non-finite guard: "
            f"max_per_subepoch={maxNonFiniteTrainBatchesPerSubepoch}, "
            f"max_consecutive={maxConsecutiveNonFiniteTrainBatches}"
        )

        optimizer.zero_grad()
        accumSteps = 0
        trainNonFiniteBatches = 0
        consecutiveNonFiniteTrainBatches = 0
        processedTrainSamples = 0

        for i, (prevImg, currentImg, dynamicData, labels, vectorTimesBatch, egoHistory) in enumerate(trainLoader):
            prevImg = prevImg.to(device)
            currentImg = currentImg.to(device)
            labels = labels.to(device)
            vectorTimesBatch = vectorTimesBatch.to(device)
            egoHistory = egoHistory.to(device)
            vectorTimesBatch = torch.nan_to_num(vectorTimesBatch, nan=0.1, posinf=3.0, neginf=0.1)
            vectorTimesBatch = torch.clamp(vectorTimesBatch, min=1e-3, max=10.0)
            if not torch.isfinite(egoHistory).all():
                egoHistory = torch.nan_to_num(egoHistory, nan=0.0, posinf=100.0, neginf=-100.0)
                egoHistory = torch.clamp(egoHistory, min=-100.0, max=100.0)

            batchSizeCurrent = labels.size(0)

            gtKinematic, gtPositions = buildKinematicTargets(labels, vectorTimesBatch)

            perStepWeights = perStepWeightsBase
            if vectorTimesBatch.numel() and torch.any(vectorTimesBatch > 0):
                meanTimes = vectorTimesBatch.mean(dim=0)
                maxT = float(meanTimes.max()) if meanTimes.numel() else 1.0
                perStepWeights = torch.exp(-meanTimes / max(maxT, 1e-6)) * 1.5 + 0.3

            with autocast(device_type='cuda', dtype=ampDtype, enabled=useAmp):
                preds, auxOut, _ = model(
                    currentImg,
                    prevImg,
                    gtTraj=gtKinematic,
                    teacherForcing=True,
                    tfRatio=tfRatio,
                    egoHistory=egoHistory,
                    vectorTimes=vectorTimesBatch,
                    returnAttn=False,
                )

                # Keep non-finite diagnostics lightweight to avoid severe console slowdown.
                finiteMask = torch.isfinite(preds)
                if not finiteMask.all():
                    trainNonFiniteBatches += 1
                    consecutiveNonFiniteTrainBatches += 1
                    if trainNonFiniteBatches <= 3 or trainNonFiniteBatches % 50 == 0:
                        finiteRatio = finiteMask.float().mean().item()
                        print(
                            f"Non-finite predictions at full epoch {fullEpoch + 1}, batch {i}. "
                            f"Skipped train batches this sub-epoch: {trainNonFiniteBatches}. "
                            f"Consecutive non-finite batches: {consecutiveNonFiniteTrainBatches}. "
                            f"Finite ratio in preds: {finiteRatio:.4f}"
                        )

                    # Reset partial gradient accumulation and release cached blocks after bad batches.
                    if accumSteps > 0:
                        optimizer.zero_grad(set_to_none=True)
                        accumSteps = 0
                    if device.type == "cuda" and trainNonFiniteBatches % 10 == 0:
                        torch.cuda.empty_cache()

                    if (
                        trainNonFiniteBatches >= maxNonFiniteTrainBatchesPerSubepoch
                        or consecutiveNonFiniteTrainBatches >= maxConsecutiveNonFiniteTrainBatches
                    ):
                        print(
                            "Reached non-finite batch safety limit for this sub-epoch. "
                            "Stopping run early to prevent VRAM runaway."
                        )
                        stoppedByNonFinite = True
                        break
                    continue

                consecutiveNonFiniteTrainBatches = 0
                
                lossMain = smoothnessKinematicLoss(preds, gtKinematic, gtPositions, perStepWeights, vectorTimesBatch)
                if not torch.isfinite(lossMain):
                    lossMain = torch.tensor(0.0, device=device, dtype=preds.dtype)
                loss = lossMain

                if useAuxDyn and auxOut is not None:
                    auxLoss = torch.tensor(0.0, device=device)
                    loss += auxDynWeight * auxLoss

            loss = loss / gradAccumSteps
            scaler.scale(loss).backward()
            accumSteps += 1

            if accumSteps == gradAccumSteps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaleBeforeStep = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimizerStepped = scaler.get_scale() >= scaleBeforeStep
                if optimizerStepped:
                    scheduler.step()
                    globalStep += 1
                accumSteps = 0

            processedTrainSamples += batchSizeCurrent
            runningLoss += lossMain.item() * batchSizeCurrent
            predPositions = kinematicIntegration(preds.detach(), vectorTimesBatch)
            ade, fde = adeFde(predPositions, gtPositions)
            runningADE += ade * batchSizeCurrent
            runningFDE += fde * batchSizeCurrent

            completion = (i + 1) / len(trainLoader)
            epochElapsed = time.time() - epochStartTime
            batchesDone = i + 1
            timePerBatch = epochElapsed / max(1, batchesDone)
            eta = (len(trainLoader) - batchesDone) * timePerBatch
            etaFinish = datetime.fromtimestamp(time.time() + eta)
            etaTime = etaFinish.strftime("%Y-%m-%d %H:%M:%S")
            totalElapsed = time.time() - trainingStartTime
            h, rem = divmod(int(totalElapsed), 3600)
            m, s = divmod(rem, 60)
            elapsedFmt = f"{h:02d}:{m:02d}:{s:02d}"

            avgLoss = runningLoss / max(1, processedTrainSamples)
            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)}"
                f"Avg Train Loss: {avgLoss:.4f} - ETA: {etaTime} - Time: {elapsedFmt}", end="\r")

        if stoppedByNonFinite:
            if accumSteps > 0:
                optimizer.zero_grad(set_to_none=True)
                accumSteps = 0
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("Training stopped early due to non-finite predictions.")
            break

        if trainNonFiniteBatches > 0:
            print(f"\nSkipped {trainNonFiniteBatches} train batches due to non-finite predictions in this sub-epoch.")

        trainLoss = runningLoss / len(trainLoader.dataset)
        trainADE = runningADE / len(trainLoader.dataset)
        trainFDE = runningFDE / len(trainLoader.dataset)
        history["train_loss"].append(trainLoss)
        history["train_ADE"].append(trainADE)
        history["train_FDE"].append(trainFDE)

        model.eval()
        valLoss, valADE, valFDE = 0.0, 0.0, 0.0
        valLoader = buildValLoader(valSubsetIndices)

        print('\n\nValidation Progress:')

        valStartTime = time.time()
        bucketTotals = {"straight": 0, "right": 0, "left": 0, "s": 0}
        bucketAde = {"straight": 0.0, "right": 0.0, "left": 0.0, "s": 0.0}
        bucketFde = {"straight": 0.0, "right": 0.0, "left": 0.0, "s": 0.0}
        valCursor = 0
        valNonFiniteBatches = 0
        processedValSamples = 0

        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels, vectorTimesBatch, egoHistory) in enumerate(valLoader):
                prevImg = prevImg.to(device)
                currentImg = currentImg.to(device)
                labels = labels.to(device)
                vectorTimesBatch = vectorTimesBatch.to(device)
                egoHistory = egoHistory.to(device)
                vectorTimesBatch = torch.nan_to_num(vectorTimesBatch, nan=0.1, posinf=3.0, neginf=0.1)
                vectorTimesBatch = torch.clamp(vectorTimesBatch, min=1e-3, max=10.0)
                if not torch.isfinite(egoHistory).all():
                    egoHistory = torch.nan_to_num(egoHistory, nan=0.0, posinf=100.0, neginf=-100.0)
                    egoHistory = torch.clamp(egoHistory, min=-100.0, max=100.0)

                batchSizeCurrent = labels.size(0)
                batchIndices = valSubsetIndices[valCursor:valCursor + batchSizeCurrent]
                valCursor += batchSizeCurrent

                gtKinematic, gtPositions = buildKinematicTargets(labels, vectorTimesBatch)

                perStepWeights = perStepWeightsBase
                if vectorTimesBatch.numel() and torch.any(vectorTimesBatch > 0):
                    meanTimes = vectorTimesBatch.mean(dim=0)
                    maxT = float(meanTimes.max()) if meanTimes.numel() else 1.0
                    perStepWeights = torch.exp(-meanTimes / max(maxT, 1e-6)) * 1.5 + 0.3
                with autocast(device_type='cuda', dtype=ampDtype, enabled=useAmp):
                    preds, _, _ = model(
                        currentImg,
                        prevImg,
                        teacherForcing=False,
                        tfRatio=0.0,
                        egoHistory=egoHistory,
                        vectorTimes=vectorTimesBatch,
                        returnAttn=False,
                    )

                    finiteMask = torch.isfinite(preds)
                    if not finiteMask.all():
                        valNonFiniteBatches += 1
                        if valNonFiniteBatches <= 3 or valNonFiniteBatches % 50 == 0:
                            finiteRatio = finiteMask.float().mean().item()
                            print(
                                f"Non-finite validation predictions at full epoch {fullEpoch + 1}, batch {i}. "
                                f"Skipped val batches this sub-epoch: {valNonFiniteBatches}. "
                                f"Finite ratio in preds: {finiteRatio:.4f}"
                            )
                        if device.type == "cuda" and valNonFiniteBatches % 10 == 0:
                            torch.cuda.empty_cache()
                        continue
                    
                    lossMain = smoothnessKinematicLoss(preds, gtKinematic, gtPositions, perStepWeights, vectorTimesBatch)
                    if not torch.isfinite(lossMain):
                        continue
                processedValSamples += batchSizeCurrent
                valLoss += lossMain.item() * batchSizeCurrent
                predPositions = kinematicIntegration(preds, vectorTimesBatch)
                ade, fde = adeFde(predPositions, gtPositions)
                valADE += ade * batchSizeCurrent
                valFDE += fde * batchSizeCurrent

                diff = predPositions - gtPositions
                dists = torch.norm(diff, dim=-1)
                adePerSample = dists.mean(dim=1).detach().cpu().numpy()
                fdePerSample = dists[:, -1].detach().cpu().numpy()
                for sampleIdx, sampleGlobalIdx in enumerate(batchIndices):
                    className = indexToClass.get(sampleGlobalIdx)
                    if className in bucketTotals:
                        bucketTotals[className] += 1
                        bucketAde[className] += float(adePerSample[sampleIdx])
                        bucketFde[className] += float(fdePerSample[sampleIdx])
                completion = (i + 1) / len(valLoader)
                
                batchesDone = i + 1
                valElapsed = time.time() - valStartTime
                valTimePerBatch = valElapsed / max(1, batchesDone)
                valEta = (len(valLoader) - batchesDone) * valTimePerBatch
                valEtaFinish = datetime.fromtimestamp(time.time() + valEta)
                valEtaTime = valEtaFinish.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)}"
                    f"Avg Val Loss: {valLoss/max(1, processedValSamples):.4f} (best: {bestValLoss:.4f})"
                    f" - ETA: {valEtaTime}", end="\r")

        if valNonFiniteBatches > 0:
            print(f"\nSkipped {valNonFiniteBatches} validation batches due to non-finite predictions in this sub-epoch.")

        valLoss /= len(valLoader.dataset)
        valADE /= len(valLoader.dataset)
        valFDE /= len(valLoader.dataset)
        history["val_loss"].append(valLoss)
        history["val_ADE"].append(valADE)
        history["val_FDE"].append(valFDE)

        print(f"\nTrain Loss: {trainLoss:.4f} | "
              f"Val Loss: {valLoss:.4f} | ADE: {valADE:.4f} | FDE: {valFDE:.4f}")

        bucketReport = []
        for className in ["straight", "right", "left", "s"]:
            count = bucketTotals[className]
            if count > 0:
                bucketReport.append(
                    f"{className}: ADE={bucketAde[className] / count:.3f}, "
                    f"FDE={bucketFde[className] / count:.3f} ({count})"
                )
            else:
                bucketReport.append(f"{className}: n/a (0)")
        print("Val per-bucket: " + " | ".join(bucketReport))

        historyDf = pd.DataFrame(history)
        historyDf.to_csv(os.path.join(runDir, "training_history.csv"), index=False)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            checkpoint = {
                'epoch': fullEpoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': bestValLoss,
                'epochs_no_improve': epochsNoImprove,
                'history': history
            }
            torch.save(checkpoint, os.path.join(runDir, "best_checkpoint.pth"))
            torch.save(model.state_dict(), os.path.join(runDir, "best_model.pth"))  # Keep for compatibility
            print(f"New best model saved: {bestValLoss:.4f}")
            epochsNoImprove = 0
        else:
            checkpoint = {
                'epoch': fullEpoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': bestValLoss,
                'epochs_no_improve': epochsNoImprove,
                'history': history
            }
            torch.save(checkpoint, os.path.join(runDir, "last_checkpoint.pth"))
            torch.save(model.state_dict(), os.path.join(runDir, "last_model.pth"))  # Keep for compatibility
            epochsNoImprove += 1

        if epochsNoImprove >= patience:
            print(f"Early stopping after {patience} epochs with no improvement.")
            break

        if stopAfterSubepochCallback is not None:
            shouldStop = bool(stopAfterSubepochCallback(subEpoch + 1, numSubEpochs))
            if shouldStop:
                print("Stopping training after current sub-epoch as requested by scheduler.")
                stoppedByScheduler = True
                break

        print("\n")

    plotHistory(history, os.path.join(runDir, "loss_plot.png"))

    lastModelPath = os.path.join(runDir, "last_model.pth")
    bestModelPath = os.path.join(runDir, "best_model.pth")
    if os.path.exists(lastModelPath):
        resumePath = lastModelPath
    elif os.path.exists(bestModelPath):
        resumePath = bestModelPath
    else:
        resumePath = resumeModelPath

    if stoppedByScheduler:
        print("Training paused by scheduler.")
    elif stoppedByNonFinite:
        print("Training paused by non-finite safety guard.")
    else:
        print("Training completed.")

    return {
        "runDir": runDir,
        "resumeModelPath": resumePath,
        "stoppedByScheduler": stoppedByScheduler,
        "stoppedByNonFinite": stoppedByNonFinite,
    }


if __name__ == "__main__":
    """
    PLEASE MAKE SURE TO USE A CORRECT DATASET (like image size, time window, etc.)
    """
    datasetPath = r"C:\Users\Projet_3NC\Desktop\SC-ADS\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360"
    datasetMaxSize = None           # Maximum number of samples to load from the dataset (None = use all available)
    numEpochs = 2_000               # ~1 full pass at 10k samples/epoch for ~9.6M samples
    patience = 100                  # Early stopping patience (stop if no val improvement for this many epochs)
    batchSize = 24                  # Number of samples per training batch (controls GPU memory usage)
    gradAccumSteps = 1              # Gradient accumulation steps (simulates larger effective batch if >1)
    trainValSplit = 0.8             # Train/validation split ratio
    trainSamplesPerEpoch = 8_000    # Random samples per epoch for fast iterations
    valSamplesPerEpoch = trainSamplesPerEpoch * 2                          # Fixed val samples per epoch (2x trainSamplesPerEpoch)
    seed = 42                       # Base seed for reproducibility
    splitSeed = 42                  # Train/val split seed (keep fixed to avoid contamination)
    learningRate = 5e-5             # Conservative default for stable planner training
    featDim = 384                   # Feature dimension in the model
    hiddenDim = 768                 # Hidden dimension in the model
    baseChannels = 48               # Backbone width (increases compute per frame)
    numHeads = 8                    # Cross-attention heads
    numLayers = 4                   # Recurrent planner depth
    useAuxDyn = False               # Whether to enable auxiliary dynamics head (speed/accel prediction)
    resumeModelPath =  r"C:\Users\Projet_3NC\Desktop\SC-ADS\Autopilot\training\run23\last_model.pth"  # Set to path like "training/run13/best_model.pth" to resume training

    trainModel(
        datasetDir=datasetPath,
        numEpochs=numEpochs, 
        batchSize=batchSize,
        learningRate=learningRate,
        trainValSplit=trainValSplit,
        patience=patience,
        datasetMaxSize=datasetMaxSize,
        gradAccumSteps=gradAccumSteps,
        trainSamplesPerEpoch=trainSamplesPerEpoch,
        valSamplesPerEpoch=valSamplesPerEpoch,
        seed=seed,
        splitSeed=splitSeed,
        useAuxDyn=useAuxDyn,
        featDim=featDim,
        hiddenDim=hiddenDim,
        baseChannels=baseChannels,
        numHeads=numHeads,
        numLayers=numLayers,
        resumeModelPath=resumeModelPath,
    )