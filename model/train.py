import os
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
import traceback

from CreateModel import TrajectoryModel
from progressBar import getProgressBar

# Set multiprocessing start method to 'spawn' for Windows compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Enable expandable segments for better memory management when supported.
if torch.cuda.is_available() and os.name != "nt" and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DrivingDataset(Dataset):
    """
    Dataset for trajectory prediction.
    Loads paired images (previous and current) and trajectory vectors.
    Optionally filters samples using 'balanced.json' if present.
    """

    def __init__(self, datasetDir, transform=None, maxSize=None, predSteps=6, dtype=torch.float32, verbose=True):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, "labels")
        self.imagesDir = os.path.join(datasetDir, "images")
        self.transform = transform
        self.predSteps = predSteps
        self.dtype = dtype
        self.maxSize = maxSize
        self.labelsOnly = False
        self.verbose = verbose

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

    def __len__(self):
        if self.maxSize is not None:
            return min(len(self.samples), self.maxSize)
        return len(self.samples)

    def _parseLabelFile(self, labelPath):
        vectors = None
        vectorTimes = None
        speed = 0.0
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

        return vectors, vectorTimes, speed, videoPath, prevFrameIndex, frameIndex

    def _readFramesFromVideo(self, videoPath, prevFrameIndex, frameIndex):
        cap = None
        try:
            cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(videoPath)
            if not cap.isOpened():
                print(f"[DrivingDataset] Unable to open video: {videoPath}")
                return None, None

            cap.set(cv2.CAP_PROP_POS_FRAMES, prevFrameIndex)
            retPrev, prevFrame = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            retCurr, currFrame = cap.read()

            if not retPrev or not retCurr:
                print(f"[DrivingDataset] Failed to read frames from {videoPath}: prev={retPrev}, curr={retCurr} ({prevFrameIndex},{frameIndex})")
                return None, None

            prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2RGB)
            currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
            prevFrame = cv2.resize(prevFrame, (640, 360))
            currFrame = cv2.resize(currFrame, (640, 360))
            return Image.fromarray(prevFrame), Image.fromarray(currFrame)
        except Exception as e:
            print(f"[DrivingDataset] Exception reading video {videoPath}: {e}")
            traceback.print_exc()
            return None, None
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def __getitem__(self, idx):
        # Wrap __getitem__ to ensure any unexpected exception is printed (not silently swallowed)
        sampleId = self.samples[idx]
        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        try:
            vectors, vectorTimes, speed, videoPath, prevFrameIndex, frameIndex = self._parseLabelFile(labelPath)

            if self.labelsOnly and videoPath is not None and prevFrameIndex is not None and frameIndex is not None:
                prevImage, currentImage = self._readFramesFromVideo(videoPath, prevFrameIndex, frameIndex)
                if prevImage is None or currentImage is None:
                    raise RuntimeError(f"Failed to read frames from video: {videoPath} ({prevFrameIndex}, {frameIndex})")
            else:
                prevImgPath = os.path.join(self.imagesDir, f"{sampleId}_prev.png")
                currentImgPath = os.path.join(self.imagesDir, f"{sampleId}_current.png")
                try:
                    prevImage = Image.open(prevImgPath).convert("RGB")
                    currentImage = Image.open(currentImgPath).convert("RGB")
                except Exception as e:
                    print(f"[DrivingDataset] Error opening images for sample {sampleId}: {prevImgPath}, {currentImgPath} -> {e}")
                    traceback.print_exc()
                    raise

            if self.transform:
                prevImage = self.transform(prevImage)
                currentImage = self.transform(currentImage)

            dynamicData = torch.tensor([speed], dtype=self.dtype)

            return prevImage, currentImage, dynamicData, torch.from_numpy(vectors).to(self.dtype), torch.from_numpy(vectorTimes).to(self.dtype)
        except Exception as e:
            print(f"[DrivingDataset] Exception in __getitem__ for sample {sampleId}: {e}")
            traceback.print_exc()
            raise


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
    plt.savefig(savePath)
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
    lossFn = nn.SmoothL1Loss(reduction="none", beta=delta)
    raw = lossFn(pred, target).mean(dim=-1)
    weights = perStepWeights.to(raw.device).view(1, -1)
    weighted = raw * weights
    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    return weighted


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
        numLayers (int): Number of transformer decoder layers.
        predSteps (int or None): Number of prediction steps (inferred from labels if None).
        intervalSeconds (float or None): Time interval between prediction steps (inferred from labels if None).
        deviceOverride (str or None): Device to use ('cpu' or 'cuda'), or None for auto-detect.
        resumeModelPath (str or None): Path to a previously trained model to resume training from.
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
    scaler = GradScaler(device='cuda', enabled=useAmp)

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

    baseDataset = DrivingDataset(datasetDir, transform=None, maxSize=datasetMaxSize, predSteps=predSteps, verbose=True)
    trainDataset = DrivingDataset(datasetDir, transform=trainTransform, maxSize=datasetMaxSize, predSteps=predSteps, verbose=False)
    valDataset = DrivingDataset(datasetDir, transform=valTransform, maxSize=datasetMaxSize, predSteps=predSteps, verbose=False)
    totalSamples = len(baseDataset)

    # Default workers; on Windows use 0 to avoid multiprocessing/persistent_workers issues
    numWorkers = 1
    dataloaderTimeout = 60 # s
    if os.name == 'nt':
        numWorkers = 0
        dataloaderPinMemory = False
        dataloaderPersistentWorkers = False
    else:
        dataloaderPinMemory = True
        # persistent workers can cause hangs; keep disabled by default
        dataloaderPersistentWorkers = False

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

    sampleIndexMap = {sampleId: idx for idx, sampleId in enumerate(baseDataset.samples)}
    classIndexBuckets = {key: [] for key in classBuckets}
    indexToClass = {}
    for className, sampleIds in classBuckets.items():
        for sampleId in sampleIds:
            idx = sampleIndexMap.get(sampleId)
            if idx is not None:
                classIndexBuckets[className].append(idx)
                indexToClass[idx] = className

    def buildStratifiedValIndices():
        if not balancedData:
            return []

        totalCount = sum(len(v) for v in classIndexBuckets.values())
        if totalCount == 0:
            return []

        if trainSamplesPerEpoch is not None:
            targetValCount = max(1, int(round(trainSamplesPerEpoch * 2)))
        else:
            targetValCount = max(1, int(round(totalSamples * (2.0 / 3.0))))

        rng = random.Random(seed)
        valIndices = []
        remainingSlots = targetValCount
        remainders = []

        for className, indices in classIndexBuckets.items():
            proportion = len(indices) / totalCount if totalCount else 0.0
            exactCount = proportion * targetValCount
            classTarget = int(math.floor(exactCount))
            remainders.append((exactCount - classTarget, className))
            if classTarget > 0:
                pickCount = min(classTarget, len(indices))
                valIndices.extend(rng.sample(indices, pickCount))
                remainingSlots -= pickCount

        remainders.sort(reverse=True)
        for _, className in remainders:
            if remainingSlots <= 0:
                break
            indices = classIndexBuckets[className]
            available = [i for i in indices if i not in valIndices]
            if not available:
                continue
            pickCount = min(remainingSlots, len(available))
            valIndices.extend(rng.sample(available, pickCount))
            remainingSlots -= pickCount

        if remainingSlots > 0:
            allIndices = [i for i in range(totalSamples) if i not in valIndices]
            if allIndices:
                pickCount = min(remainingSlots, len(allIndices))
                valIndices.extend(rng.sample(allIndices, pickCount))

        valIndices.sort()
        return valIndices

    valIndices = buildStratifiedValIndices()
    if not valIndices:
        valIndices = []
        if trainSamplesPerEpoch is not None:
            targetValCount = max(1, int(round(trainSamplesPerEpoch * 2)))
        else:
            targetValCount = max(1, int(round(totalSamples * (2.0 / 3.0))))
        rng = random.Random(seed)
        valIndices = rng.sample(range(totalSamples), min(targetValCount, totalSamples))
    valIndexSet = set(valIndices)
    trainIndices = [i for i in range(totalSamples) if i not in valIndexSet]

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
        loaderKwargs = dict(batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                             pin_memory=dataloaderPinMemory, persistent_workers=dataloaderPersistentWorkers,
                             timeout=dataloaderTimeout)
        if numWorkers > 0:
            loaderKwargs['prefetch_factor'] = 2
        return DataLoader(subset, **loaderKwargs)

    def buildValLoader(valSubsetIndices):
        subset = Subset(valDataset, valSubsetIndices)
        loaderKwargs = dict(batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                             pin_memory=dataloaderPinMemory, persistent_workers=dataloaderPersistentWorkers,
                             timeout=dataloaderTimeout)
        if numWorkers > 0:
            loaderKwargs['prefetch_factor'] = 2
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
    }
    with open(os.path.join(runDir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Transformer-friendly optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)

    if resumeModelPath:
        # Set initial_lr for scheduler compatibility when resuming
        for group in optimizer.param_groups:
            group['initial_lr'] = learningRate

    trainSubsetSize = getTrainSubsetSize()
    batchesPerEpoch = math.ceil(trainSubsetSize / batchSize)
    totalSteps = math.ceil((batchesPerEpoch * numEpochs) / max(1, gradAccumSteps))
    warmupSteps = min(2000, max(50, int(0.05 * totalSteps)))  # Longer warmup for transformer

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
    tfDecayEpochs = min(30, max(5, int(0.2 * numEpochs)))

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

    for epoch in range(startEpoch, numEpochs):
        trainSubsetIndices = getTrainSubsetIndices(epoch)
        valSubsetIndices = getValSubsetIndices(epoch)
        trainLoader = buildTrainLoader(trainSubsetIndices)
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

        tfRatio = (
            teacherForcingStart - (epoch / tfDecayEpochs) * (teacherForcingStart - teacherForcingEnd)
            if epoch < tfDecayEpochs
            else teacherForcingEnd
        )

        print(f"###### Epoch {epoch + 1}/{numEpochs} - TF Ratio: {tfRatio:.2f} ######")

        optimizer.zero_grad()
        accumSteps = 0

        for i, (prevImg, currentImg, dynamicData, labels, vectorTimesBatch) in enumerate(trainLoader):
            prevImg = prevImg.to(device)
            currentImg = currentImg.to(device)
            labels = labels.to(device)
            vectorTimesBatch = vectorTimesBatch.to(device)
            batchSize = labels.size(0)

            perStepWeights = perStepWeightsBase
            if vectorTimesBatch.numel() and torch.any(vectorTimesBatch > 0):
                meanTimes = vectorTimesBatch.mean(dim=0)
                maxT = float(meanTimes.max()) if meanTimes.numel() else 1.0
                perStepWeights = torch.exp(-meanTimes / max(maxT, 1e-6)) * 1.5 + 0.3

            with autocast(device_type='cuda', enabled=useAmp):
                preds, auxOut, _ = model(currentImg, prevImg, gtTraj=labels, teacherForcing=True, tfRatio=tfRatio, vectorTimes=vectorTimesBatch)
                
                # Check for NaN in predictions
                if torch.isnan(preds).any():
                    print(f"NaN detected in predictions at epoch {epoch+1}, batch {i}")
                    print(f"Preds: {preds}")
                    print(f"Labels: {labels}")
                    print(f"PrevImg stats: min={prevImg.min()}, max={prevImg.max()}, mean={prevImg.mean()}")
                    print(f"CurrentImg stats: min={currentImg.min()}, max={currentImg.max()}, mean={currentImg.mean()}")
                    continue  # or break to stop training
                
                lossMain = trajectoryLossWithWeights(preds, labels, perStepWeights, reduction="mean")
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
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                globalStep += 1
                accumSteps = 0

            runningLoss += lossMain.item() * batchSize
            ade, fde = adeFde(preds.detach(), labels)
            runningADE += ade * batchSize
            runningFDE += fde * batchSize

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

            avgLoss = runningLoss / ((i + 1) * batchSize)
            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)}"
                  f"Avg Train Loss: {avgLoss/batchesDone:.4f} - ETA: {etaTime} - Time: {elapsedFmt}", end="\r")

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

        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels, vectorTimesBatch) in enumerate(valLoader):
                prevImg = prevImg.to(device)
                currentImg = currentImg.to(device)
                labels = labels.to(device)
                vectorTimesBatch = vectorTimesBatch.to(device)
                batchSize = labels.size(0)
                batchIndices = valSubsetIndices[valCursor:valCursor + batchSize]
                valCursor += batchSize

                perStepWeights = perStepWeightsBase
                if vectorTimesBatch.numel() and torch.any(vectorTimesBatch > 0):
                    meanTimes = vectorTimesBatch.mean(dim=0)
                    maxT = float(meanTimes.max()) if meanTimes.numel() else 1.0
                    perStepWeights = torch.exp(-meanTimes / max(maxT, 1e-6)) * 1.5 + 0.3
                with autocast(device_type='cuda', enabled=useAmp):
                    preds, _, _ = model(currentImg, prevImg, teacherForcing=False, tfRatio=0.0, vectorTimes=vectorTimesBatch)
                    
                    # Check for NaN in validation predictions
                    if torch.isnan(preds).any():
                        print(f"NaN detected in validation predictions at epoch {epoch+1}, batch {i}")
                        continue
                    
                    lossMain = trajectoryLossWithWeights(preds, labels, perStepWeights, reduction="mean")
                valLoss += lossMain.item() * labels.size(0)
                ade, fde = adeFde(preds, labels)
                valADE += ade * labels.size(0)
                valFDE += fde * labels.size(0)

                diff = preds - labels
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
                    f"Avg Val Loss: {valLoss/batchesDone:.4f} (best: {bestValLoss:.4f})"
                    f" - ETA: {valEtaTime}", end="\r")

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
                'epoch': epoch,
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
                'epoch': epoch,
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

        print("\n")

    plotHistory(history, os.path.join(runDir, "loss_plot.png"))
    print("Training completed.")


if __name__ == "__main__":
    """
    PLEASE MAKE SURE TO USE A CORRECT DATASET (like image size, time window, etc.)
    """
    datasetPath = r"F:\Projects\Autopilot\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360(1)"
    datasetMaxSize = None           # Maximum number of samples to load from the dataset (None = use all available)
    numEpochs = 2_000               # ~1 full pass at 10k samples/epoch for ~9.6M samples
    patience = 100                   # Early stopping patience (stop if no val improvement for this many epochs)
    batchSize = 12                  # Number of samples per training batch (controls GPU memory usage)
    gradAccumSteps = 1              # Gradient accumulation steps (simulates larger effective batch if >1)
    trainValSplit = 0.8             # Train/validation split ratio
    trainSamplesPerEpoch = 5_000    # Random samples per epoch for fast iterations
    valSamplesPerEpoch = trainSamplesPerEpoch * 2                          # Fixed val samples per epoch (2x trainSamplesPerEpoch)
    seed = 42                       # Base seed for reproducibility
    splitSeed = 42                  # Train/val split seed (keep fixed to avoid contamination)
    learningRate = 5e-5             # Optimized for transformer stability
    featDim = 384                   # Feature dimension in the model
    hiddenDim = 768                 # Hidden dimension in the model
    baseChannels = 48               # Backbone width (increases compute per frame)
    numHeads = 8                    # Transformer attention heads
    numLayers = 4                   # Transformer depth
    useAuxDyn = False               # Whether to enable auxiliary dynamics head (speed/accel prediction)
    resumeModelPath = r"D:\VS_Python_Project\Autopilot\Autopilot\training\run21\last_model.pth"          #"D:/VS_Python_Project/Autopilot/Autopilot/training/run18/best_model.pth"    # Set to path like "training/run13/best_model.pth" to resume training

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