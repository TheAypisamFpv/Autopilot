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

from CreateModel import TrajectoryModel
from progressBar import getProgressBar

# Enable expandable segments for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class DrivingDataset(Dataset):
    """
    Dataset for trajectory prediction.
    Loads paired images (previous and current) and trajectory vectors.
    Optionally filters samples using 'balanced.json' if present.
    """

    def __init__(self, datasetDir, transform=None, maxSize=None, predSteps=6, dtype=torch.float32):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, "labels")
        self.imagesDir = os.path.join(datasetDir, "images")
        self.transform = transform
        self.predSteps = predSteps
        self.dtype = dtype
        self.maxSize = maxSize
        self.labelsOnly = False

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
        allSamples = [f.split(".")[0] for f in os.listdir(self.labelsDir) if f.endswith(".txt")]

        # Check for balanced.json
        balancedJsonPath = os.path.join(datasetDir, "balanced.json")
        if os.path.exists(balancedJsonPath):
            print(f"Found balanced.json - using filtered dataset ({balancedJsonPath})")
            with open(balancedJsonPath, 'r') as f:
                balancedData = json.load(f)
            
            keepIndices = set(balancedData.get("keep", []))
            self.samples = [idx for idx in allSamples if idx in keepIndices]
            totalSamples = len(self.samples)
            effectiveSamples = min(totalSamples, self.maxSize) if self.maxSize is not None else totalSamples
            print(f"Filtered to {effectiveSamples} / {totalSamples} samples based on balanced.json")
        else:
            self.samples = allSamples
            totalSamples = len(self.samples)
            effectiveSamples = min(totalSamples, self.maxSize) if self.maxSize is not None else totalSamples
            print(f"No balanced.json found - using {effectiveSamples} / {totalSamples} samples")

    def __len__(self):
        if self.maxSize is not None:
            return min(len(self.samples), self.maxSize)
        return len(self.samples)

    def _parse_label_file(self, labelPath):
        vectors = None
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

        return vectors, speed, videoPath, prevFrameIndex, frameIndex

    def _read_frames_from_video(self, videoPath, prevFrameIndex, frameIndex):
        cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            return None, None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, prevFrameIndex)
            ret_prev, prevFrame = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            ret_curr, currFrame = cap.read()

            if not ret_prev or not ret_curr:
                return None, None

            prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2RGB)
            currFrame = cv2.cvtColor(currFrame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(prevFrame), Image.fromarray(currFrame)
        finally:
            cap.release()

    def __getitem__(self, idx):
        sampleId = self.samples[idx]
        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        vectors, speed, videoPath, prevFrameIndex, frameIndex = self._parse_label_file(labelPath)

        if self.labelsOnly and videoPath is not None and prevFrameIndex is not None and frameIndex is not None:
            prevImage, currentImage = self._read_frames_from_video(videoPath, prevFrameIndex, frameIndex)
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

        dynamicData = torch.tensor([speed], dtype=self.dtype)

        return prevImage, currentImage, dynamicData, torch.from_numpy(vectors).to(self.dtype)


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


def trajectory_loss_with_weights(pred, target, per_step_weights, reduction="mean", delta=1.0):
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
    weights = per_step_weights.to(raw.device).view(1, -1)
    weighted = raw * weights
    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    return weighted


def ADE_FDE(pred, target):
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
    predSteps=12,
    intervalSeconds=0.25,
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
        predSteps (int): Number of prediction steps.
        intervalSeconds (float): Time interval between prediction steps.
        deviceOverride (str or None): Device to use ('cpu' or 'cuda'), or None for auto-detect.
        resumeModelPath (str or None): Path to a previously trained model to resume training from.
    """
    print()

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
        if "predSteps" in loadedParams:
            predSteps = loadedParams["predSteps"]
        if "intervalSeconds" in loadedParams:
            intervalSeconds = loadedParams["intervalSeconds"]
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

    device = (
        torch.device(deviceOverride)
        if deviceOverride
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    useAmp = device.type == "cuda"
    scaler = GradScaler(device='cuda', enabled=useAmp)

    transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Higher resolution
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.15),
        transforms.RandomAffine(degrees=2.5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DrivingDataset(datasetDir, transform=transform, maxSize=datasetMaxSize, predSteps=predSteps)
    totalSamples = len(dataset)
    trainSize = int(trainValSplit * totalSamples)
    valSize = totalSamples - trainSize
    splitGenerator = torch.Generator().manual_seed(int(splitSeed))
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize], generator=splitGenerator)

    numWorkers = 1

    def get_train_subset_size():
        if trainSamplesPerEpoch is None:
            return len(trainDataset)
        
        return max(1, min(int(trainSamplesPerEpoch), len(trainDataset)))

    def get_val_subset_size():
        if valSamplesPerEpoch is None:
            return len(valDataset)

        return max(1, min(int(valSamplesPerEpoch), len(valDataset)))

    def build_train_loader(epoch):
        if trainSamplesPerEpoch is None or trainSamplesPerEpoch >= len(trainDataset):
            return DataLoader(trainDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
        
        subset_size = get_train_subset_size()
        rng = random.Random(seed + epoch)
        subset_indices = rng.sample(range(len(trainDataset)), subset_size)
        subset = Subset(trainDataset, subset_indices)
        return DataLoader(subset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)

    def build_val_loader(epoch):
        if valSamplesPerEpoch is None or valSamplesPerEpoch >= len(valDataset):
            return DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)

        subset_size = get_val_subset_size()
        rng = random.Random(seed + 100000 + epoch)
        subset_indices = rng.sample(range(len(valDataset)), subset_size)
        subset = Subset(valDataset, subset_indices)
        return DataLoader(subset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)

    model = TrajectoryModel(
        featDim=featDim, hiddenDim=hiddenDim, predSteps=predSteps, useAuxDyn=useAuxDyn, intervalSeconds=intervalSeconds
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
        "predSteps": predSteps,
        "intervalSeconds": intervalSeconds,
        "deviceOverride": deviceOverride,
        "modelName": modelName,
    }
    with open(os.path.join(runDir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Transformer-friendly optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)

    trainSubsetSize = get_train_subset_size()
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

    perStepWeights = torch.tensor([1.6, 1.3, 1.0, 0.8, 0.6, 0.5] + [0.4] * (predSteps - 6), dtype=torch.float32)  # Adjust for longer horizon
    auxDynWeight = 0.02
    teacherForcingStart = 0.9
    teacherForcingEnd = 0.0
    tfDecayEpochs = min(30, max(5, int(0.2 * numEpochs)))

    bestValLoss = float("inf")
    epochsNoImprove = 0
    trainingStartTime = time.time()

    if resumeModelPath:
        bestValLoss = min(history["val_loss"])
        epochsNoImprove = len(history["val_loss"]) - history["val_loss"].index(bestValLoss) - 1

    for epoch in range(startEpoch, numEpochs):
        trainLoader = build_train_loader(epoch)
        model.train()
        runningLoss, runningADE, runningFDE = 0.0, 0.0, 0.0
        epochStartTime = time.time()

        tfRatio = (
            teacherForcingStart - (epoch / tfDecayEpochs) * (teacherForcingStart - teacherForcingEnd)
            if epoch < tfDecayEpochs
            else teacherForcingEnd
        )

        print(f"###### Epoch {epoch + 1}/{numEpochs} - TF Ratio: {tfRatio:.2f} ######")

        optimizer.zero_grad()
        accumSteps = 0

        for i, (prevImg, currentImg, dynamicData, labels) in enumerate(trainLoader):
            prevImg, currentImg, labels = prevImg.to(device), currentImg.to(device), labels.to(device)
            batchSize = labels.size(0)

            with autocast(device_type='cuda', enabled=useAmp):
                preds, auxOut, _ = model(currentImg, prevImg, gtTraj=labels, teacherForcing=True, tfRatio=tfRatio)
                
                # Check for NaN in predictions
                if torch.isnan(preds).any():
                    print(f"NaN detected in predictions at epoch {epoch+1}, batch {i}")
                    print(f"Preds: {preds}")
                    print(f"Labels: {labels}")
                    print(f"PrevImg stats: min={prevImg.min()}, max={prevImg.max()}, mean={prevImg.mean()}")
                    print(f"CurrentImg stats: min={currentImg.min()}, max={currentImg.max()}, mean={currentImg.mean()}")
                    continue  # or break to stop training
                
                lossMain = trajectory_loss_with_weights(preds, labels, perStepWeights, reduction="mean")
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
            ade, fde = ADE_FDE(preds.detach(), labels)
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
        valLoader = build_val_loader(epoch)

        print('\n\nValidation Progress:')

        valStartTime = time.time()
        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels) in enumerate(valLoader):
                prevImg, currentImg, labels = prevImg.to(device), currentImg.to(device), labels.to(device)
                with autocast(device_type='cuda', enabled=useAmp):
                    preds, _, _ = model(currentImg, prevImg, teacherForcing=False, tfRatio=0.0)
                    
                    # Check for NaN in validation predictions
                    if torch.isnan(preds).any():
                        print(f"NaN detected in validation predictions at epoch {epoch+1}, batch {i}")
                        continue
                    
                    lossMain = trajectory_loss_with_weights(preds, labels, perStepWeights, reduction="mean")
                valLoss += lossMain.item() * labels.size(0)
                ade, fde = ADE_FDE(preds, labels)
                valADE += ade * labels.size(0)
                valFDE += fde * labels.size(0)
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

        historyDf = pd.DataFrame(history)
        historyDf.to_csv(os.path.join(runDir, "training_history.csv"), index=False)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            torch.save(model.state_dict(), os.path.join(runDir, "best_model.pth"))
            print(f"New best model saved: {bestValLoss:.4f}")
            epochsNoImprove = 0
        else:
            torch.save(model.state_dict(), os.path.join(runDir, "last_model.pth"))
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
    datasetPath = r"F:\Projects\Autopilot\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360"
    # Parse dataset path to extract parameters
    basename = os.path.basename(datasetPath)
    if basename.startswith('output_'):
        parts = basename.split('_')
        if len(parts) >= 3:
            numVectors = int(parts[-4])
            vectorTimeWindow = float(parts[-3])
            intervalSeconds = vectorTimeWindow / numVectors
            # temporalContext = float(parts[3])
            # imageSize = parts[4].removeprefix("framesize").split("x")
            # imageSize = (int(imageSize[1]), int(imageSize[0]))  # (height, width)
            
            predSteps = numVectors
            print(f"Parsed from dataset path: numVectors={numVectors}, vectorTimeWindow={vectorTimeWindow}, intervalSeconds={intervalSeconds}")
        else:
            print("Warning: Could not parse dataset path, using defaults")
            predSteps = 12
            intervalSeconds = 0.25
    else:
        print("Warning: Dataset path does not start with 'output_', using defaults")
        predSteps = 12
        intervalSeconds = 0.25

    datasetMaxSize = None           # Maximum number of samples to load from the dataset (None = use all available)
    numEpochs = 10000               # ~1 full pass at 10k samples/epoch for ~9.6M samples
    patience = 50                   # Early stopping patience (stop if no val improvement for this many epochs)
    batchSize = 16                  # Number of samples per training batch (controls GPU memory usage)
    gradAccumSteps = 1              # Gradient accumulation steps (simulates larger effective batch if >1)
    trainValSplit = 0.8             # Train/validation split ratio
    trainSamplesPerEpoch = 1_000    # Random samples per epoch for fast iterations
    valSamplesPerEpoch = int((1 - trainValSplit)*trainSamplesPerEpoch)      # Random val samples per epoch (20% of trainSamplesPerEpoch)
    seed = 42                       # Base seed for reproducibility
    splitSeed = 42                  # Train/val split seed (keep fixed to avoid contamination)
    learningRate = 5e-5             # Optimized for transformer stability
    featDim = 256                   # Feature dimension in the model
    hiddenDim = 512                 # Hidden dimension in the model
    useAuxDyn = False               # Whether to enable auxiliary dynamics head (speed/accel prediction)
    resumeModelPath = None          #"D:/VS_Python_Project/Autopilot/Autopilot/training/run18/best_model.pth"    # Set to path like "training/run13/best_model.pth" to resume training

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
        predSteps=predSteps,
        intervalSeconds=intervalSeconds,
        resumeModelPath=resumeModelPath,
    )