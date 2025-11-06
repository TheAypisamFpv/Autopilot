"""
Training script for trajectory prediction using TrajectoryModel.
Handles dataset loading, model training with AMP and gradient accumulation,
and evaluation with ADE/FDE metrics.
"""

import os
import time
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from CreateModel import TrajectoryModel
from progressBar import getProgressBar


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
            print(f"Filtered to {len(self.samples)} / {len(allSamples)} samples based on balanced.json")
        else:
            self.samples = allSamples
            print(f"No balanced.json found - using all {len(self.samples)} samples")

        self.maxSize = maxSize

    def __len__(self):
        if self.maxSize is not None:
            return min(len(self.samples), self.maxSize)
        return len(self.samples)

    def __getitem__(self, idx):
        sampleId = self.samples[idx]
        prevImgPath = os.path.join(self.imagesDir, f"{sampleId}_prev.png")
        currentImgPath = os.path.join(self.imagesDir, f"{sampleId}_current.png")

        prevImage = Image.open(prevImgPath).convert("RGB")
        currentImage = Image.open(currentImgPath).convert("RGB")

        if self.transform:
            prevImage = self.transform(prevImage)
            currentImage = self.transform(currentImage)

        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        with open(labelPath, "r") as f:
            lines = f.readlines()

        vectorsLine = lines[0].strip().split(" : ")[1]
        if vectorsLine == "None":
            vectors = np.zeros((self.predSteps, 2), dtype=np.float32)
        else:
            vectorsList = [list(map(float, v.split(","))) for v in vectorsLine.split(" ")]
            while len(vectorsList) < self.predSteps:
                vectorsList.append([0.0, 0.0])
            vectors = np.array(vectorsList[:self.predSteps], dtype=np.float32)

        speed = float(lines[1].strip().split(" : ")[1])
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
        useAuxDyn (bool): Whether to use auxiliary dynamics head.
        featDim (int): Feature dimension for model.
        hiddenDim (int): Hidden dimension for model.
        predSteps (int): Number of prediction steps.
        intervalSeconds (float): Time interval between prediction steps.
        deviceOverride (str or None): Device to use ('cpu' or 'cuda'), or None for auto-detect.
        resumeModelPath (str or None): Path to a previously trained model to resume training from.
    """
    print()
    
    if not os.path.exists(datasetDir):
        raise FileNotFoundError(f"Dataset directory '{datasetDir}' does not exist.")

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

        with open(historyPath, "r") as f:
            historyDf = pd.read_csv(f)

        history = {
            "train_loss": historyDf["train_loss"].tolist(),
            "val_loss": historyDf["val_loss"].tolist(),
            "train_ADE": historyDf["train_ADE"].tolist(),
            "val_ADE": historyDf["val_ADE"].tolist(),
            "train_FDE": historyDf["train_FDE"].tolist(),
            "val_FDE": historyDf["val_FDE"].tolist(),
        }

        # calculate start epoch
        startEpoch = len(history["train_loss"])
        
        deviceOverride = loadedParams.get("deviceOverride", deviceOverride)
    else:
        runDir = getRunDir()
        startEpoch = 0
        history = {"train_loss": [], "val_loss": [], "train_ADE": [], "val_ADE": [], "train_FDE": [], "val_FDE": []}

    device = (
        torch.device(deviceOverride)
        if deviceOverride
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    useAmp = device.type == "cuda"
    scaler = GradScaler(device='cuda', enabled=useAmp)

    model = TrajectoryModel(
        featDim=featDim, hiddenDim=hiddenDim, predSteps=predSteps, useAuxDyn=useAuxDyn, intervalSeconds=intervalSeconds
    ).to(device)

    if resumeModelPath:
        model.load_state_dict(torch.load(resumeModelPath))

    # Use model's output spec to determine prediction steps if available
    if hasattr(model, 'outputSpec') and 'num_vectors' in model.outputSpec:
        predSteps = model.outputSpec['num_vectors']

    try:
        modelName = model.name
    except AttributeError:
        modelName = "I Guess We'll Never Know"

    # Use model's output spec to determine prediction steps if available
    if hasattr(model, 'outputSpec') and 'num_vectors' in model.outputSpec:
        predSteps = model.outputSpec['num_vectors']
    
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
        "useAuxDyn": useAuxDyn,
        "featDim": featDim,
        "hiddenDim": hiddenDim,
        "predSteps": predSteps,
        "intervalSeconds": intervalSeconds,
        "deviceOverride": deviceOverride,
        "modelName": modelName,
    }
    if resumeModelPath:
        loadedParams["modelName"] = modelName
        params = loadedParams
    
    with open(os.path.join(runDir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Determine input image size from model spec or default
    inputImageSize = (360, 640)  # Default fallback
    if hasattr(model, 'inputSpec') and 'image_size' in model.inputSpec:
        inputImageSize = model.inputSpec['image_size']

    transform = transforms.Compose([
        transforms.Resize(inputImageSize),  # Use model's expected input image size or default
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
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    numWorkers = 2
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=1e-4, eps=1e-8)

    totalSteps = math.ceil((len(trainLoader) * numEpochs) / max(1, gradAccumSteps))
    warmupSteps = min(500, max(50, int(0.01 * totalSteps)))

    def lrLambda(step):
        if step < warmupSteps:
            return step / max(1, warmupSteps)
        progress = (step - warmupSteps) / max(1, totalSteps - warmupSteps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

    # Generate per-step weights based on number of prediction steps
    perStepWeights = torch.exp(torch.linspace(math.log(1.6), math.log(0.025), predSteps, dtype=torch.float32))
    auxDynWeight = 0.02
    teacherForcingStart = 0.9
    teacherForcingEnd = 0.0
    tfDecayEpochs = min(30, max(5, int(0.2 * numEpochs)))

    bestValLoss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    epochsNoImprove = 0
    trainingStartTime = time.time()
    globalStep = 0

    for epoch in range(startEpoch, numEpochs):
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
                    # Skip this batch or break
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Reduced from 5.0
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
                  f"Avg Train Loss: {avgLoss:.4f} - ETA: {etaTime} - Time: {elapsedFmt}", end="\r")

        trainLoss = runningLoss / len(trainLoader.dataset)
        trainADE = runningADE / len(trainLoader.dataset)
        trainFDE = runningFDE / len(trainLoader.dataset)
        history["train_loss"].append(trainLoss)
        history["train_ADE"].append(trainADE)
        history["train_FDE"].append(trainFDE)

        model.eval()
        valLoss, valADE, valFDE = 0.0, 0.0, 0.0

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
                    f"Avg Val Loss: {valLoss/((i+1)*labels.size(0)):.4f} (best: {bestValLoss:.4f})"
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
    PLEASE MAKE SURE TO USE THE CORRECT DATASET (like image size, time window, etc.)
    """
    datasetPath = r"D:\VS_Python_Project\Autopilot\Autopilot\dataset\output_12_3.0_0.1_framesize640x360_balanced"
    # Parse dataset path to extract parameters
    basename = os.path.basename(datasetPath)
    if basename.startswith('output_'):
        parts = basename.split('_')
        if len(parts) >= 3:
            numVectors = int(parts[1])
            vectorTimeWindow = float(parts[2])
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

    datasetMaxSize = None     # Maximum number of samples to load from the dataset (None = use all available)
    numEpochs = 150           # Total number of training epochs (full passes through the dataset)
    patience = 15             # Early stopping patience (stop if no val improvement for this many epochs)
    batchSize = 12            # Number of samples per training batch (controls GPU memory usage)
    gradAccumSteps = 1        # Gradient accumulation steps (simulates larger effective batch if >1)
    learningRate = 1e-4       # Reduced from 3e-4 to prevent instability
    featDim = 256             # Feature dimension of encoder output (controls model width / capacity)
    hiddenDim = 256           # Hidden size of the GRU decoder (affects model memory and temporal capacity)
    useAuxDyn = False         # Whether to enable auxiliary dynamics head (speed/accel prediction)
    resumeModelPath = None    # Set to path like "training/run13/best_model.pth" to resume training

    trainModel(
        datasetDir=datasetPath,
        numEpochs=numEpochs,
        batchSize=batchSize,
        learningRate=learningRate,
        trainValSplit=0.8,
        patience=patience,
        datasetMaxSize=datasetMaxSize,
        gradAccumSteps=gradAccumSteps,
        useAuxDyn=useAuxDyn,
        featDim=featDim,
        hiddenDim=hiddenDim,
        predSteps=predSteps,
        intervalSeconds=intervalSeconds,
        resumeModelPath=resumeModelPath,
    )