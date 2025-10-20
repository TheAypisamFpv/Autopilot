"""
Training script for trajectory prediction using TrajectoryModel.
Handles dataset loading, model training with AMP and gradient accumulation,
and evaluation with ADE/FDE metrics.
"""

import os
import time
import math
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
    """

    def __init__(self, datasetDir, transform=None, maxSize=None, dtype=torch.float32):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, "labels")
        self.imagesDir = os.path.join(datasetDir, "images")
        self.transform = transform
        self.samples = [f.split(".")[0] for f in os.listdir(self.labelsDir) if f.endswith(".txt")]
        self.maxSize = maxSize
        self.dtype = dtype

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
            vectors = np.zeros((12, 2), dtype=np.float32)
        else:
            vectorsList = [list(map(float, v.split(","))) for v in vectorsLine.split(" ")]
            while len(vectorsList) < 6:
                vectorsList.append([0.0, 0.0])
            vectors = np.array(vectorsList[:12], dtype=np.float32)

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
    grad_accum_steps=2,
    use_aux_dyn=False,
    feat_dim=256,
    hidden_dim=256,
    pred_steps=6,
    device_override=None,
):
    """
    Main training function for TrajectoryModel.

    Handles data loading, training, validation, early stopping, and saving checkpoints.
    """
    if not os.path.exists(datasetDir):
        raise FileNotFoundError(f"Dataset directory '{datasetDir}' does not exist.")

    runDir = getRunDir()
    device = (
        torch.device(device_override)
        if device_override
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    useAmp = device.type == "cuda"
    scaler = GradScaler(device='cuda', enabled=useAmp)

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.15),
        transforms.RandomAffine(degrees=2.5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DrivingDataset(datasetDir, transform=transform, maxSize=datasetMaxSize)
    total_samples = len(dataset)
    trainSize = int(trainValSplit * total_samples)
    valSize = total_samples - trainSize
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    num_workers = 2
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=num_workers, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = TrajectoryModel(
        feat_dim=feat_dim, hidden_dim=hidden_dim, pred_steps=pred_steps, use_aux_dyn=use_aux_dyn
    ).to(device)


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
        "grad_accum_steps": grad_accum_steps,
        "use_aux_dyn": use_aux_dyn,
        "feat_dim": feat_dim,
        "hidden_dim": hidden_dim,
        "pred_steps": pred_steps,
        "device_override": device_override,
        "model_name": modelName,
    }
    with open(os.path.join(runDir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=1e-4, eps=1e-8)

    total_steps = math.ceil((len(trainLoader) * numEpochs) / max(1, grad_accum_steps))
    warmup_steps = min(500, max(50, int(0.01 * total_steps)))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    per_step_weights = torch.tensor([1.6, 1.3, 1.0, 0.8, 0.6, 0.5], dtype=torch.float32)
    aux_dyn_weight = 0.02
    teacher_forcing_start = 0.9
    teacher_forcing_end = 0.0
    tf_decay_epochs = min(30, max(5, int(0.2 * numEpochs)))

    history = {"train_loss": [], "val_loss": [], "train_ADE": [], "val_ADE": [], "train_FDE": [], "val_FDE": []}
    bestValLoss = float("inf")
    epochsNoImprove = 0
    trainingStartTime = time.time()
    global_step = 0

    for epoch in range(numEpochs):
        model.train()
        runningLoss, runningADE, runningFDE = 0.0, 0.0, 0.0
        epochStartTime = time.time()

        tf_ratio = (
            teacher_forcing_start - (epoch / tf_decay_epochs) * (teacher_forcing_start - teacher_forcing_end)
            if epoch < tf_decay_epochs
            else teacher_forcing_end
        )

        optimizer.zero_grad()
        accum_steps = 0

        for i, (prevImg, currentImg, dynamicData, labels) in enumerate(trainLoader):
            prevImg, currentImg, labels = prevImg.to(device), currentImg.to(device), labels.to(device)
            batch_size = labels.size(0)

            with autocast(device='cuda', enabled=useAmp):
                preds, aux_out, _ = model(currentImg, prevImg, gt_traj=labels, teacher_forcing=True, tf_ratio=tf_ratio)
                loss_main = trajectory_loss_with_weights(preds, labels, per_step_weights, reduction="mean")
                loss = loss_main

                if use_aux_dyn and aux_out is not None:
                    aux_loss = torch.tensor(0.0, device=device)
                    loss += aux_dyn_weight * aux_loss

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps == grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accum_steps = 0

            runningLoss += loss_main.item() * batch_size
            ade, fde = ADE_FDE(preds.detach(), labels)
            runningADE += ade * batch_size
            runningFDE += fde * batch_size

            completion = (i + 1) / len(trainLoader)
            epochElapsed = time.time() - epochStartTime
            batchesDone = i + 1
            timePerBatch = epochElapsed / max(1, batchesDone)
            eta = (len(trainLoader) - batchesDone) * timePerBatch
            etaTime = time.strftime("%H:%M:%S", time.localtime(time.time() + eta))
            totalElapsed = time.time() - trainingStartTime
            h, rem = divmod(int(totalElapsed), 3600)
            m, s = divmod(rem, 60)
            elapsedFmt = f"{h:02d}:{m:02d}:{s:02d}"

            avgLoss = runningLoss / ((i + 1) * batch_size)
            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)} -> "
                  f"Avg Train Loss: {avgLoss:.4f} - ETA: {etaTime} - Time: {elapsedFmt}", end="\r")

        trainLoss = runningLoss / len(trainLoader.dataset)
        trainADE = runningADE / len(trainLoader.dataset)
        trainFDE = runningFDE / len(trainLoader.dataset)
        history["train_loss"].append(trainLoss)
        history["train_ADE"].append(trainADE)
        history["train_FDE"].append(trainFDE)

        model.eval()
        valLoss, valADE, valFDE = 0.0, 0.0, 0.0

        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels) in enumerate(valLoader):
                prevImg, currentImg, labels = prevImg.to(device), currentImg.to(device), labels.to(device)
                with autocast(device='cuda', enabled=useAmp):
                    preds, _, _ = model(currentImg, prevImg, teacher_forcing=False, tf_ratio=0.0)
                    loss_main = trajectory_loss_with_weights(preds, labels, per_step_weights, reduction="mean")
                valLoss += loss_main.item() * labels.size(0)
                ade, fde = ADE_FDE(preds, labels)
                valADE += ade * labels.size(0)
                valFDE += fde * labels.size(0)
                completion = (i + 1) / len(valLoader)
                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)} -> "
                      f"Avg Val Loss: {valLoss/((i+1)*labels.size(0)):.4f} (best: {bestValLoss:.4f})", end="\r")

        valLoss /= len(valLoader.dataset)
        valADE /= len(valLoader.dataset)
        valFDE /= len(valLoader.dataset)
        history["val_loss"].append(valLoss)
        history["val_ADE"].append(valADE)
        history["val_FDE"].append(valFDE)

        print(f"\nEpoch {epoch+1}/{numEpochs} | Train Loss: {trainLoss:.4f} | "
              f"Val Loss: {valLoss:.4f} | ADE: {valADE:.4f} | FDE: {valFDE:.4f}")

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            torch.save(model.state_dict(), os.path.join(runDir, "best_model.pth"))
            epochsNoImprove = 0
        else:
            torch.save(model.state_dict(), os.path.join(runDir, "last_model.pth"))
            epochsNoImprove += 1

        if epochsNoImprove >= patience:
            print(f"Early stopping after {patience} epochs with no improvement.")
            break

    historyDf = pd.DataFrame(history)
    historyDf.to_csv(os.path.join(runDir, "training_history.csv"), index=False)
    plotHistory(history, os.path.join(runDir, "loss_plot.png"))
    print("Training completed.")


if __name__ == "__main__":
    datasetPath = r"D:\VS_Python_Project\Autopilot\Autopilot\dataset\output"
    datasetMaxSize = 75_000   # Maximum number of samples to load from the dataset (None = use all available)
    num_epochs = 150          # Total number of training epochs (full passes through the dataset)
    patience = 15             # Early stopping patience (stop if no val improvement for this many epochs)
    batchSize = 48            # Number of samples per training batch (controls GPU memory usage)
    grad_accum_steps = 1      # Gradient accumulation steps (simulates larger effective batch if >1)
    learningRate = 3e-4       # Initial learning rate for the optimizer
    feat_dim = 256            # Feature dimension of encoder output (controls model width / capacity)
    hidden_dim = 256          # Hidden size of the GRU decoder (affects model memory and temporal capacity)
    pred_steps = 6            # Number of waypoints (time steps) predicted for each sample
    use_aux_dyn = False       # Whether to enable auxiliary dynamics head (speed/accel prediction)


    trainModel(
        datasetDir=datasetPath,
        numEpochs=num_epochs,
        batchSize=batchSize,
        learningRate=learningRate,
        trainValSplit=0.8,
        patience=patience,
        datasetMaxSize=datasetMaxSize,
        grad_accum_steps=grad_accum_steps,
        use_aux_dyn=use_aux_dyn,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        pred_steps=pred_steps,
    )
