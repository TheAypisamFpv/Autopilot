import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.amp import GradScaler
import os
import json
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from createModel import TrajectoryPredictionModel
from progressBar import getProgressBar

class DrivingDataset(Dataset):
    """
    Custom Dataset for loading the driving data.
    """

    def __init__(self, datasetDir, transform=None):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, 'labels')
        self.imagesDir = os.path.join(datasetDir, 'images')
        self.transform = transform
        self.samples = [f.split('.')[0] for f in os.listdir(self.labelsDir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sampleId = self.samples[idx]
        
        # Load images
        prevImgPath = os.path.join(self.imagesDir, f"{sampleId}_prev.png")
        currentImgPath = os.path.join(self.imagesDir, f"{sampleId}_current.png")
        
        prevImage = Image.open(prevImgPath).convert('RGB')
        currentImage = Image.open(currentImgPath).convert('RGB')

        if self.transform:
            prevImage = self.transform(prevImage)
            currentImage = self.transform(currentImage)

        # Load labels
        labelPath = os.path.join(self.labelsDir, f"{sampleId}.txt")
        with open(labelPath, 'r') as f:
            lines = f.readlines()
            
        # Parse vectors
        vectorsLine = lines[0].strip().split(' : ')[1]
        if vectorsLine == "None":
            vectors = np.zeros((12, 2), dtype=np.float32)
        else:
            vectorsList = [list(map(float, v.split(','))) for v in vectorsLine.split(' ')]
            # Pad if necessary
            while len(vectorsList) < 12:
                vectorsList.append([0.0, 0.0])
            vectors = np.array(vectorsList[:12], dtype=np.float32)

        # Parse other data
        speed = float(lines[1].strip().split(' : ')[1])
        acceleration = float(lines[2].strip().split(' : ')[1])
        turnRate = float(lines[3].strip().split(' : ')[1])
        
        dynamicData = torch.tensor([speed, acceleration, turnRate], dtype=torch.float32)
        
        return prevImage, currentImage, dynamicData, torch.from_numpy(vectors)

def getRunDir(baseDir="training"):
    """
    Gets the next run directory.

    Args:
        baseDir: The base directory for runs.

    Returns:
        str: The path to the new run directory.
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
    """Plots and saves the training history."""
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(savePath)
    plt.close()

def trainModel(datasetDir, numEpochs=25, batchSize=24, learningRate=0.001, trainValSplit=0.7, patience=5):
    """
    Main training function.

    Args:
        datasetDir: The directory containing the dataset.
        numEpochs: The number of training epochs.
        batchSize: The batch size for training.
        learningRate: The learning rate for the optimizer.
        trainValSplit: The train/validation split ratio.
        patience: Number of epochs to wait for improvement before stopping.
    """

    # check if dataset directory exists
    if not os.path.exists(datasetDir):
        raise FileNotFoundError(f"Dataset directory '{datasetDir}' does not exist.")
    
    runDir = getRunDir()
    print(f"Starting new run in: {runDir}")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]) # publicly available pre-calculated normalization values for ImageNet

    print("Creating dataset and dataloaders...", end='')
    # Dataset and Dataloaders
    dataset = DrivingDataset(datasetDir=datasetDir, transform=transform)
    print("Done.")
    print(f"Total samples in dataset: {len(dataset)}\n")

    print("Splitting dataset into training and validation sets...", end='')
    # Split dataset into training and validation
    trainSize = int(trainValSplit * len(dataset))
    valSize = len(dataset) - trainSize
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    print("Done.\n")

    print("Creating model, loss function, and optimizer...", end='')
    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPredictionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    print("Done.\n")

    # saving the training parameters in a .json file
    params = {
        "datasetDir": datasetDir,
        "datasetSize": len(dataset),
        "runDir": runDir,
        "device": str(device),
        "numEpochs": numEpochs,
        "batchSize": batchSize,
        "learningRate": learningRate,
        "trainValSplit": trainValSplit,
        "patience": patience
    }
    with open(os.path.join(runDir, 'training_params.json'), 'w') as paramsFile:
        json.dump(params, paramsFile, indent=4)

    print(f"Training parameters:\n{json.dumps(params, indent=4)}\n")


    bestValLoss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    epochsNoImprove = 0

    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        print(f"\nEpoch {epoch+1}/{numEpochs} - Training")
        startTime = time.time()
        for i, (prevImg, currentImg, dynamicData, labels) in enumerate(trainLoader):
            completion = (i + 1) / len(trainLoader)

            prevImg, currentImg, dynamicData, labels = prevImg.to(device), currentImg.to(device), dynamicData.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(prevImg, currentImg, dynamicData)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            runningLoss += loss.item()
            avgTrainLoss = runningLoss / (i + 1)
            
            # ETA calculation
            elapsedTime = time.time() - startTime
            batchesDone = i + 1
            batchesLeft = len(trainLoader) - batchesDone
            timePerBatch = elapsedTime / batchesDone
            eta = batchesLeft * timePerBatch
            completion_time = time.time() + eta
            etaFormatted = time.strftime('%H:%M:%S', time.localtime(completion_time))

            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=50)} -> Avg Train Loss: {avgTrainLoss:.4f} - ETA: {etaFormatted}    ", end='\r')
        
        trainLoss = runningLoss / len(trainLoader)
        history['train_loss'].append(trainLoss)

        # Validation
        model.eval()
        valLoss = 0.0
        print(f"\nEpoch {epoch+1}/{numEpochs} - Validation")
        startTime = time.time()
        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels) in enumerate(valLoader):
                completion = (i + 1) / len(valLoader)

                prevImg, currentImg, dynamicData, labels = prevImg.to(device), currentImg.to(device), dynamicData.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(prevImg, currentImg, dynamicData)
                    loss = criterion(outputs, labels)
                valLoss += loss.item()
                avgValLoss = valLoss / (i + 1)

                # ETA calculation
                elapsedTime = time.time() - startTime
                batchesDone = i + 1
                batchesLeft = len(valLoader) - batchesDone
                timePerBatch = elapsedTime / batchesDone
                eta = batchesLeft * timePerBatch
                completion_time = time.time() + eta
                etaFormatted = time.strftime('%H:%M:%S', time.localtime(completion_time))

                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=50)} -> Avg Val Loss: {avgValLoss:.4f} (Best: {bestValLoss:.4f}) - ETA: {etaFormatted}    ", end='\r')

        valLoss /= len(valLoader)
        history['val_loss'].append(valLoss)

        print(f"\nEpoch {epoch+1}/{numEpochs}, Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")

        # Save best model
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            torch.save(model.state_dict(), os.path.join(runDir, 'best_model.pth'))
            print("New best model saved.")
            epochsNoImprove = 0
        else:
            epochsNoImprove += 1
        
        if epochsNoImprove >= patience:
            print(f"Early stopping after {patience} epochs with no improvement.")
            break

    # Save history and plot
    historyDf = pd.DataFrame(history)
    historyDf.to_csv(os.path.join(runDir, 'training_history.csv'), index=False)
    plotHistory(history, os.path.join(runDir, 'loss_plot.png'))
    
    print("Training finished.")

if __name__ == '__main__':
    datasetPath = r"D:\VS_Python_Project\Autopilot\Autopilot\dataset\output"

    epoch = 100
    patience = 10
    batchSize = 12

    trainModel(datasetDir=datasetPath, numEpochs=epoch, patience=patience, batchSize=batchSize)
