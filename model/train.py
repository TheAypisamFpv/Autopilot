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


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class DrivingDataset(Dataset):
    """
    Custom Dataset for loading the driving data.
    """

    def __init__(self, datasetDir, transform=None, maxSize=None, dtype=torch.float32):
        self.datasetDir = datasetDir
        self.labelsDir = os.path.join(datasetDir, 'labels')
        self.imagesDir = os.path.join(datasetDir, 'images')
        self.transform = transform
        self.samples = [f.split('.')[0] for f in os.listdir(self.labelsDir) if f.endswith('.txt')]
        self.maxSize = maxSize
        self.dtype = dtype  # Store the precision type for tensor creation


    def __len__(self):
        if self.maxSize is not None:
            return min(len(self.samples), self.maxSize)

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
            vectors = np.zeros((6, 2), dtype=np.float32)
        else:
            vectorsList = [list(map(float, v.split(','))) for v in vectorsLine.split(' ')]
            # Pad if necessary
            while len(vectorsList) < 6:
                vectorsList.append([0.0, 0.0])
            vectors = np.array(vectorsList[:6], dtype=np.float32)

        # Parse only speed data
        speed = float(lines[1].strip().split(' : ')[1])
        
        # Only speed as dynamic data, use the specified precision
        dynamicData = torch.tensor([speed], dtype=self.dtype)
        
        # Convert vectors to tensor with the specified dtype
        return prevImage, currentImage, dynamicData, torch.from_numpy(vectors).to(self.dtype)

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

def trainModel(datasetDir, numEpochs=25, batchSize=24, learningRate=0.001, trainValSplit=0.6, patience=5, datasetMaxSize=None):
    """
    Main training function.

    Args:
        datasetDir: The directory containing the dataset.
        numEpochs: The number of training epochs.
        batchSize: The batch size for training.
        learningRate: The learning rate for the optimizer.
        trainValSplit: The train/validation split ratio.
        patience: Number of epochs to wait for improvement before stopping.
        datasetMaxSize: Maximum size of the dataset to use for training.
    """

    # check if dataset directory exists
    if not os.path.exists(datasetDir):
        raise FileNotFoundError(f"Dataset directory '{datasetDir}' does not exist.")
    
    runDir = getRunDir()
    print(f"Starting new run in: {runDir}")

    # Determine precision type based on CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("CUDA detected: Using mixed precision training with AMP")
        dtype = torch.float16
        useAmp = True
    else:
        print("CPU detected: Using full precision training")
        dtype = torch.float32
        useAmp = False

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) # publicly available pre-calculated normalization values for ImageNet

    print("Creating dataset and dataloaders...", end='')
    # Dataset and Dataloaders with specified precision
    dataset = DrivingDataset(datasetDir=datasetDir, transform=transform, maxSize=datasetMaxSize, dtype=dtype)
    print("Done.")
    print(f"Total samples in dataset: {len(dataset)}, using precision: {dtype}\n")

    print("Splitting dataset into training and validation sets...", end='')
    # Split dataset into training and validation
    trainSize = int(trainValSplit * len(dataset))
    valSize = len(dataset) - trainSize
    trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)
    print("Done.\n")

    print("Creating model, loss function, and optimizer...", end='')
    # Model, Loss, Optimizer - use the previously determined device
    model = TrajectoryPredictionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    scaler = GradScaler(enabled=useAmp)  # Use our determined AMP flag
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
        "patience": patience,
        "precision": "mixed (fp16/fp32)" if useAmp else "fp32",
    }
    with open(os.path.join(runDir, 'training_params.json'), 'w') as paramsFile:
        json.dump(params, paramsFile, indent=4)

    print(f"Training parameters:\n{json.dumps(params, indent=4)}\n")


    bestValLoss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    epochsNoImprove = 0
    trainingStartTime = time.time()

    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        print(f"\nEpoch {epoch+1}/{numEpochs} - Training")
        epochStartTime = time.time()
        for i, (prevImg, currentImg, dynamicData, labels) in enumerate(trainLoader):
            completion = (i + 1) / len(trainLoader)

            prevImg, currentImg, dynamicData, labels = prevImg.to(device), currentImg.to(device), dynamicData.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=useAmp):
                outputs = model(prevImg, currentImg, dynamicData)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            runningLoss += loss.item()
            avgTrainLoss = runningLoss / (i + 1)
            
            # ETA calculation
            epochElapsedTime = time.time() - epochStartTime
            batchesDone = i + 1
            batchesLeft = len(trainLoader) - batchesDone
            timePerBatch = epochElapsedTime / batchesDone
            eta = batchesLeft * timePerBatch
            completion_time = time.time() + eta
            epochEtaFormatted = time.strftime('%H:%M:%S', time.localtime(completion_time))

            trainingElapsedTime = time.time() - trainingStartTime
            hours, remainder = divmod(int(trainingElapsedTime), 3600)
            minutes, seconds = divmod(remainder, 60)
            trainingElapsedTimeFormatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)} -> Avg Train Loss: {avgTrainLoss:.4f} - epoch train ETA: {epochEtaFormatted} - Training time: {trainingElapsedTimeFormatted}    ", end='\r')

        trainLoss = runningLoss / len(trainLoader)
        history['train_loss'].append(trainLoss)

        # Validation
        model.eval()
        valLoss = 0.0
        print(f"\nEpoch {epoch+1}/{numEpochs} - Validation")
        epochStartTime = time.time()
        with torch.no_grad():
            for i, (prevImg, currentImg, dynamicData, labels) in enumerate(valLoader):
                completion = (i + 1) / len(valLoader)

                prevImg, currentImg, dynamicData, labels = prevImg.to(device), currentImg.to(device), dynamicData.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=useAmp):
                    outputs = model(prevImg, currentImg, dynamicData)
                    loss = criterion(outputs, labels)
                valLoss += loss.item()
                avgValLoss = valLoss / (i + 1)

                # ETA calculation
                epochElapsedTime = time.time() - epochStartTime
                batchesDone = i + 1
                batchesLeft = len(valLoader) - batchesDone
                timePerBatch = epochElapsedTime / batchesDone
                eta = batchesLeft * timePerBatch
                completion_time = time.time() + eta
                epochEtaFormatted = time.strftime('%H:%M:%S', time.localtime(completion_time))

                trainingElapsedTime = time.time() - trainingStartTime
                hours, remainder = divmod(int(trainingElapsedTime), 3600)
                minutes, seconds = divmod(remainder, 60)
                trainingElapsedTimeFormatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                print(f"{getProgressBar(completion, wheelIndex=i, maxbarLength=75)} -> Avg Val Loss: {avgValLoss:.4f} (Best: {bestValLoss:.4f}) - epoch val ETA: {epochEtaFormatted} - Training time: {trainingElapsedTimeFormatted}    ", end='\r')

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
            torch.save(model.state_dict(), os.path.join(runDir, 'last_model.pth'))
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
    datasetPath = r"F:\VS_Python_Project\Autopilot\Autopilot\dataset\output"
    datasetMaxSize = 75_000

    
    epoch = 150
    patience = 15
    batchSize = 24

    trainModel(datasetDir=datasetPath, numEpochs=epoch, patience=patience, batchSize=batchSize, datasetMaxSize=datasetMaxSize)
