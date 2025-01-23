import gc
import json
import os
import random
import time
import pandas as pd
import numpy as np
import tracemalloc
from math import e, floor
import matplotlib.pyplot as plt
import psutil
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing
from multiprocessing import Pool, Manager, Lock
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split

from progressBar import getProgressBar

# Set a random seed for reproducibility
RANDOM_SEED = np.random.randint(0, 1000)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# set torch to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


averageWorkerUtilization = 17 #% average worker utilization

class NeuralNet(nn.Module):
    def __init__(self, layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dropoutRates = dropoutRates
        self.l2Reg = l2Reg

        # Add input layer
        self.layers.append(nn.Linear(layers[0], layers[1]))
        self.layers.append(self.getActivation(inputActivation))
        self.layers.append(nn.BatchNorm1d(layers[1]))
        self.layers.append(nn.Dropout(dropoutRates[0]))

        # Add hidden layers
        for i in range(1, len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.getActivation(hiddenActivation))
            self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            dropoutRateIndex = min(i, len(dropoutRates) - 1)
            self.layers.append(nn.Dropout(dropoutRates[dropoutRateIndex]))

        # Add output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(self.getActivation(outputActivation))

    def getActivation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        elif activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                if x.size(0) == 1:
                    continue  # Skip batch normalization if batch size is 1
            x = layer(x)
        return x

    def fit(self, trainLoader, testLoader, optimizer, lossFn, epochs, device):
        bestValAccuracy = 0
        patience = 10
        patienceCounter = 0

        for epoch in range(epochs):
            self.train()
            for XBatch, yBatch in trainLoader:
                # Move data to the GPU
                XBatch, yBatch = XBatch.to(device), yBatch.to(device)
                
                optimizer.zero_grad()
                outputs = self(XBatch)
                loss = lossFn(outputs, yBatch)
                loss.backward()
                optimizer.step()


            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for XBatch, yBatch in testLoader:
                    # Move data to the GPU
                    XBatch, yBatch = XBatch.to(device), yBatch.to(device)
                    
                    outputs = self(XBatch)
                    predicted = outputs
                    # Calculate correct predictions
                    error = torch.abs(predicted - yBatch)
                    # correct function : f(x) = -log10(x)
                    correct += torch.sum(torch.where(error <= 0.1, torch.tensor(1.0), -torch.log10(error)))
                    total += yBatch.numel()  # Total number of elements

            valAccuracy = correct / total
            if valAccuracy > bestValAccuracy:
                bestValAccuracy = float(valAccuracy)
                patienceCounter = 0
            else:
                patienceCounter += 1

            if patienceCounter >= patience:
                break

        return bestValAccuracy


def loadData(filepath):
    print("Loading CSV data...", end=" ")
    data = pd.read_csv(filepath)
    print("done.")
    return data

def preprocessData(data):
    print("Preprocessing data...", end=" ")
    firstColumnName = str(data.columns[0])
    secondColumnName = str(data.columns[1])
    
    # Convert string representation of lists to numpy arrays
    data[secondColumnName] = data[secondColumnName].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    data[firstColumnName] = data[firstColumnName].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    
    resizedFrameData = np.stack(data[secondColumnName].values)
    originalFrameData = np.stack(data[firstColumnName].values)
    
    print("done.")
    return resizedFrameData, originalFrameData

def createModel(layers, dropoutRates, l2Reg, learningRate, inputActivation, hiddenActivation, outputActivation, loss, optimizer, metrics):
    model = NeuralNet(layers, dropoutRates, l2Reg, inputActivation, hiddenActivation, outputActivation)

    model.to(device)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=l2Reg)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learningRate, weight_decay=l2Reg)
    # Add other optimizers as needed

    if loss == 'binary_crossentropy':
        loss = nn.BCELoss()
    elif loss == 'mse':
        loss = nn.MSELoss()
    elif loss == 'l1':
        loss = nn.L1Loss()
    # Add other loss functions as needed

    return model, optimizer, loss, device

def evaluateIndividual(params, XTrain, yTrain, XTest, yTest, populationSize, index):
    time.sleep(index * 0.05)  # wait a bit for a better printout

    model, optimizer, lossFn, device = createModel(
        layers=params['layers'],
        dropoutRates=params['dropoutRates'],
        l2Reg=params['l2Reg'],
        learningRate=params['learningRate'],
        inputActivation=params['inputActivation'],
        hiddenActivation=params['hiddenActivation'],
        outputActivation=params['outputActivation'],
        loss=params['loss'],
        optimizer=params['optimizer'],
        metrics=params['metrics']
    )

    trainDataset = TensorDataset(torch.tensor(XTrain, dtype=torch.float32), torch.tensor(yTrain, dtype=torch.float32))
    testDataset = TensorDataset(torch.tensor(XTest, dtype=torch.float32), torch.tensor(yTest, dtype=torch.float32))

    trainLoader = DataLoader(trainDataset, batch_size=params['batch_size'], shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=params['batch_size'], shuffle=True) # fuck it, true

    

    bestValAccuracy = model.fit(
        trainLoader,
        testLoader,
        optimizer,
        lossFn,
        params['epochs'],
        device
        )

    params['fitness'] = bestValAccuracy

    del model
    del trainLoader
    del testLoader
    torch.cuda.empty_cache()
    gc.collect()

    return params

def worker(paramsIndex: tuple):
    params, index, XTrain, yTrain, XTest, yTest, populationSize, cpuPowerLimit, workerQueue, queueLock = paramsIndex

    time.sleep(index * 0.01)  # wait a bit for a better printout

    while True:
        with queueLock:
            currentCpuLoad = psutil.cpu_percent(interval=0.5)
            workersAhead = len(workerQueue)
            estimatedBaseCpuLoad = max(10, currentCpuLoad - averageWorkerUtilization * workersAhead)
            estimatedNewCpuLoad = estimatedBaseCpuLoad + (averageWorkerUtilization * (workersAhead + 1)) - 2 #2% margin of error

            if estimatedNewCpuLoad <= cpuPowerLimit * 100:
                workerQueue.append(index)  # Add to active workers
                break

        print(f"{workerQueue} | Worker {index} waiting for CPU..." + " " * 5, end="\r")
        time.sleep(0.1)  # Wait before rechecking

    try:
        print(f"{workerQueue} |" + " " * 30, end="\r")
        return evaluateIndividual(params, XTrain, yTrain, XTest, yTest, populationSize, index)
    finally:
        with queueLock:
            workerQueue.remove(index)  # Remove from active workers after completion

def evaluatePopulation(population: list[dict], XTrain, yTrain, XTest, yTest, cpuPowerLimit: float, nJobs: int):
    print("Evaluating population..." + " " * 5, end="\r")

    manager = Manager()
    workerQueue = manager.list()
    queueLock = manager.Lock()

    populationSize = len(population)

    poolArgs = [
        (params, idx+1, XTrain, yTrain, XTest, yTest, populationSize, cpuPowerLimit, workerQueue, queueLock)
        for idx, params in enumerate(population)
    ]

    if nJobs == 1:
        evaluatedPopulation = [worker(args) for args in poolArgs]
    else:
        with Pool(processes=nJobs, maxtasksperchild=1) as pool:
            evaluatedPopulation = pool.map(worker, poolArgs)

    gc.collect()

    print("Population evaluated. " + " " * 5, end="\r")

    return evaluatedPopulation

def crossover(parent1: dict, parent2: dict):
    child = {}
    for key in parent1.keys():
        if np.random.rand() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(child: dict, mutationRate: float, mutationForce: float):
    """
    Mutate the child parameters with a given mutation rate.
    """

    if np.random.rand() > mutationRate:
        return child

    mutateLow, mutateHigh = 1 - mutationForce, 1 + mutationForce
    
    optimizerChoices = ['adam', 'sgd']
    activationChoices = ['relu', 'sigmoid', 'tanh', 'softmax', 'leakyrelu', 'prelu', 'elu']

    noMutationKeys = ['fitness', 'metrics', 'loss']


    def getMutatedValue(value):
        if isinstance(value, int):
            mutation = np.random.normal(mutateLow, mutateHigh)
            return int(abs(mutation * value))
        elif isinstance(value, float):
            mutation = np.random.normal(mutateLow, mutateHigh)
            return round(abs(mutation * value), 6)
        
        # return the original value if not int or float
        return value

    goodKey = False

    while not goodKey:
        randomKey = np.random.choice(list(child.keys()))
        if randomKey.lower() not in noMutationKeys:
            goodKey = True

    if randomKey.lower() == 'layers':
        addLayer = np.random.rand() < mutationRate 
        removeLayer = np.random.rand() < mutationRate / 1.5
        
        hiddenLayerIndex = np.random.randint(1, len(child[randomKey]) - 1)
        minNeurons = 250

        if addLayer:
            print(f"Adding layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug

            if hiddenLayerIndex == 1:
                numHiddenNeurons = int(child[randomKey][1])
            elif hiddenLayerIndex == len(child[randomKey]) - 2:
                numHiddenNeurons = int(child[randomKey][-2])
            else:
                numHiddenNeurons = (int(child[randomKey][hiddenLayerIndex - 1]) + int(child[randomKey][hiddenLayerIndex + 1])) // 2

            child[randomKey].insert(hiddenLayerIndex, np.clip(getMutatedValue(numHiddenNeurons), a_min=minNeurons, a_max=None))

        elif removeLayer and len(child[randomKey]) > 4:
            print(f"Removing layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug
            child[randomKey].pop(hiddenLayerIndex)

        else:
            print(f"Mutating layer at index {hiddenLayerIndex}... " + " "*10, end="\r") # Debug
            child[randomKey][hiddenLayerIndex] = np.clip(getMutatedValue(child[randomKey][hiddenLayerIndex]), a_min=minNeurons, a_max=None)


        
    elif "rate" in randomKey.lower() or "reg" in randomKey.lower():
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        if isinstance(child[randomKey], list):
            for i in range(len(child[randomKey])):
                child[randomKey][i] = np.clip(getMutatedValue(child[randomKey][i]), 0.000001, 0.999999)
        else:
            child[randomKey] = np.clip(getMutatedValue(child[randomKey]), 0.000001, 0.999999)

        

    elif randomKey.lower() == 'optimizer':
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        choice = child[randomKey]
        while choice == child[randomKey]:
            choice = np.random.choice(optimizerChoices)
        
        child[randomKey] = choice
        

    elif 'activation' in randomKey.lower():
        print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
        choice = child[randomKey]
        while choice == child[randomKey]:
            choice = np.random.choice(activationChoices)
        
        child[randomKey] = choice

    # elif randomKey.lower() in ['batch_size', 'epochs']:
    #     print(f"Mutating {randomKey}... " + " "*20, end="\r") # Debug
    #     child[randomKey] = int(np.clip(getMutatedValue(child[randomKey]), 20, None))
        

    # time.sleep(1) # Debug

    return child

def convertNumpyTypes(obj):
    """
    Convert all numpy types to Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convertNumpyTypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertNumpyTypes(i) for i in obj]
    else:
        return obj


def saveBestModel(modelPath, bestParams, XTrain, yTrain, XTest, yTest):
    print("Training best model...                 ", end="\r")

    model, optimizer, lossFn, device = createModel(
        layers=bestParams['layers'],
        dropoutRates=bestParams['dropoutRates'],
        l2Reg=bestParams['l2Reg'],
        learningRate=bestParams['learningRate'],
        inputActivation=bestParams['inputActivation'],
        hiddenActivation=bestParams['hiddenActivation'],
        outputActivation=bestParams['outputActivation'],
        loss=bestParams['loss'],
        optimizer=bestParams['optimizer'],
        metrics=bestParams['metrics']
    )

    trainDataset = TensorDataset(torch.tensor(XTrain, dtype=torch.float32), torch.tensor(yTrain, dtype=torch.float32))
    testDataset = TensorDataset(torch.tensor(XTest, dtype=torch.float32), torch.tensor(yTest, dtype=torch.float32))

    trainLoader = DataLoader(trainDataset, batch_size=bestParams['batch_size'], shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=bestParams['batch_size'], shuffle=False)

    bestValAccuracy = model.fit(
        trainLoader,
        testLoader,
        optimizer,
        lossFn,
        bestParams['epochs'],
        device
        )

    print(f"Saving best model... ", end="\r")
    torch.save(model.state_dict(), modelPath)
    bestParamsPath = modelPath.removesuffix('.pt') + 'Params.json'


    bestParams = {k: convertNumpyTypes(v) for k, v in bestParams.items()}

    with open(bestParamsPath, 'w') as f:
        json.dump(bestParams, f, indent=4)

    print("Best model saved.     ", end="\r")
    return model

def saveHistory(generationHistory: list, historyPath):

    # Convert numpy types to Python types
    generationHistory = [convertNumpyTypes(individual) for individual in generationHistory]
    
    with open(historyPath, 'w') as f:
        json.dump(generationHistory, f, indent=4)


def neuralEvolution(XTrain, yTrain, XTest, yTest, initialParams, generations=50, populationSize=10, mutationRate=0.7, mutationForce=0.2, cpuPowerLimit=0.7, numParallelWorkers=1):
    """
    Perform Neuro-Evolution to find the best model parameters for the given dataset.

    ---
        
    ### Args:
        - XTrain (`ndarray`): Training data features
        - yTrain (`ndarray`): Training data labels
        - XTest (`ndarray`): Testing data features
        - yTest (`ndarray`): Testing data labels
        - initialParams (`dict`): Initial model parameters
        - generations (`int`): Number of generations
        - populationSize (`int`): Number of individuals in the population
        - mutationRate (`float`): Mutation rate (percentage of the population that will have one of their parameters mutated)
        - mutationForce (`float`): Mutation force (how much to mutate the parameters)
        - cpuPowerLimit (`float`): CPU power limit as a percentage (0.0 - 1.0) (only relevent if not using GPU)
        - numParallelWorkers (`int`): Number of parallel workers (default: 1)
    """
    cpuPowerLimit = min(max(cpuPowerLimit, 0.1), 0.9)
    numParallelWorkers = int(max(1, numParallelWorkers))

    modelPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\bestModel.pt'
    historyPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\generationHistory.json'

    print(f"\nPerforming Neuro-Evolution with {numParallelWorkers} parallel{'s' if numParallelWorkers > 1 else ''} workers for {generations} generations and a population size of {populationSize} (CPU Power Limit: ~{cpuPowerLimit * 100:.0f}%)...\n")

    generationHistory = []

    population = [initialParams.copy() for _ in range(populationSize)]

    bestIndividual = population[0]

    progressWheelIndex = 0
    totalTime = 0.0
    completion = 0.0

    print(" " * len(str([0] * numParallelWorkers)) + " |" + " " * 50 +
          f"gen 0/{generations} : " +
          getProgressBar(completion, wheelIndex=progressWheelIndex) + 
          f"Best Accuracy: {population[0]['fitness'] * 100:.2f}% / {bestIndividual['fitness'] * 100:.2f}%  |  Time remaining: --h--min --s  -  est. Finish Time: --h-- ", end='\r')

    for generation in range(generations):
        startTime = pd.Timestamp.now()

        topPopulation = population[:max(4, populationSize // 10)]

        newPopulation = []
        for _ in range(populationSize):
            print(f"Mutating individual {len(newPopulation) + 1}/{populationSize}... " + " "*25, end="\r")
            parent1, parent2 = np.random.choice(topPopulation, size=2, replace=False)

            child = crossover(parent1, parent2)

            child = mutate(child, mutationRate, mutationForce)

            newPopulation.append(child)

        newPopulation = evaluatePopulation(newPopulation, XTrain, yTrain, XTest, yTest, cpuPowerLimit, numParallelWorkers)

        population = topPopulation + newPopulation

        population.sort(key=lambda x: x['fitness'], reverse=True)

        # Keep the same population size
        population = population[:populationSize]

        generationHistory.append(population[0].copy())
        saveHistory(generationHistory, historyPath) # TODO bug fix

        if population[0]['fitness'] > bestIndividual['fitness']:
            bestIndividual = population[0]
            saveBestModel(modelPath, bestIndividual, XTrain, yTrain, XTest, yTest)

        progressWheelIndex += 1
        completion = progressWheelIndex / generations
        endTime = pd.Timestamp.now()
        elapsed = endTime - startTime

        totalTime += elapsed.total_seconds()
        averageTime = totalTime / progressWheelIndex

        estTimeRemaining = averageTime * (generations - progressWheelIndex)
        estHours = int(estTimeRemaining // 3600)
        estMinutes = int((estTimeRemaining % 3600) // 60)
        estSeconds = int(estTimeRemaining % 60)

        estDays = estHours // 24
        estHours = estHours % 24

        if estDays > 0:
            estTimeStr = f"{estDays}day{'s' if estDays > 1 else ''} {estHours:02}h{estMinutes:02}min {estSeconds:02}s"
        else:
            estTimeStr = f"{estHours:02}h{estMinutes:02}min {estSeconds:02}s"

        estFinishTime = pd.Timestamp.now() + pd.Timedelta(seconds=estTimeRemaining)
        if estTimeRemaining > 86400:
            estFinishTimeStr = estFinishTime.strftime('%d %b %Y %H:%M')
        else:
            estFinishTimeStr = f"{estFinishTime.hour:02}h{estFinishTime.minute:02}"

        print(" " * len(str([0] * numParallelWorkers)) + " |" + " " * 50 +
              f"gen {generation + 1}/{generations} : " +
              getProgressBar(completion, wheelIndex=progressWheelIndex) + 
              f"Best Accuracy: {newPopulation[0]['fitness'] * 100:.2f}% / {bestIndividual['fitness'] * 100:.2f}%  |  Time remaining: {estTimeStr}  -  est. Finish Time: {estFinishTimeStr} ", end='\r')

    return bestIndividual, generationHistory

def plotResults(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

if __name__ == '__main__':
    # Check if GPU is available
    print("System Information:")
    print(" - PyTorch version:", torch.__version__)
    print(" - Available GPUs:", torch.cuda.device_count())
    print(" - CUDA available:", torch.cuda.is_available())
    print(" - Current CUDA device id:", torch.cuda.current_device())
    print(" - CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    print("-" * 50, "\n")

    # Start memory profiler
    tracemalloc.start()
    # Main script
    csvPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\dataset_120x90.csv'

    csvRoot, csvName = csvPath.rsplit('\\', 1)
    csvName = csvName.removesuffix('.csv')
    csvParam = csvName.split('_')[-1]
    width, height = csvParam.split('x')
    try:
        width = int(width)
        height = int(height)
    except ValueError:
        print(f"Invalid video parameter: {csvParam}")
        exit(1)


    modelPath = csvRoot + f'/{csvParam}_model.keras'
    bestParamsPath = csvRoot + f'/bestModelParams.json'
    historyPath = csvRoot + f'/generationHistory.json'
    
    data = loadData(csvPath)
    resizedFrameData, originalFrameData = preprocessData(data)
    
    
    XTrain, XTest, yTrain, yTest = train_test_split(resizedFrameData, originalFrameData, test_size=0.2, random_state=42)

    inputNeurons = XTrain.shape[1]
    outputNeurons = yTrain.shape[1]

    print(f"\nInput neurons: {inputNeurons}")
    print(f"Output neurons: {outputNeurons}")

    input("Press Enter to start the training...")

    initialParams = {
        'layers': [
            inputNeurons,
            11000,
            11000,
            11000,
            outputNeurons
        ],
        'dropoutRates': [
            0.002095
        ],
        'l2Reg': 0.00042,
        'learningRate': 0.3,
        'inputActivation': 'relu',
        'hiddenActivation': 'tanh',
        'outputActivation': 'relu',
        'loss': 'l1',
        'optimizer': 'sgd',
        'metrics': [
            'accuracy',
            'AUC',
            'Precision',
            'Recall'
        ],
        'batch_size': 50,
        'epochs': 100,
        'fitness': 0.0
    }
    

    # check if there is a saved best model
    if os.path.isfile(bestParamsPath):
        with open(bestParamsPath, 'r') as f:
            bestModelParam = json.load(f)
            # check if input neurons and output neurons match the data
            if bestModelParam['layers'][0] == inputNeurons and bestModelParam['layers'][-1] == outputNeurons:
                initialParams = bestModelParam
                print("\nLoaded best model parameters.")
            else:
                print("\nInvalid best model parameters. Using initial parameters.")


    generations = 10
    populationSize = 5
    mutationRate = 0.9 # percentage of the population that will have one of their parameters mutated (does not include the parents of the population)
    mutationForce = 0.3 # will mutate numerical values by a factor of 1 +/- mutationForce
    cpuPowerLimit = 0.85
    nJobs = 1
    
    bestParams, generationHistory = neuralEvolution(XTrain=XTrain, yTrain=yTrain, XTest=XTest, yTest=yTest, initialParams=initialParams, generations=generations, populationSize=populationSize, mutationRate=mutationRate, mutationForce=mutationForce, cpuPowerLimit=cpuPowerLimit, numParallelWorkers=nJobs)    
    
    saveHistory(generationHistory, historyPath)