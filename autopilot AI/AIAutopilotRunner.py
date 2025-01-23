import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

def displayFrame(frameId, frame, width, height):
    # Resize the frame to a predefined size (e.g., 960x720)
    predefinedWidth = 480
    predefinedHeight = 360
    frame = cv2.resize(frame, (predefinedWidth, predefinedHeight))
    cv2.imshow('Video Frame', frame)
    cv2.waitKey(1)  # Display the frame for 1 ms

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

def main():
    modelPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\bestModel.pt'
    paramsPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\bestModelParams.json'
    dataCsvPath = r'D:\VS_Python_Project\Autopilot\Autopilot\autopilot AI\dataset_120x90.csv'

    # Load model parameters from JSON file
    print(f"Loading model parameters from {paramsPath}...", end=' ')
    with open(paramsPath, 'r') as f:
        bestParams = json.load(f)
    print("Done")

    width, height = dataCsvPath.split('_')[-1].split('.')[0].split('x')
    outputWidth = int(width)
    outputHeight = int(height)

    print("Loading model...", end=' ')
    # Load model
    model = NeuralNet(
        layers=bestParams['layers'],
        dropoutRates=bestParams['dropoutRates'],
        l2Reg=bestParams['l2Reg'],
        inputActivation=bestParams['inputActivation'],
        hiddenActivation=bestParams['hiddenActivation'],
        outputActivation=bestParams['outputActivation']
    )
    state_dict = torch.load(modelPath, map_location=torch.device('cpu'))
    print("Using CPU")
    model.load_state_dict(state_dict)
    model.eval()
    print("Done\n")

    input("Press Enter to start the video...")

    # Load input frames from CSV file
    data_df = pd.read_csv(dataCsvPath)
    inputFrames = data_df['OG'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).values

    channels = 1  # Grayscale

    frameId = 0
    try:
        while frameId < len(inputFrames):
            # Get the input frame
            inputFrame = inputFrames[frameId]
            inputTensor = torch.tensor([inputFrame], dtype=torch.float32)
            
            with torch.no_grad():
                prediction = model(inputTensor).numpy()
    
            # Reshape the predicted output to an image
            predictedFrame = prediction.reshape((outputHeight, outputWidth, channels))
    
            # Display frame
            print(f"Frame ID: {frameId}")
            displayFrame(frameId, predictedFrame, outputWidth, outputHeight)
    
            # Increment frame ID for the next prediction
            frameId += 1
            time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()