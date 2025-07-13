import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProcessingBranch(nn.Module):
    def __init__(self):
        super(ImageProcessingBranch, self).__init__()
        
        # More sophisticated convolutional blocks with additional features
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        # Spatial attention mechanism
        self.spatialAttention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Processing after concatenation
        self.finalConv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        )
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

    def forwardOne(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        
        # Apply spatial attention
        attention = self.spatialAttention(x)
        x = x * attention  # Element-wise multiplication for attention
        
        return x

    def forward(self, img1, img2):
        # Process both images with shared weights
        feat1 = self.forwardOne(img1)
        feat2 = self.forwardOne(img2)

        # Concatenate features
        concatenatedFeatures = torch.cat((feat1, feat2), dim=1)
        
        # Final convolution and pooling
        finalFeatures = self.finalConv(concatenatedFeatures)
        vectorizedFeatures = self.globalAvgPool(finalFeatures)
        vectorizedFeatures = vectorizedFeatures.view(vectorizedFeatures.size(0), -1)
        
        return vectorizedFeatures

class TrajectoryPredictionModel(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionModel, self).__init__()
        self.imageBranch = ImageProcessingBranch()
        
        # Dynamic Input Processing Branch
        self.dynamicBranch = nn.Sequential(
            nn.Linear(1, 32),  # Only speed as input
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        
        self.fusionAndFullyConnected = nn.Sequential(
            nn.Linear(1024 + 32, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # 6 vectors × 2 coordinates
        )

    def forward(self, img1, img2, dynamicData):
        imageFeatures = self.imageBranch(img1, img2)
        dynamicFeatures = self.dynamicBranch(dynamicData)
        
        fusedFeatures = torch.cat((imageFeatures, dynamicFeatures), dim=1)
        
        output = self.fusionAndFullyConnected(fusedFeatures)
        
        # Reshape to (batch_size, 6, 2)
        output = output.view(output.size(0), 6, 2)
        
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor
    # Input shape for images: (batch_size, channels, height, width) -> (1, 3, 270, 480)
    # Input shape for dynamic data: (batch_size, features) -> (1, 1) - only speed
    print("Creating dummy inputs for testing...")
    
    # Create batch with 2 samples to avoid BatchNorm issues
    batchSize = 2
    dummyImg1 = torch.randn(batchSize, 3, 270, 480)
    dummyImg2 = torch.randn(batchSize, 3, 270, 480)
    dummyDynamicData = torch.randn(batchSize, 1)  # Only speed

    print("\nInstantiating the TrajectoryPredictionModel...")
    # Instantiate the model
    model = TrajectoryPredictionModel()
    
    # Set the model to evaluation mode to handle BatchNorm properly
    model.eval()

    print("\nModel instantiated successfully.")
    # Get the model's prediction
    with torch.no_grad():  # No need to track gradients for testing
        prediction = model(dummyImg1, dummyImg2, dummyDynamicData)

    # Print the output shape
    print("\nOutput shape:", prediction.shape)
    print(f"Expected: torch.Size([{batchSize}, 6, 2])")  # batch_size, 6 vectors, 2 coordinates
    
    # Count the number of parameters
    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {totalParams}")
