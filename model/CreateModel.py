import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProcessingBranch(nn.Module):
    def __init__(self):
        super(ImageProcessingBranch, self).__init__()
        # Shared layers for initial processing
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convBlock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convBlock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Processing after concatenation
        self.finalConv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img1, img2):
        # Process both images with shared weights
        feat1 = self.convBlock1(img1)
        feat2 = self.convBlock1(img2)

        # Continue processing
        feat1 = self.convBlock2(feat1)
        feat2 = self.convBlock2(feat2)

        # Continue processing
        feat1 = self.convBlock3(feat1)
        feat2 = self.convBlock3(feat2)

        # Concatenate features
        concatenatedFeatures = torch.cat((feat1, feat2), dim=1)
        
        # Final convolution and pooling
        finalFeatures = self.finalConv(concatenatedFeatures)
        vectorizedFeatures = self.globalAvgPool(finalFeatures)
        vectorizedFeatures = vectorizedFeatures.view(vectorizedFeatures.size(0), -1)
        
        return vectorizedFeatures

class DynamicInputProcessingBranch(nn.Module):
    def __init__(self):
        super(DynamicInputProcessingBranch, self).__init__()
        self.dense_block = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

    def forward(self, x):
        return self.dense_block(x)

class TrajectoryPredictionModel(nn.Module):
    def __init__(self):
        super(TrajectoryPredictionModel, self).__init__()
        self.imageBranch = ImageProcessingBranch()
        self.dynamicBranch = DynamicInputProcessingBranch()
        
        self.fusion_and_fully_connected = nn.Sequential(
            nn.Linear(512 + 32, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    def forward(self, img1, img2, dynamic_data):
        imageFeatures = self.imageBranch(img1, img2)
        dynamicFeatures = self.dynamicBranch(dynamic_data)
        
        fusedFeatures = torch.cat((imageFeatures, dynamicFeatures), dim=1)
        
        output = self.fusion_and_fully_connected(fusedFeatures)
        
        # Reshape to (batch_size, 12, 2)
        output = output.view(output.size(0), 12, 2)
        
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor
    # Input shape for images: (batch_size, channels, height, width) -> (1, 1, 480, 270)
    # Input shape for dynamic data: (batch_size, features) -> (1, 3)
    print("Creating dummy inputs for testing...")
    dummyImg1 = torch.randn(1, 3, 270, 480)
    dummyImg2 = torch.randn(1, 3, 270, 480)
    dummyDynamicData = torch.randn(1, 3)

    print("\nInstantiating the TrajectoryPredictionModel...")
    # Instantiate the model
    model = TrajectoryPredictionModel()

    print("\nModel instantiated successfully.")
    # Get the model's prediction
    prediction = model(dummyImg1, dummyImg2, dummyDynamicData)

    # Print the output shape
    print("\nOutput shape:", prediction.shape)
    
    # Count the number of parameters
    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {totalParams/1e6:.1f}M")
