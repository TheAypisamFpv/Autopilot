import os
import cv2
import pandas as pd
import numpy as np

# Constants
DATASET_DIR = 'D:/VS_Python_Project/Autopilot/Autopilot/autopilot AI/dataset'
NORMALIZED_SIZE = (120, 90)  # 4:3 aspect ratio
CSV_PATH = f'D:/VS_Python_Project/Autopilot/Autopilot/autopilot AI/dataset_{NORMALIZED_SIZE[0]}x{NORMALIZED_SIZE[1]}.csv'

def processImages(datasetDir:str, normalizedSize:tuple, csvPath:str):
    data = []

    fileCount = len([file for file in os.listdir(datasetDir) if file.endswith('_OG.png')])
    fileID = 0
    
    for root, _, files in os.walk(datasetDir):
        for file in files:
            if file.endswith('_OG.png'):
                fileID += 1
                print(f"Processing file {fileID:<4}/{fileCount:<4}", end='\r')
                ogPath = os.path.join(root, file)
                labelPath = ogPath.replace('_OG.png', '_LABELED.png')

                if os.path.exists(labelPath):
                    # Read and resize images
                    ogImage = cv2.imread(ogPath, cv2.IMREAD_GRAYSCALE)
                    labelImage = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)

                    resizedOgImage = cv2.resize(ogImage, normalizedSize)
                    resizedLabelImage = cv2.resize(labelImage, normalizedSize)

                    # Normalize pixel values to range [0, 1]
                    normalizedOgImage = resizedOgImage / 255.0
                    normalizedLabelImage = cv2.threshold(resizedLabelImage, 10, 255, cv2.THRESH_BINARY)[1] / 255.0

                    # Flatten the images
                    ogData = normalizedOgImage.flatten().tolist()
                    labelData = normalizedLabelImage.flatten().tolist()
                    data.append([ogData, labelData])
    print()

    # Create DataFrame and save to CSV
    print(f"Saving data to CSV file...")
    df = pd.DataFrame(data, columns=['OG', 'LABEL'])
    df.to_csv(csvPath, index=False)
    print(f"CSV file saved to {csvPath}")

if __name__ == "__main__":
    processImages(DATASET_DIR, NORMALIZED_SIZE, CSV_PATH)