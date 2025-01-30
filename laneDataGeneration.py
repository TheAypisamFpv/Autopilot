import cv2
import os
import random
import numpy as np
from progressBar import getProgressBar

# Constants
VIDEO_DIR = 'D:/VS_Python_Project/Autopilot/Autopilot/Test drive'
NORMALIZED_SIZE = (640, 480)
DATASET_DIR = 'D:/VS_Python_Project/Autopilot/Autopilot/dataset'
NUM_IMAGES = 100  # Number of images to create

def getRandomFrame(videoPath: str):
    cap = cv2.VideoCapture(videoPath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    randomFrame = random.randint(0, frameCount - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, randomFrame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame, randomFrame
    else:
        return None, None

def getRandomVideoPath(directory: str):
    videos = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                videos.append(os.path.join(root, file))
    
    # print(f"loaded {len(videos)} video files")
    return random.choice(videos) if videos else None

def resizeImage(image, size):
    return cv2.resize(image, size)

def saveImage(resizedFrame, folderName, videoName, frameId):
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    imageName = f"{folderName}_{videoName}_{frameId}.png"
    imagePath = os.path.join(DATASET_DIR, imageName)
    cv2.imwrite(imagePath, resizedFrame)
    # print(f"Original image saved to {imagePath}")

def main():
    print("Starting image generation...")
    while True:
        existingImageCount = len([name for name in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, name))])
        if existingImageCount >= NUM_IMAGES:
            break

        videoPath = getRandomVideoPath(VIDEO_DIR)
        if not videoPath:
            print("No video files found in the directory.")
            return

        frame, frameId = getRandomFrame(videoPath)
        if frame is None:
            print("Failed to retrieve a frame from the video.")
            continue

        folderName = os.path.basename(os.path.dirname(videoPath))
        videoName = os.path.splitext(os.path.basename(videoPath))[0]

        # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resizedFrame = resizeImage(frame, NORMALIZED_SIZE)
        saveImage(resizedFrame, folderName, videoName, frameId)

        # Display progress bar
        progress = existingImageCount / NUM_IMAGES
        progressBar = getProgressBar(progress)
        progressText = f"{existingImageCount}/{NUM_IMAGES}".ljust(10)
        print(f"{progressText} {progressBar}", end='\r')

    # Display progress bar
    progress = existingImageCount / NUM_IMAGES
    progressBar = getProgressBar(progress)
    progressText = f"{existingImageCount}/{NUM_IMAGES}".ljust(10)
    print(f"{progressText} {progressBar}", end='\r')
    print("\nImage generation completed.")

if __name__ == "__main__":
    main()