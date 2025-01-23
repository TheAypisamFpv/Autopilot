import cv2
import os
import random
import numpy as np
import collections

# Constants
VIDEO_DIR = 'D:/VS_Python_Project/Autopilot/Autopilot/autopilot non AI/Test drive'
NORMALIZED_SIZE = (480, 360)  # 4:3 aspect ratio
DATASET_DIR = 'D:/VS_Python_Project/Autopilot/Autopilot/autopilot AI/dataset'
SMOOTHING_WINDOW_SIZE = 10


# Global variables
drawing = False
ix, iy = -1, -1
xPositions = collections.deque(maxlen=SMOOTHING_WINDOW_SIZE)
yPositions = collections.deque(maxlen=SMOOTHING_WINDOW_SIZE)
pointA = None

def getRandomFrame(videoPath: str):
    cap = cv2.VideoCapture(videoPath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    randomFrame = random.randint(0, frameCount - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, randomFrame)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def getRandomVideoPath(directory: str):
    videos = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                videos.append(os.path.join(root, file))
    
    print(f"loaded {len(videos)} video files")
    return random.choice(videos) if videos else None

def resizeImage(image, size):
    return cv2.resize(image, size)

def smoothPosition(x, y):
    xPositions.append(x)
    yPositions.append(y)
    smoothedX = sum(xPositions) // len(xPositions)
    smoothedY = sum(yPositions) // len(yPositions)
    return smoothedX, smoothedY

def drawLaneLines(event, x, y, flags, param):
    global ix, iy, drawing, pointA


    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if pointA is None:
                pointA = (x, y)
            else:
                cv2.line(param, pointA, (x, y), (255, 255, 255), 2)  # Draw line from point A to current point
                pointA = None  # Reset point A
        else:
            drawing = True
            ix, iy = x, y


    elif event == cv2.EVENT_MOUSEMOVE:
        x, y = smoothPosition(x, y)
        
        if pointA is not None:
            temp_image = param.copy()
            cv2.line(temp_image, pointA, (x, y), (255, 255, 255), 2)  # Draw preview line
            cv2.imshow('image', temp_image)
        elif drawing:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.line(param, (ix, iy), (x, y), (255, 255, 255), 2)
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                cv2.line(param, (ix, iy), (x, y), (0, 0, 0), 8)
            ix, iy = x, y

    x, y = smoothPosition(x, y)
    
            

def saveImages(resizedFrame, laneLinesImage):
    # if the lane lines image is empty, skip saving
    if np.all(laneLinesImage == 0):
        print("Lane lines image is empty. Skipping save.")
        return
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    imageId = -1
    # check if the imageId already exists
    while os.path.exists(os.path.join(DATASET_DIR, f'{imageId}_OG.png')) or imageId == -1:
        imageId = random.randint(1000, 9999)
    
    originalImagePath = os.path.join(DATASET_DIR, f'{imageId}_OG.png')
    laneLinesImagePath = os.path.join(DATASET_DIR, f'{imageId}_LABELED.png')

    cv2.imwrite(originalImagePath, resizedFrame)
    cv2.imwrite(laneLinesImagePath, laneLinesImage)

    print(f"Original image saved to {originalImagePath}")
    print(f"Lane lines image saved to {laneLinesImagePath}")

def main():
    while True:
        videoPath = getRandomVideoPath(VIDEO_DIR)
        if not videoPath:
            print("No video files found in the directory.")
            return

        print(f"Selected video: {videoPath}")

        frame = getRandomFrame(videoPath)
        if frame is None:
            print("Failed to retrieve a frame from the video.")
            return

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resizedFrame = resizeImage(grayFrame, NORMALIZED_SIZE)
        laneLinesImage = np.zeros_like(resizedFrame)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', drawLaneLines, laneLinesImage)

        while True:
            combinedImage = cv2.addWeighted(resizedFrame, 0.7, laneLinesImage, 0.3, 0)
            cv2.imshow('image', combinedImage)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC key to quit the program
                cv2.destroyAllWindows()
                return
            elif k == ord('q') or k == 13:  # 'q' key or ENTER key to finish current image and load next
                saveImages(resizedFrame, laneLinesImage)
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()