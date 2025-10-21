import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
import json
from datetime import datetime, timedelta
import math
import time
from typing import List, Tuple, Dict, Any
import sys

from model.CreateModel import TrajectoryModel
from dataset.generator import (
    loadGpsData,
    findFutureGpsPoints,
    isGpsDataValid,
    findGpsPointAtTime,
    calculateFutureTrajectory,
)

blue = (251, 152, 52) #RGB 52, 152, 251
green = (52, 251, 152) #RGB 52, 251, 152

def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    """Draws text with a black outline."""
    # Draw the outline in black
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw the main text in the specified color
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def visualizePredictions(frame, predictions, groundTruth):
    originPoint = (frame.shape[1] // 2, frame.shape[0] - 5)
    vecToPixel = 20
    vectorThickness = 15

    # add text about which trajectory is which
    putTextWithOutline(frame, "Predicted (blue)", (10, originPoint[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
    putTextWithOutline(frame, "Ground Truth (green)", (10, originPoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)

    # Draw ground truth trajectory (green)
    if groundTruth is not None and len(groundTruth) > 0:
        currentPointGt = originPoint
        if isinstance(groundTruth, torch.Tensor):
            groundTruth = groundTruth.cpu().numpy()
        for point in groundTruth:
            endXGt = int(currentPointGt[0] + point[0] * vecToPixel)
            endYGt = int(currentPointGt[1] - point[1] * vecToPixel)
            cv2.line(frame, currentPointGt, (endXGt, endYGt), green, vectorThickness, cv2.LINE_AA)
            currentPointGt = (endXGt, endYGt)
            
            cv2.line(frame, currentPointGt, (currentPointGt[0]+200, currentPointGt[1]), (0, 0, 0), 2, cv2.LINE_AA)

    # Draw predicted trajectory (blue)
    if predictions is not None:
        currentPointPred = originPoint
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        for point in predictions:
            endXPred = int(currentPointPred[0] + point[0] * vecToPixel)
            endYPred = int(currentPointPred[1] - point[1] * vecToPixel)
            cv2.line(frame, currentPointPred, (endXPred, endYPred), blue, vectorThickness, cv2.LINE_AA)
            currentPointPred = (endXPred, endYPred)
            
            cv2.line(frame, currentPointPred, (currentPointPred[0]-200, currentPointPred[1]), (0, 0, 0), 2, cv2.LINE_AA)

    # Resize frame to fit the display
    newWidth = frame.shape[1] // 2
    newHeight = frame.shape[0] // 2

    frame = cv2.resize(frame, (newWidth, newHeight))

    cv2.imshow("Prediction (blue) vs Ground Truth (green)", frame)


def runModel(modelPath, videoPath, temporalContextTimeWindow=0.1):
    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryModel().to(device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- File and Data Loading ---
    jsonPath = videoPath.replace('.MP4', '.json')
    if not os.path.exists(videoPath):
        print(f"Video file not found: {videoPath}")
        return
    if not os.path.exists(jsonPath):
        print(f"JSON file not found: {jsonPath}")
        return

    gpsData = loadGpsData(jsonPath)
    if not gpsData:
        return

    # --- Video Capture Setup ---
    cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("Error: Could not open video with MSMF backend, trying default...")
        cap = cv2.VideoCapture(videoPath)  # Fallback to default
        if not cap.isOpened():
            print("Error: Could not open video with any backend.")
            return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    timePerFrame = 1.0 / fps
    videoStartTime = gpsData[0]['timestamp']
    frameBuffer = []
    frameBufferSize = max(int(fps * temporalContextTimeWindow), 2)
    nextFrameTime = time.time()

    # --- Main Processing Loop ---
    with torch.no_grad():
        while cap.isOpened():
            currentTime = time.time()
            if currentTime >= nextFrameTime:
                ret, frame = cap.read()
                if not ret:
                    break

                visualizeFrame = frame.copy()
                currentVideoMs = cap.get(cv2.CAP_PROP_POS_MSEC)
                currentFrameTime = videoStartTime + timedelta(milliseconds=currentVideoMs)

                # --- GPS and Dynamic Data Handling ---
                currentGpsPoint, _ = findGpsPointAtTime(gpsData, currentFrameTime)
                gpsIsValid, fixType, accuracy = [False, 0, float('inf')]
                if currentGpsPoint:
                    gpsIsValid, fixType, accuracy = isGpsDataValid(currentGpsPoint, gpsData, currentFrameTime)
                    speed = currentGpsPoint.get('speed2d', 0.0)
                    acceleration = currentGpsPoint.get('acceleration', 0.0)
                    turnRate = currentGpsPoint.get('turnRate', 0.0)
                else:
                    speed, acceleration, turnRate = 0.0, 0.0, 0.0

                # --- Debug Text Overlay ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                fontColor = (255, 255, 255)
                thickness = 2
                textX = 10
                textYStart = 40
                lineHeight = 40

                if currentGpsPoint:
                    # Timestamp
                    timeText = f"Time: {currentGpsPoint['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
                    putTextWithOutline(visualizeFrame, timeText, (textX, textYStart), font, fontScale, fontColor, thickness)

                    # GPS Fix
                    fixTypeVal = currentGpsPoint.get('fixType', 'N/A')
                    GPSColor = (0, 255, 0) if fixTypeVal == 3 else (0, 0, 255)
                    fixText = f"Fix: {fixTypeVal}"
                    putTextWithOutline(visualizeFrame, fixText, (textX, textYStart + lineHeight), font, fontScale, GPSColor, thickness)

                    # Accuracy
                    accuracyVal = currentGpsPoint.get('accuracy', 99)
                    accuracyColor = (0, 255, 0) if accuracyVal <= 3 else (0, 0, 255)
                    accText = f"Accuracy: {accuracyVal:.2f}m"
                    putTextWithOutline(visualizeFrame, accText, (textX, textYStart + 2 * lineHeight), font, fontScale, accuracyColor, thickness)

                    # Speed
                    speedVal = currentGpsPoint.get('speed2d', 0)
                    speed2dKph = speedVal * 3.6
                    speedText = f"Speed: {speedVal:.1f} m/s ({speed2dKph:.1f} km/h)"
                    putTextWithOutline(visualizeFrame, speedText, (textX, textYStart + 3 * lineHeight), font, fontScale, fontColor, thickness)

                    # Acceleration
                    accelVal = currentGpsPoint.get('acceleration', 0)
                    accelText = f"Acceleration: {accelVal:.1f} m/s^2"
                    putTextWithOutline(visualizeFrame, accelText, (textX, textYStart + 4 * lineHeight), font, fontScale, fontColor, thickness)

                    # Turn Rate
                    turnRateVal = currentGpsPoint.get('turnRate', 0)
                    turnText = f"Turn Rate: {turnRateVal:.0f} deg/s"
                    putTextWithOutline(visualizeFrame, turnText, (textX, textYStart + 5 * lineHeight), font, fontScale, fontColor, thickness)

                if not gpsIsValid:
                    warningText = "Warning: Low GPS quality"
                    cv2.putText(visualizeFrame, warningText, (textX, textYStart + 6 * lineHeight), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # --- Frame Buffering and Tensor Preparation ---
                frameBuffer.append(frame)

                prediction = None
                labels = None

                if len(frameBuffer) >= frameBufferSize:
                    currentFrameOrig = frameBuffer[-1]
                    prevFrameOrig = frameBuffer[0]
                    frameBuffer.pop(0)

                prevImage = Image.fromarray(cv2.cvtColor(prevFrameOrig, cv2.COLOR_BGR2RGB))
                currentImage = Image.fromarray(cv2.cvtColor(currentFrameOrig, cv2.COLOR_BGR2RGB))
                prevImgTensor = transform(prevImage).unsqueeze(0).to(device)
                currentImgTensor = transform(currentImage).unsqueeze(0).to(device)

                # --- Prediction ---
                prediction, _, _ = model(currentImgTensor, prevImgTensor)

                # --- Ground Truth Trajectory ---
                labels = None
                if gpsIsValid and currentGpsPoint:
                    interval = 3.0 / 12.0
                    futureTrajectory = calculateFutureTrajectory(gpsData, currentGpsPoint, currentFrameTime, duration=3.0, interval=interval)
                    validVectors = [v for v in futureTrajectory if v is not None]
                    if len(validVectors) >= 12:
                        labelsList = [[v['x'], v['y']] for v in validVectors]
                        labels = torch.tensor(labelsList[:12], dtype=torch.float32)

                # --- Visualization ---
                visualizePredictions(visualizeFrame, prediction.squeeze() if prediction is not None else None, labels)

                # Check for quit key
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                # Update next frame time
                nextFrameTime += timePerFrame

            else:
                # Wait until next frame time
                waitTime = (nextFrameTime - currentTime) * 1000
                key = cv2.waitKey(max(1, int(waitTime)))
                if key & 0xFF == ord('q'):
                    break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    modelPath = r"D:\VS_Python_Project\Autopilot\Autopilot\training\run10\best_model.pth"
    videoPath = r"D:\VS_Python_Project\Autopilot\Autopilot\Test_drive\2025.06.25\GP035971.MP4"
    runModel(modelPath, videoPath)
