import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
from datetime import timedelta
import math
import time

from model.CreateModel import TrajectoryModel
from dataset.generator import (
    loadGpsData,
    findFutureGpsPoints,
    isGpsDataValid,
    findGpsPointAtTime,
    calculateFutureTrajectory,
)

blue = (251, 152, 52) #RGB 52, 152, 251
gray = (128, 128, 128)

def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    """Draws text with a black outline."""
    # Draw the outline in black
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw the main text in the specified color
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def visualizePredictions(frame, predictions, groundTruth, attnMap=None, warpingMatrix=None, dstPoints=None):
    trajectoryCanvasWidth = 1000
    trajectoryCanvasHeight = 1500
    trajectoryFrame = np.zeros((trajectoryCanvasHeight, trajectoryCanvasWidth, 4), dtype=np.uint8)  # BGRA
    vecToPixel = 20 # pixels per meter
    vectorThickness = 40
    originPoint = (trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness)
    
    # Parameters for perspective warp using angles
    # trajAngle = 0  # degrees, 0 parallel to image plane, 90 perpendicular
    # translationY = 0.0  # meters, up/down
    # translationZ = 20.0  # meters, forward/backward
    # scale = 1.0  # overall scale
    

    # Draw ground truth trajectory (gray)
    if groundTruth is not None and len(groundTruth) > 0:
        currentPointGt = originPoint
        if isinstance(groundTruth, torch.Tensor):
            groundTruth = groundTruth.cpu().numpy()
        for point in groundTruth:
            endXGt = int(currentPointGt[0] + point[0] * vecToPixel)
            endYGt = int(currentPointGt[1] - point[1] * vecToPixel)
            cv2.line(trajectoryFrame, currentPointGt, (endXGt, endYGt), (*gray, 255), vectorThickness, cv2.LINE_AA)
            currentPointGt = (endXGt, endYGt)
            
            # cv2.line(trajectoryFrame, currentPointGt, (currentPointGt[0]+200, currentPointGt[1]), (0, 0, 0), 2, cv2.LINE_AA)

    # Draw predicted trajectory (blue)
    if predictions is not None:
        currentPointPred = originPoint
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        for point in predictions:
            endXPred = int(currentPointPred[0] + point[0] * vecToPixel)
            endYPred = int(currentPointPred[1] - point[1] * vecToPixel)
            cv2.line(trajectoryFrame, currentPointPred, (endXPred, endYPred), (*blue, 255), vectorThickness, cv2.LINE_AA)
            currentPointPred = (endXPred, endYPred)
            
            # cv2.line(trajectoryFrame, currentPointPred, (currentPointPred[0]-200, currentPointPred[1]), (0, 0, 0), 2, cv2.LINE_AA)

    # Apply 3D warping
    srcPoints = np.array([(0, 0), (trajectoryCanvasWidth - 1, 0), (trajectoryCanvasWidth - 1, trajectoryCanvasHeight - 1), (0, trajectoryCanvasHeight - 1)], dtype=np.float32)
    
    if warpingMatrix is not None:
        M = warpingMatrix
        if dstPoints is None:
            # Fallback: compute dstPoints if not provided
            centerX = frame.shape[1] // 2
            centerY = frame.shape[0] // 2
            f = 1000
            cx = centerX
            cy = centerY
            w = 50 * 1.0  # scale=1.0
            h = 75 * 1.0
            d = 39  # translationZ
            theta = -math.radians(102)  # trajAngle
            R = np.array([
                [1, 0, 0],
                [0, math.cos(theta), -math.sin(theta)],
                [0, math.sin(theta), math.cos(theta)]
            ])
            center_world = np.array([0, 11, d])  # translationY
            corners_local = [
                (-w/2, h/2, 0),
                (w/2, h/2, 0),
                (w/2, -h/2, 0),
                (-w/2, -h/2, 0)
            ]
            dstPointsComputed = []
            for local in corners_local:
                world = R @ np.array(local) + center_world
                if world[2] > 0:
                    u = f * world[0] / world[2] + cx
                    v = f * world[1] / world[2] + cy
                    dstPointsComputed.append((u, v))
                else:
                    dstPointsComputed.append((cx, cy))
            dstPointsComputed = np.array(dstPointsComputed, dtype=np.float32)
            dstPointsComputed = dstPointsComputed[::-1]
            dstPoints = dstPointsComputed
    else:
        # Original computation (for backward compatibility)
        centerX = frame.shape[1] // 2
        centerY = frame.shape[0] // 2
        f = 1000
        cx = centerX
        cy = centerY
        w = 50 * 1.0
        h = 75 * 1.0
        d = 39
        theta = -math.radians(102)
        R = np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]
        ])
        center_world = np.array([0, 11, d])
        corners_local = [
            (-w/2, h/2, 0),
            (w/2, h/2, 0),
            (w/2, -h/2, 0),
            (-w/2, -h/2, 0)
        ]
        dstPoints = []
        for local in corners_local:
            world = R @ np.array(local) + center_world
            if world[2] > 0:
                u = f * world[0] / world[2] + cx
                v = f * world[1] / world[2] + cy
                dstPoints.append((u, v))
            else:
                dstPoints.append((cx, cy))
        dstPoints = np.array(dstPoints, dtype=np.float32)
        dstPoints = dstPoints[::-1]
        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    
    warped = cv2.warpPerspective(trajectoryFrame, M, (frame.shape[1], frame.shape[0]))
    
    # Blend with alpha
    alpha = warped[..., 3] / 255.0
    frame = ((1 - alpha[..., None]) * frame + alpha[..., None] * warped[..., :3]).astype(np.uint8)

    # Draw red dots on visualizeFrame at destination points
    # for i, pt in enumerate(dstPoints):
    #     cv2.circle(frame, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
    #     # Draw lines between points
    #     cv2.line(frame, tuple(dstPoints[i].astype(int)), tuple(dstPoints[(i+1)%len(dstPoints)].astype(int)), (0, 0, 255), 2)

    # Add trajectory labels on the main frame
    putTextWithOutline(frame, "Predicted (blue)", (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
    putTextWithOutline(frame, "Ground Truth (gray)", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, gray, 2)

    # Overlay attention map if provided
    if attnMap is not None:
        attnResized = cv2.resize(attnMap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        attnNorm = attnResized / (attnResized.max() + 1e-8)
        attnDisplay = (attnNorm ** 0.5)  # gamma correction to amplify midrange
        attnDisplay = (attnDisplay * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attnDisplay, cv2.COLORMAP_JET)

        # Overlay with alpha
        alpha = 0.2
        # Frame = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        heatmap = cv2.resize(heatmap, (frame.shape[1] // 5, frame.shape[0] // 5))
        cv2.imshow("Attention Map Overlay", heatmap)

    # Resize frame to fit the display
    newWidth = frame.shape[1] // 2
    newHeight = frame.shape[0] // 2

    trajectoryResized = cv2.resize(trajectoryFrame, (trajectoryFrame.shape[1] // 4, trajectoryFrame.shape[0] // 4))
    cv2.imshow("Trajectory Frame", trajectoryResized)

    frame = cv2.resize(frame, (newWidth, newHeight))

    cv2.imshow("Prediction (blue) vs Ground Truth (gray)", frame)


def runModel(modelPath, videoPath, temporalContextTimeWindow=0.1, useGPU=True):
    # --- Initialization ---
    device = torch.device("cuda" if useGPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = TrajectoryModel(feat_dim=512, hidden_dim=1024, pred_steps=12).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    transformVisual = transforms.Compose([
        transforms.Resize((270, 480))
    ])

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

    # Get frame dimensions for warping
    ret, testFrame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    frameHeight, frameWidth = testFrame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)  # Reset to beginning

    # Precompute warping matrix and points
    trajAngle = 102
    translationY = 11
    translationZ = 39
    scale = 1.0
    trajectoryCanvasWidth = 1000
    trajectoryCanvasHeight = 1500
    srcPoints = np.array([(0, 0), (trajectoryCanvasWidth - 1, 0), (trajectoryCanvasWidth - 1, trajectoryCanvasHeight - 1), (0, trajectoryCanvasHeight - 1)], dtype=np.float32)
    centerX = frameWidth // 2
    centerY = frameHeight // 2
    f = 1000
    cx = centerX
    cy = centerY
    w = 50 * scale
    h = 75 * scale
    d = translationZ
    theta = -math.radians(trajAngle)
    R = np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])
    center_world = np.array([0, translationY, d])
    corners_local = [
        (-w/2, h/2, 0),
        (w/2, h/2, 0),
        (w/2, -h/2, 0),
        (-w/2, -h/2, 0)
    ]
    dstPoints = []
    for local in corners_local:
        world = R @ np.array(local) + center_world
        if world[2] > 0:
            u = f * world[0] / world[2] + cx
            v = f * world[1] / world[2] + cy
            dstPoints.append((u, v))
        else:
            dstPoints.append((cx, cy))
    dstPoints = np.array(dstPoints, dtype=np.float32)
    dstPoints = dstPoints[::-1]  # Reverse order
    warpingMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    timePerFrame = 1.0 / fps
    videoStartTime = gpsData[0]['timestamp']
    frameBuffer = []
    frameBufferSize = max(int(fps * temporalContextTimeWindow), 2)
    nextFrameTime = time.time()

    # Create control window with sliders
    # cv2.namedWindow("Controls")
    # cv2.createTrackbar("Traj Angle", "Controls", 0, 150, lambda x: None)  # 0 parallel, 150 perpendicular
    # cv2.createTrackbar("Translation Y", "Controls", 50, 100, lambda x: None)  # 0 = -50m, 50 = 0m, 100 = 50m
    # cv2.createTrackbar("Translation Z", "Controls", 30, 80, lambda x: None)  # 0 = 20m, 30 = 50m, 80 = 100m
    # cv2.createTrackbar("Scale", "Controls", 10, 20, lambda x: None)  # 10 = 1.0, 20 = 2.0

    lastPrediction = None
    lastLabels = None
    lastAttnMap = None

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


                # --- Frame Buffering and Tensor Preparation ---
                frameBuffer.append(frame)

                prediction = None
                labels = None
                attnMaps = None

                if len(frameBuffer) >= frameBufferSize:
                    currentFrameOrig = frameBuffer[-1]
                    prevFrameOrig = frameBuffer[0]
                    frameBuffer.pop(0)

                    prevImage = Image.fromarray(cv2.cvtColor(prevFrameOrig, cv2.COLOR_BGR2RGB))
                    currentImage = Image.fromarray(cv2.cvtColor(currentFrameOrig, cv2.COLOR_BGR2RGB))
                    prevImgTensor = transform(prevImage).unsqueeze(0).to(device)
                    currentImgTensor = transform(currentImage).unsqueeze(0).to(device)

                    # OgSize = visualizeFrame.shape
                    # visualizeFrame = cv2.resize(currentFrameOrig, (480, 270))
                    # visualizeFrame = cv2.resize(visualizeFrame, (OgSize[1], OgSize[0]))
                    

                    # --- Prediction ---
                    prediction, _, attnMaps = model(currentImgTensor, prevImgTensor)

                    if attnMap is not None:
                        print(f"Attention map stats: min={attnMap.min():.4f}, max={attnMap.max():.4f}, mean={attnMap.mean():.4f}")

                    lastPrediction = prediction
                    lastAttnMap = attnMaps[-1].squeeze().cpu().numpy() if attnMaps else None


                    # Show previous frame with attention map
                    # if attnMaps:
                    #     prevAttn = attnMaps[-1].squeeze().cpu().numpy()
                    #     prevVis = prevFrameOrig.copy()
                    #     # Overlay attention map
                    #     attnResized = cv2.resize(prevAttn, (prevVis.shape[1], prevVis.shape[0]), interpolation=cv2.INTER_LINEAR)
                    #     attnNorm = cv2.normalize(attnResized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    #     heatmap = cv2.applyColorMap(attnNorm, cv2.COLORMAP_JET)
                    #     alpha = 0.5
                    #     prevVis = cv2.addWeighted(prevVis, 1 - alpha, heatmap, alpha, 0)
                    #     # Resize to 50% smaller than current display (current is //2, so //4)
                    #     prevHeight, prevWidth = prevVis.shape[:2]
                    #     prevResized = cv2.resize(prevVis, (prevWidth // 4, prevHeight // 4))
                    #     cv2.imshow("Previous Frame", prevResized)

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


                # --- Ground Truth Trajectory ---
                labels = None
                if gpsIsValid and currentGpsPoint:
                    interval = 3.0 / 12.0
                    futureTrajectory = calculateFutureTrajectory(gpsData, currentGpsPoint, currentFrameTime, duration=3.0, interval=interval)
                    validVectors = [v for v in futureTrajectory if v is not None]
                    if len(validVectors) >= 12:
                        labelsList = [[v['x'], v['y']] for v in validVectors]
                        labels = torch.tensor(labelsList[:12], dtype=torch.float32)

                lastLabels = labels

                # --- Visualization ---
                attnMap = lastAttnMap
                visualizePredictions(visualizeFrame, lastPrediction.squeeze() if lastPrediction is not None else None, lastLabels, attnMap, warpingMatrix, dstPoints)

                # Check for quit key
                key = cv2.waitKey(1)
                if key != -1:
                    print(f"Key pressed: {key}")
                if key & 0xFF == ord('q'):
                    print(f"Final warping parameters:")
                    print(f"Traj Angle: 102°")
                    print(f"Translation Y: 11m")
                    print(f"Translation Z: 39m")
                    print(f"Scale: 1.0")
                    break
                elif key == 106 or key == 242:  # J: go back 10s
                    print("Seeking back 10 seconds")
                    newMs = max(0, currentVideoMs - 10000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, newMs)
                    frameBuffer = []
                    nextFrameTime = time.time()
                elif key == 108 or key == 243:  # L: go forward 10s
                    print("Seeking forward 10 seconds")
                    newMs = currentVideoMs + 10000
                    cap.set(cv2.CAP_PROP_POS_MSEC, newMs)
                    frameBuffer = []
                    nextFrameTime = time.time()

                # Update next frame time
                nextFrameTime += timePerFrame

            else:
                # Wait until next frame time
                waitTime = (nextFrameTime - currentTime) * 1000

                # --- Visualization ---
                attnMap = lastAttnMap
                visualizePredictions(visualizeFrame, lastPrediction.squeeze() if lastPrediction is not None else None, lastLabels, attnMap, warpingMatrix, dstPoints)

                key = cv2.waitKey(max(1, int(waitTime)))
                if key != -1:
                    print(f"Key pressed: {key}")
                if key & 0xFF == ord('q'):
                    print(f"Final warping parameters:")
                    print(f"Traj Angle: 102°")
                    print(f"Translation Y: 11m")
                    print(f"Translation Z: 39m")
                    print(f"Scale: 1.0")
                    break
                elif key == 2424832 or key == 242:  # Left arrow: go back 10s
                    print("Seeking back 10 seconds")
                    newMs = max(0, currentVideoMs - 10000)
                    cap.set(cv2.CAP_PROP_POS_MSEC, newMs)
                    frameBuffer = []
                    nextFrameTime = time.time()
                elif key == 2555904 or key == 243:  # Right arrow: go forward 10s
                    print("Seeking forward 10 seconds")
                    newMs = currentVideoMs + 10000
                    cap.set(cv2.CAP_PROP_POS_MSEC, newMs)
                    frameBuffer = []
                    nextFrameTime = time.time()

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    # cv2.destroyWindow("Controls")


if __name__ == '__main__':
    modelPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\training\run13\best_model.pth"
    videoPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.06.24\GP065969.MP4"
    useGPU = False
    runModel(modelPath, videoPath, useGPU=useGPU)
