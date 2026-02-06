import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
from datetime import datetime, timedelta
import math
import time
import json
import glob
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
import warnings

from model.CreateModel import TrajectoryModel
from dataset.generator import (
    loadEgomotionParquet,
    loadCameraTimestampsParquet,
    interpolateEgomotionState,
    calculateFutureTrajectoryEgomotion,
)


blueColor = (251, 152, 52)  # RGB 52, 152, 251
grayColor = (128, 128, 128)

trajectoryCanvasWidth = 500
trajectoryCanvasHeight = 750
vecToPixel = 10
vectorThickness = 10
scaleFactor = 1.0


global DOWNSCALE, SHOWATTENTION, SHOWORIGINALTRAJ, USEGPU, DEBUG


def safeTimestampFromUs(timestampUs):
    if timestampUs is None:
        return None
    try:
        return datetime.fromtimestamp(timestampUs / 1e6)
    except (OSError, OverflowError, ValueError):
        try:
            return datetime(1970, 1, 1) + timedelta(microseconds=int(timestampUs))
        except Exception:
            return None


def resolveNvidiaEgomotionPaths(videoPath):
    baseName = os.path.basename(videoPath)
    if ".camera_front_wide_120fov.mp4" in baseName:
        clipUuid = baseName.split(".camera_front_wide_120fov.mp4")[0]
    else:
        clipUuid = os.path.splitext(baseName)[0]

    cameraDir = os.path.dirname(videoPath)
    timestampsPath = os.path.join(cameraDir, f"{clipUuid}.camera_front_wide_120fov.timestamps.parquet")

    egomotionCandidates = [
        os.path.join(cameraDir, f"{clipUuid}.egomotion.parquet"),
        os.path.join(os.path.dirname(cameraDir), "egomotion", f"{clipUuid}.egomotion.parquet"),
        os.path.join(os.path.dirname(os.path.dirname(cameraDir)), "labels", "egomotion", f"{clipUuid}.egomotion.parquet"),
    ]
    egomotionPath = next((p for p in egomotionCandidates if os.path.exists(p)), None)

    if os.path.exists(timestampsPath) and egomotionPath:
        return {
            "clipUuid": clipUuid,
            "timestampsPath": timestampsPath,
            "egomotionPath": egomotionPath,
        }

    return None


def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)


def loadCalibrationData(calibrationRoot, clipUuid, cameraName):
    cameraIntrinsicsPath = os.path.join(calibrationRoot, "camera_intrinsics")
    sensorExtrinsicsPath = os.path.join(calibrationRoot, "sensor_extrinsics")
    vehicleDimensionsPath = os.path.join(calibrationRoot, "vehicle_dimensions")

    intrinsics = None
    extrinsics = None
    vehicleDims = None

    for pqFile in glob.glob(os.path.join(cameraIntrinsicsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            intrinsics = df.loc[(clipUuid, cameraName)].to_dict()
            break
        except KeyError:
            continue

    for pqFile in glob.glob(os.path.join(sensorExtrinsicsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            extrinsics = df.loc[(clipUuid, cameraName)].to_dict()
            break
        except KeyError:
            continue

    for pqFile in glob.glob(os.path.join(vehicleDimensionsPath, "*.parquet")):
        df = pd.read_parquet(pqFile)
        try:
            vehicleDims = df.loc[clipUuid].to_dict()
            break
        except KeyError:
            continue

    return intrinsics, extrinsics, vehicleDims


def projectPointFisheye(pCam, fwPoly, cx, cy, allowBehind=False):
    x, y, z = pCam
    if z <= 0:
        if not allowBehind:
            return None
        z = 1e-3
    rho = math.sqrt(x * x + y * y)
    if rho < 1e-8:
        return (cx, cy)
    dist = math.sqrt(rho * rho + z * z)
    theta = math.acos(z / dist)
    rD = sum(fwPoly[i] * (theta ** i) for i in range(5))
    u = cx + rD * (x / rho)
    v = cy + rD * (y / rho)
    return (u, v)


def projectRigPoints(pointsRig, intrinsics, extrinsics, downscale):
    if intrinsics is None or extrinsics is None or pointsRig is None:
        return None

    rRigFromCam = R_scipy.from_quat([extrinsics['qx'], extrinsics['qy'], extrinsics['qz'], extrinsics['qw']]).as_matrix()
    rCamFromRig = rRigFromCam.T
    t = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])

    s = 1.0 / downscale
    cx = float(intrinsics['cx']) * s
    cy = float(intrinsics['cy']) * s
    fwPoly = np.array([intrinsics[f'fw_poly_{i}'] for i in range(5)], dtype=float) * s

    projected = []
    for pRig in pointsRig:
        pCam = rCamFromRig @ (pRig - t)
        projected.append(projectPointFisheye(pCam, fwPoly, cx, cy, allowBehind=True))
    return projected


def buildRigPointsFromVectors(vectors, scale=1.0):
    if vectors is None or len(vectors) == 0:
        return None
    vectorsNp = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors, dtype=float)

    steps = np.zeros((len(vectorsNp), 3), dtype=float)
    steps[:, 0] = vectorsNp[:, 1]
    steps[:, 1] = -vectorsNp[:, 0]
    steps[:, 2] = 0.0

    points = np.zeros((len(vectorsNp) + 1, 3), dtype=float)
    points[1:] = np.cumsum(steps, axis=0) * scale
    return points


def drawProjectedPolyline(overlay, projectedPoints, color, thickness):
    if projectedPoints is None:
        return

    currentSegment = []
    for pt in projectedPoints:
        if pt is not None:
            currentSegment.append(pt)
        else:
            if len(currentSegment) >= 2:
                pts = np.array(currentSegment, dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            currentSegment = []
    if len(currentSegment) >= 2:
        pts = np.array(currentSegment, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def drawProjectedRibbonFromRig(overlay, rigPoints, intrinsics, extrinsics, downscale, widthMeters, color):
    if rigPoints is None or intrinsics is None or extrinsics is None:
        return
    if len(rigPoints) < 2:
        return

    count = len(rigPoints)
    dirs = [None] * (count - 1)
    for i in range(count - 1):
        seg = rigPoints[i + 1][:2] - rigPoints[i][:2]
        segLen = float(np.linalg.norm(seg))
        if segLen < 1e-6:
            continue
        dirs[i] = seg / segLen

    joinDirs = [None] * count
    for i in range(count):
        prevD = dirs[i - 1] if i - 1 >= 0 else None
        nextD = dirs[i] if i < len(dirs) else None
        if prevD is not None and nextD is not None:
            avg = prevD + nextD
            norm = float(np.linalg.norm(avg))
            if norm > 1e-6:
                joinDirs[i] = avg / norm
            else:
                joinDirs[i] = nextD
        else:
            joinDirs[i] = prevD if prevD is not None else nextD

    halfWidth = widthMeters / 2
    leftRig = []
    rightRig = []
    for i in range(count):
        if joinDirs[i] is None:
            leftRig.append(None)
            rightRig.append(None)
            continue
        dir2d = joinDirs[i]
        normal = np.array([-dir2d[1], dir2d[0]], dtype=float)
        leftPt = np.array([rigPoints[i][0], rigPoints[i][1], rigPoints[i][2]], dtype=float) + np.array([normal[0], normal[1], 0.0]) * halfWidth
        rightPt = np.array([rigPoints[i][0], rigPoints[i][1], rigPoints[i][2]], dtype=float) - np.array([normal[0], normal[1], 0.0]) * halfWidth
        leftRig.append(leftPt)
        rightRig.append(rightPt)

    hasLeft = any(p is not None for p in leftRig)
    hasRight = any(p is not None for p in rightRig)
    leftArray = np.array([p for p in leftRig if p is not None]) if hasLeft else None
    rightArray = np.array([p for p in rightRig if p is not None]) if hasRight else None
    leftProj = projectRigPoints(leftArray, intrinsics, extrinsics, downscale)
    rightProj = projectRigPoints(rightArray, intrinsics, extrinsics, downscale)

    if leftProj is None or rightProj is None:
        return

    # Rebuild projected lists with None where missing
    leftProjected = []
    rightProjected = []
    li = 0
    ri = 0
    for i in range(count):
        if leftRig[i] is None:
            leftProjected.append(None)
        else:
            leftProjected.append(leftProj[li])
            li += 1
        if rightRig[i] is None:
            rightProjected.append(None)
        else:
            rightProjected.append(rightProj[ri])
            ri += 1

    for i in range(count - 1):
        l0 = leftProjected[i]
        l1 = leftProjected[i + 1]
        r0 = rightProjected[i]
        r1 = rightProjected[i + 1]
        if l0 is None and l1 is None:
            continue
        if r0 is None and r1 is None:
            continue
        if l0 is None:
            l0 = l1
        if l1 is None:
            l1 = l0
        if r0 is None:
            r0 = r1
        if r1 is None:
            r1 = r0
        if l0 is None or l1 is None or r0 is None or r1 is None:
            continue

        quad = np.array([
            [int(round(l0[0])), int(round(l0[1]))],
            [int(round(l1[0])), int(round(l1[1]))],
            [int(round(r1[0])), int(round(r1[1]))],
            [int(round(r0[0])), int(round(r0[1]))],
        ], dtype=np.int32)
        cv2.fillConvexPoly(overlay, quad, color, lineType=cv2.LINE_AA)


def drawProjectedCarCuboid(overlay, intrinsics, extrinsics, downscale, carWidth, carLength, rearAxleToCenter):
    if intrinsics is None or extrinsics is None or carWidth is None or carLength is None:
        return

    carHalfWidth = carWidth / 2
    carHalfLength = carLength / 2
    frontX = rearAxleToCenter + carHalfLength
    rearX = rearAxleToCenter - carHalfLength

    rectBaseZ = 0.2
    rectHeight = 0.5
    rectTopZ = rectBaseZ + rectHeight

    carCornersRig = np.array([
        [frontX, carHalfWidth, rectBaseZ],
        [frontX, -carHalfWidth, rectBaseZ],
        [rearX, -carHalfWidth, rectBaseZ],
        [rearX, carHalfWidth, rectBaseZ],
        [frontX, carHalfWidth, rectTopZ],
        [frontX, -carHalfWidth, rectTopZ],
        [rearX, -carHalfWidth, rectTopZ],
        [rearX, carHalfWidth, rectTopZ],
    ], dtype=float)

    projected = projectRigPoints(carCornersRig, intrinsics, extrinsics, downscale)
    if projected is None or len(projected) < 8:
        return
    if any(pt is None for pt in projected):
        return

    height, width = overlay.shape[:2]
    clamped = []
    for pt in projected:
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        clamped.append([x, y])

    base = np.array([clamped[0], clamped[1], clamped[2], clamped[3]], dtype=np.int32)
    top = np.array([clamped[4], clamped[5], clamped[6], clamped[7]], dtype=np.int32)
    left = np.array([clamped[0], clamped[3], clamped[7], clamped[4]], dtype=np.int32)
    right = np.array([clamped[1], clamped[2], clamped[6], clamped[5]], dtype=np.int32)
    front = np.array([clamped[0], clamped[1], clamped[5], clamped[4]], dtype=np.int32)
    back = np.array([clamped[3], clamped[2], clamped[6], clamped[7]], dtype=np.int32)

    for face in (base, top, left, right, front, back):
        cv2.fillConvexPoly(overlay, face, (10, 10, 10, 100), lineType=cv2.LINE_AA)


def toIntPoint(point):
    if point is None:
        return None
    return (int(round(point[0])), int(round(point[1])))


def drawTopDownCanvas(vectors, color, thickness):
    canvas = np.zeros((trajectoryCanvasHeight, trajectoryCanvasWidth, 4), dtype=np.uint8)
    originPoint = (trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness)

    if vectors is None or len(vectors) == 0:
        return canvas

    vectorsNp = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors, dtype=float)
    currentPoint = originPoint

    for vec in vectorsNp:
        endX = int(currentPoint[0] + vec[0] * vecToPixel)
        endY = int(currentPoint[1] - vec[1] * vecToPixel)
        cv2.line(canvas, currentPoint, (endX, endY), (*color, 255), thickness, cv2.LINE_AA)
        currentPoint = (endX, endY)

    return canvas


def drawTopDownRibbon(vectors, color, widthPx):
    canvas = np.zeros((trajectoryCanvasHeight, trajectoryCanvasWidth, 4), dtype=np.uint8)
    originPoint = np.array([trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness], dtype=float)

    if vectors is None or len(vectors) == 0:
        return canvas

    vectorsNp = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else np.asarray(vectors, dtype=float)
    currentPoint = originPoint
    halfWidth = max(1.0, widthPx / 2.0)

    for vec in vectorsNp:
        dx = float(vec[0]) * vecToPixel
        dy = -float(vec[1]) * vecToPixel
        segment = np.array([dx, dy], dtype=float)
        segLen = float(np.linalg.norm(segment))
        if segLen < 1e-6:
            continue

        nextPoint = currentPoint + segment
        normal = np.array([-segment[1], segment[0]], dtype=float) / segLen
        offset = normal * halfWidth

        p0l = currentPoint + offset
        p0r = currentPoint - offset
        p1l = nextPoint + offset
        p1r = nextPoint - offset

        quad = np.array([
            [int(round(p0l[0])), int(round(p0l[1]))],
            [int(round(p1l[0])), int(round(p1l[1]))],
            [int(round(p1r[0])), int(round(p1r[1]))],
            [int(round(p0r[0])), int(round(p0r[1]))],
        ], dtype=np.int32)
        cv2.fillConvexPoly(canvas, quad, (*color, 255), lineType=cv2.LINE_AA)

        currentPoint = nextPoint

    return canvas


def drawTopDownCarWidth(canvas, carWidth, lineLengthMeters):
    if carWidth is None or lineLengthMeters <= 0:
        return

    originPoint = (trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness)
    halfWidthPx = int((carWidth / 2) * vecToPixel) + 2
    endY = int(originPoint[1] - lineLengthMeters * vecToPixel)

    leftX = originPoint[0] - halfWidthPx
    rightX = originPoint[0] + halfWidthPx

    thickness = 1
    cv2.line(canvas, (leftX, originPoint[1]), (leftX, endY), (255, 255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(canvas, (rightX, originPoint[1]), (rightX, endY), (255, 255, 255, 255), thickness, cv2.LINE_AA)


def buildOffsetRigPoints(pointsRig, lateralOffsetMeters):
    if pointsRig is None:
        return None
    offset = np.array([0.0, lateralOffsetMeters, 0.0], dtype=float)
    return pointsRig + offset


def drawTopDownCar(canvas, carWidth, carLength):
    if carWidth is None or carLength is None or carWidth <= 0 or carLength <= 0:
        return

    originPoint = (trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness)
    halfWidthPx = int((carWidth / 2) * vecToPixel)
    halfLengthPx = int((carLength / 2) * vecToPixel)

    leftX = originPoint[0] - halfWidthPx
    rightX = originPoint[0] + halfWidthPx
    topY = originPoint[1] - halfLengthPx
    bottomY = originPoint[1] + halfLengthPx

    cv2.rectangle(
        canvas,
        (leftX, topY),
        (rightX, bottomY),
        (255, 255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visualizeFrame(
    frame,
    predictions,
    groundTruth,
    attnMap,
    currentTelemetry,
    interval,
    vectorTimes,
    intrinsics,
    extrinsics,
    carWidth,
    carLength,
    rearAxleToCenter,
    viewModeLabel,
    overlayAttention=False,
):
    scaledFrame = cv2.resize(frame, (frame.shape[1] // DOWNSCALE, frame.shape[0] // DOWNSCALE))

    trajectoryOverlay = np.zeros((scaledFrame.shape[0], scaledFrame.shape[1], 4), dtype=np.uint8)

    if carWidth is not None:
        gtThickness = max(1, int(round(carWidth * vecToPixel)))
        predThickness = max(1, int(round(carWidth * 0.9 * vecToPixel)))
    else:
        gtThickness = int(vectorThickness * 1.5)
        predThickness = vectorThickness

    gtCanvas = drawTopDownRibbon(groundTruth, grayColor, gtThickness)
    predCanvas = drawTopDownRibbon(predictions, blueColor, predThickness)
    canvasCombined = cv2.add(gtCanvas, predCanvas)

    gtRigPoints = buildRigPointsFromVectors(groundTruth, scaleFactor)
    predRigPoints = buildRigPointsFromVectors(predictions, scaleFactor)

    maxForward = 50.0
    if gtRigPoints is not None:
        maxForward = max(maxForward, float(np.max(gtRigPoints[:, 0])))
    if predRigPoints is not None:
        maxForward = max(maxForward, float(np.max(predRigPoints[:, 0])))
    lineLength = maxForward + 5.0

    if DEBUG and carWidth is not None:
        drawTopDownCarWidth(canvasCombined, carWidth, lineLength)

    drawTopDownCar(canvasCombined, carWidth, carLength)

    if SHOWORIGINALTRAJ:
        topDownScale = 1.0
        canvasResized = cv2.resize(
            canvasCombined,
            (int(canvasCombined.shape[1] * topDownScale), int(canvasCombined.shape[0] * topDownScale)),
        )
        cv2.imshow("Top Down Trajectory Canvas", canvasResized)

    gtProjected = projectRigPoints(gtRigPoints, intrinsics, extrinsics, DOWNSCALE) if gtRigPoints is not None else None
    predProjected = projectRigPoints(predRigPoints, intrinsics, extrinsics, DOWNSCALE) if predRigPoints is not None else None

    if carWidth is not None:
        drawProjectedRibbonFromRig(trajectoryOverlay, gtRigPoints, intrinsics, extrinsics, DOWNSCALE, carWidth, (*grayColor, 178))
        drawProjectedRibbonFromRig(trajectoryOverlay, predRigPoints, intrinsics, extrinsics, DOWNSCALE, carWidth * 0.9, (*blueColor, 178))
    else:
        projectedGtThickness = int(vectorThickness * 1.5)
        projectedPredThickness = vectorThickness
        drawProjectedPolyline(trajectoryOverlay, gtProjected, (*grayColor, 255), projectedGtThickness)
        drawProjectedPolyline(trajectoryOverlay, predProjected, (*blueColor, 255), projectedPredThickness)

    if DEBUG and carWidth is not None and extrinsics is not None and intrinsics is not None:
        leftPointsRig = np.array([[0, -carWidth / 2, 0], [lineLength, -carWidth / 2, 0]], dtype=float)
        rightPointsRig = np.array([[0, carWidth / 2, 0], [lineLength, carWidth / 2, 0]], dtype=float)
        leftProj = projectRigPoints(leftPointsRig, intrinsics, extrinsics, DOWNSCALE)
        rightProj = projectRigPoints(rightPointsRig, intrinsics, extrinsics, DOWNSCALE)
        if leftProj and leftProj[0] is not None and leftProj[1] is not None:
            leftStart = toIntPoint(leftProj[0])
            leftEnd = toIntPoint(leftProj[1])
            if leftStart and leftEnd:
                cv2.line(trajectoryOverlay, leftStart, leftEnd, (255, 255, 255, 255), 2, cv2.LINE_AA)
        
        if rightProj and rightProj[0] is not None and rightProj[1] is not None:
            rightStart = toIntPoint(rightProj[0])
            rightEnd = toIntPoint(rightProj[1])
            if rightStart and rightEnd:
                cv2.line(trajectoryOverlay, rightStart, rightEnd, (255, 255, 255, 255), 2, cv2.LINE_AA)

    if DEBUG:
        drawProjectedCarCuboid(trajectoryOverlay, intrinsics, extrinsics, DOWNSCALE, carWidth, carLength, rearAxleToCenter)

    alpha = trajectoryOverlay[..., 3] / 255.0
    scaledFrame = ((1 - alpha[..., None]) * scaledFrame + alpha[..., None] * trajectoryOverlay[..., :3]).astype(np.uint8)

    if attnMap is not None and overlayAttention:
        attnResized = cv2.resize(attnMap, (scaledFrame.shape[1], scaledFrame.shape[0]), interpolation=cv2.INTER_LINEAR)
        attnNorm = attnResized / (attnResized.max() + 1e-8)
        attnDisplay = (attnNorm ** 0.5)
        attnDisplay = (attnDisplay * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attnDisplay, cv2.COLORMAP_JET)
        # blend onto scaledFrame
        alpha_attn = 0.3
        scaledFrame = cv2.addWeighted(scaledFrame, 1 - alpha_attn, heatmap, alpha_attn, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1 / DOWNSCALE
    fontColor = (255, 255, 255)
    thickness = max(1, 2 // DOWNSCALE)
    textX = 20 // DOWNSCALE
    textYStart = 40 // DOWNSCALE
    lineHeight = 40 // DOWNSCALE

    speedCoords = (textX, textYStart + 3 * lineHeight)

    if viewModeLabel:
        putTextWithOutline(scaledFrame, viewModeLabel, (textX, textYStart), font, fontScale, fontColor, thickness)
        textYStart += lineHeight

    if currentTelemetry:
        timestampVal = currentTelemetry.get('timestamp')
        if timestampVal is not None:
            timeText = f"Time: {timestampVal.strftime('%H:%M:%S.%f')[:-3]}"
            putTextWithOutline(scaledFrame, timeText, (textX, textYStart), font, fontScale, fontColor, thickness)

        speedVal = currentTelemetry.get('speed', 0.0)
        speedKph = speedVal * 3.6
        speedText = f"Speed: {speedVal:.1f} m/s ({speedKph:.1f} km/h)"
        putTextWithOutline(scaledFrame, speedText, (textX, textYStart + lineHeight), font, fontScale, fontColor, thickness)

        accelVal = currentTelemetry.get('acceleration', 0.0)
        accelText = f"Acceleration: {accelVal:.1f} m/s^2"
        putTextWithOutline(scaledFrame, accelText, (textX, textYStart + 2 * lineHeight), font, fontScale, fontColor, thickness)

        turnRateVal = currentTelemetry.get('turnRate', 0.0)
        turnText = f"Turn Rate: {turnRateVal:.0f} deg/s"
        putTextWithOutline(scaledFrame, turnText, (textX, textYStart + 3 * lineHeight), font, fontScale, fontColor, thickness)

    putTextWithOutline(scaledFrame, "Predicted (blue)", (10, scaledFrame.shape[0] - 50), font, fontScale, blueColor, thickness)
    putTextWithOutline(scaledFrame, "Ground Truth (gray)", (10, scaledFrame.shape[0] - 30), font, fontScale, grayColor, thickness)

    if predictions is not None and len(predictions) > 0:
        firstVector = predictions[0]
        firstVectorNp = firstVector.cpu().numpy() if isinstance(firstVector, torch.Tensor) else np.asarray(firstVector)
        distance = math.sqrt(float(firstVectorNp[0]) ** 2 + float(firstVectorNp[1]) ** 2)
        stepSeconds = vectorTimes[0] if vectorTimes else interval
        requestedSpeedMs = distance / max(stepSeconds, 1e-6)
        requestedSpeedKph = requestedSpeedMs * 3.6
        speedText = f"Predicted target speed: {requestedSpeedMs:.1f} m/s ({requestedSpeedKph:.1f} km/h)"
        putTextWithOutline(scaledFrame, speedText, (speedCoords[0] + 550 // DOWNSCALE, speedCoords[1]), font, fontScale, blueColor, thickness)

    if attnMap is not None and SHOWATTENTION and not overlayAttention:
        attnResized = cv2.resize(attnMap, (scaledFrame.shape[1], scaledFrame.shape[0]), interpolation=cv2.INTER_LINEAR)
        attnNorm = attnResized / (attnResized.max() + 1e-8)
        attnDisplay = (attnNorm ** 0.5)
        attnDisplay = (attnDisplay * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attnDisplay, cv2.COLORMAP_JET)
        heatmapSmall = cv2.resize(heatmap, (scaledFrame.shape[1] // 5, scaledFrame.shape[0] // 5))
        cv2.imshow("Attention Map Overlay", heatmapSmall)

    if DEBUG:
        """DEBUG"""
        if predRigPoints is not None and predProjected is not None:
            validPred = sum(1 for p in predProjected if p is not None)
            print(f"Valid projected points: Pred {validPred}/{len(predProjected)}")
        if gtRigPoints is not None and gtProjected is not None:
            validGt = sum(1 for p in gtProjected if p is not None)
            print(f"Valid projected points: GT {validGt}/{len(gtProjected)}")
        """DEBUG"""

    cv2.imshow("Prediction (blue) vs Ground Truth (gray)", scaledFrame)


def runModel(modelPath, videoPath, calibrationRoot, temporalContextTimeWindow=0.1):
    device = torch.device("cuda" if USEGPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paramsPath = modelPath.replace('best_model.pth', 'training_params.json')
    if os.path.exists(paramsPath):
        with open(paramsPath, 'r') as f:
            trainingParams = json.load(f)
        featDim = trainingParams.get('featDim', 512)
        hiddenDim = trainingParams.get('hiddenDim', 1024)
        predSteps = trainingParams.get('predSteps', 12)
        intervalSeconds = trainingParams.get('intervalSeconds', 0.25)
        vectorTimes = trainingParams.get('vectorTimes', None)

        modelName = trainingParams.get('modelName', 'unknown_model')
        availableModelName = TrajectoryModel(featDim=featDim, hiddenDim=hiddenDim, predSteps=predSteps).to(device).name
        if modelName != availableModelName:
            raise ValueError(
                f"Invalid model architecture. Model with architecture '{modelName}' was being loaded with the architecture '{availableModelName}'."
            )

        print(f"Loaded training params from {paramsPath}: featDim={featDim}, hiddenDim={hiddenDim}, predSteps={predSteps}, intervalSeconds={intervalSeconds}")
    else:
        warnings.warn(f"{paramsPath} not found, using defaults (This may cause errors if model architecture mismatches.)\n")
        featDim, hiddenDim, predSteps = 512, 1024, 12
        intervalSeconds = 0.25
        vectorTimes = None

    model = TrajectoryModel(featDim=featDim, hiddenDim=hiddenDim, predSteps=predSteps, intervalSeconds=intervalSeconds, vectorTimes=vectorTimes).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    predSteps = model.outputSpec.get('num_vectors', 12) if hasattr(model, 'outputSpec') else 12
    interval = model.outputSpec.get('intervalSeconds', intervalSeconds) if hasattr(model, 'outputSpec') else intervalSeconds
    if vectorTimes is None:
        vectorTimes = model.vectorTimes
    inputImageSize = model.inputSpec.get('image_size', (270, 480)) if hasattr(model, 'inputSpec') else (270, 480)

    print(f"\nModel specs - Input Size: {inputImageSize} -> Output PredSteps: {predSteps}, Interval: {interval}s\n")

    transform = transforms.Compose([
        transforms.Resize(inputImageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(videoPath):
        print(f"Video file not found: {videoPath}")
        return

    nvidiaInfo = resolveNvidiaEgomotionPaths(videoPath)
    if not nvidiaInfo:
        print("NVIDIA egomotion data not found for this video.")
        return

    egoData = loadEgomotionParquet(nvidiaInfo["egomotionPath"])
    frameTimestamps = loadCameraTimestampsParquet(nvidiaInfo["timestampsPath"])
    if frameTimestamps is None or len(frameTimestamps) == 0:
        print(f"Camera timestamps missing or empty: {nvidiaInfo['timestampsPath']}")
        return

    cameraName = "camera_front_wide_120fov"
    intrinsics, extrinsics, vehicleDims = loadCalibrationData(calibrationRoot, nvidiaInfo["clipUuid"], cameraName)

    if DEBUG:
        """DEBUG"""
        print("Calibration load summary:")
        print(f"  intrinsics loaded: {intrinsics is not None}")
        print(f"  extrinsics loaded: {extrinsics is not None}")
        print(f"  vehicle dims loaded: {vehicleDims is not None}")
        if intrinsics is not None:
            print(f"  intrinsics keys: {sorted(list(intrinsics.keys()))[:6]} ...")
        if extrinsics is not None:
            print(f"  extrinsics keys: {sorted(list(extrinsics.keys()))[:6]} ...")
        if vehicleDims is not None:
            print(f"  vehicle width: {vehicleDims.get('width', None)}")
        """DEBUG"""

    if intrinsics is None or extrinsics is None:
        print("Calibration data missing. Make sure calibrationRoot points to the folder containing camera_intrinsics, sensor_extrinsics, vehicle_dimensions.")
        return

    carWidth = vehicleDims.get('width', 2.0) if vehicleDims is not None else 2.0
    carLength = vehicleDims.get('length', 4.5) if vehicleDims is not None else 4.5
    rearAxleToCenter = vehicleDims.get('rear_axle_to_bbox_center', 0.0) if vehicleDims is not None else 0.0

    cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Error: Could not open video with any backend.")
            return

    ret, testFrame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    timePerFrame = 1.0 / fps

    frameBuffer = []
    frameBufferSize = max(int(fps * temporalContextTimeWindow), 2)
    nextFrameTime = time.time()

    prediction = None
    labels = None
    lastAttnMap = None
    viewMode = 0
    attnIndex = -1

    print()
    with torch.no_grad():
        while cap.isOpened():
            currentTime = time.time()
            if currentTime >= nextFrameTime:
                loopStart = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                currentFrameIndex = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                currentEgoState = None
                currentFrameTimestampUs = None
                currentTelemetry = None

                if currentFrameIndex < len(frameTimestamps):
                    currentFrameTimestampUs = int(frameTimestamps[currentFrameIndex])
                    if currentFrameTimestampUs != np.iinfo(np.int64).min:
                        currentEgoState = interpolateEgomotionState(egoData, currentFrameTimestampUs)

                if currentEgoState is not None:
                    vx = float(currentEgoState['vx'])
                    vy = float(currentEgoState['vy'])
                    speed = math.sqrt(vx * vx + vy * vy)

                    ax = float(currentEgoState['ax'])
                    ay = float(currentEgoState['ay'])
                    acceleration = math.sqrt(ax * ax + ay * ay)

                    curvature = float(currentEgoState['curvature'])
                    turnRate = math.degrees(curvature * speed)

                    currentTelemetry = {
                        'timestamp': safeTimestampFromUs(currentFrameTimestampUs),
                        'speed': speed,
                        'acceleration': acceleration,
                        'turnRate': turnRate,
                    }

                frameBuffer.append(frame)

                prediction = None
                labels = None
                attnMaps = None

                modelTimeMs = 0.0
                trajTimeMs = 0.0
                vizTimeMs = 0.0

                if len(frameBuffer) >= frameBufferSize:
                    currentFrameOrig = frameBuffer[-1]
                    prevFrameOrig = frameBuffer[0]
                    frameBuffer.pop(0)

                    prevImage = Image.fromarray(cv2.cvtColor(prevFrameOrig, cv2.COLOR_BGR2RGB))
                    currentImage = Image.fromarray(cv2.cvtColor(currentFrameOrig, cv2.COLOR_BGR2RGB))
                    prevImgTensor = transform(prevImage).unsqueeze(0).to(device)
                    currentImgTensor = transform(currentImage).unsqueeze(0).to(device)

                    modelStart = time.perf_counter()
                    prediction, _, attnMaps = model(currentImgTensor, prevImgTensor)
                    modelTimeMs = (time.perf_counter() - modelStart) * 1000.0
                    if attnMaps:
                        if attnIndex == -1:
                            lastAttnMap = attnMaps[-1].squeeze().cpu().numpy()
                        else:
                            lastAttnMap = attnMaps[attnIndex].squeeze().cpu().numpy()
                    else:
                        lastAttnMap = None

                if currentFrameTimestampUs is not None and currentEgoState is not None:
                    trajStart = time.perf_counter()
                    if vectorTimes:
                        targetTimesUs = [currentFrameTimestampUs + int(offset * 1e6) for offset in vectorTimes]
                    else:
                        intervalUs = int(interval * 1e6)
                        targetTimesUs = [currentFrameTimestampUs + intervalUs * (i + 1) for i in range(predSteps)]
                    futureTrajectory = calculateFutureTrajectoryEgomotion(egoData, currentFrameTimestampUs, targetTimesUs)

                    validVectors = [v for v in futureTrajectory if v is not None]
                    if len(validVectors) >= predSteps:
                        labelsList = [[v['x'], v['y']] for v in validVectors]
                        labels = torch.tensor(labelsList[:predSteps], dtype=torch.float32)
                    trajTimeMs = (time.perf_counter() - trajStart) * 1000.0

                displayFrame = frame
                if viewMode == 1 or viewMode == 2:
                    modelViewSmall = cv2.resize(frame, (inputImageSize[1], inputImageSize[0]))
                    displayFrame = cv2.resize(modelViewSmall, (frame.shape[1], frame.shape[0]))

                vizStart = time.perf_counter()
                overlayAttention = (viewMode == 2)
                viewModeLabel = "Original View" if viewMode == 0 else "Model View" if viewMode == 1 else f"Internal Model View{' (All)' if attnIndex == -1 else f' (Step {attnIndex+1})'}"
                visualizeFrame(
                    displayFrame,
                    prediction.squeeze() if prediction is not None else None,
                    labels,
                    lastAttnMap,
                    currentTelemetry,
                    interval,
                    vectorTimes,
                    intrinsics,
                    extrinsics,
                    carWidth,
                    carLength,
                    rearAxleToCenter,
                    viewModeLabel,
                    overlayAttention,
                )
                vizTimeMs = (time.perf_counter() - vizStart) * 1000.0

                totalMs = (time.perf_counter() - loopStart) * 1000.0
                otherMs = max(0.0, totalMs - modelTimeMs - trajTimeMs - vizTimeMs)
                fpsText = 1000.0 / totalMs if totalMs > 1e-6 else 0.0
                print(
                    f"\rmodel:{modelTimeMs:5.1f}ms - traj:{trajTimeMs:5.1f}ms - viz:{vizTimeMs:5.1f}ms - other:{otherMs:5.1f}ms | total:{totalMs:6.1f}ms - fps:{fpsText:5.1f}",
                    end="",
                    flush=True,
                )

                nextFrameTime += timePerFrame

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('v'):
                viewMode = (viewMode + 1) % 3
            if viewMode == 2:
                if key & 0xFF == ord('j'):  # left (previous attention)
                    attnIndex = max(-1, attnIndex - 1)
                    # print(f"Left (q): attnIndex = {attnIndex}")
                elif key & 0xFF == ord('l'):  # right (next attention)
                    maxIndex = len(attnMaps) - 1 if attnMaps else -1
                    attnIndex = min(maxIndex, attnIndex + 1)
                    # print(f"Right (d): attnIndex = {attnIndex}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    DOWNSCALE = 2
    SHOWATTENTION = True
    SHOWORIGINALTRAJ = True
    USEGPU = False
    DEBUG = False

    modelPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\training\run20\best_model.pth"
    videoPath = r"C:\Users\Aypisam\Videos\Autopilot_Videos\Camera\camera_front_wide_120fov"

    calibrationRoot = r"C:\Users\Aypisam\Videos\Autopilot_Videos\calibration"

    # if video path is a list, run on each video
    if isinstance(videoPath, list):
        for vidPath in videoPath:
            runModel(modelPath, vidPath, calibrationRoot)
    else:
        # if path is folder, run on each mp4 file in the folder
        if os.path.isdir(videoPath):
            for fileName in os.listdir(videoPath):
                if fileName.lower().endswith('.mp4'):
                    vidPath = os.path.join(videoPath, fileName)
                    runModel(modelPath, vidPath, calibrationRoot)
        else:
            
            runModel(modelPath, videoPath, calibrationRoot)
