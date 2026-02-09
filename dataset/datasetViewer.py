import cv2
import os
import random
import numpy as np
import glob
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
import time

grayColor = (128, 128, 128)
vecToPixel = 10
vectorThickness = 10
trajectoryCanvasWidth = 500
trajectoryCanvasHeight = 750
jointThicknessScale = 0.02
jointThicknessMax = 3
jointAlpha = 100

def parseVectorLine(line, timeList=None):
    """Parses the vector line and returns a list of dictionaries."""
    vectors = []
    try:
        # Extract the part after 'vectors : '
        vector_data = line.split(' : ')[1].strip()
        # Split into individual vector strings
        vector_strings = vector_data.split(' ')
        time_increment = 3.0 / len(vector_strings) if vector_strings else 0

        for i, vec_str in enumerate(vector_strings):
            x_str, y_str = vec_str.split(',')
            if timeList and i < len(timeList):
                timeValue = float(timeList[i])
            else:
                timeValue = (i + 1) * time_increment
            vectors.append({
                'x': float(x_str),
                'y': float(y_str),
                'time': timeValue
            })
    except (IndexError, ValueError) as e:
        print(f"Error parsing vector line: {line.strip()} - {e}")
    return vectors


def parseVectorTimes(line):
    """Parses the vectorTimes line and returns a list of floats."""
    try:
        times_str = line.split(' : ')[1].strip()
        if times_str == "None":
            return []
        return [float(t) for t in times_str.split(' ') if t]
    except (IndexError, ValueError) as e:
        print(f"Error parsing vectorTimes line: {line.strip()} - {e}")
        return []

def parseSpeed(line):
    """Parses the speed line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing speed line: {line.strip()} - {e}")
        return 0.0

def parseAcceleration(line):
    """Parses the acceleration line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing acceleration line: {line.strip()} - {e}")
        return 0.0

def parseTurnRate(line):
    """Parses the turn rate line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing turn rate line: {line.strip()} - {e}")
        return 0.0

def getVideoFrame(videoPath, frameIndex):
    cap = cv2.VideoCapture(videoPath, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"Could not open video: {videoPath}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read frame {frameIndex} from {videoPath}")
        return None
    return frame

def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    """Draws text with a black outline."""
    # Draw the outline in black
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Draw the main text in the specified color
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def drawImageSpaceRibbon(overlay, vectors, originPoint, widthPx, color):
    if not vectors:
        return

    currentPoint = np.array(originPoint, dtype=float)
    halfWidth = max(1.0, widthPx / 2.0)
    joints = []

    for vector in vectors:
        dx = float(vector['x']) * vecToPixel
        dy = -float(vector['y']) * vecToPixel
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
        cv2.fillConvexPoly(overlay, quad, (*color, 178), lineType=cv2.LINE_AA)
        joints.append((p1l, p1r))

        currentPoint = nextPoint

    jointThickness = max(1, min(jointThicknessMax, int(round(widthPx * jointThicknessScale))))
    for p1l, p1r in joints:
        jointStart = (int(round(p1l[0])), int(round(p1l[1])))
        jointEnd = (int(round(p1r[0])), int(round(p1r[1])))
        cv2.line(overlay, jointStart, jointEnd, (0, 0, 0, jointAlpha), jointThickness, cv2.LINE_AA)

def drawTopDownRibbonFromVectors(vectors, color, widthPx):
    canvas = np.zeros((trajectoryCanvasHeight, trajectoryCanvasWidth, 4), dtype=np.uint8)
    originPoint = np.array([trajectoryCanvasWidth // 2, trajectoryCanvasHeight - vectorThickness], dtype=float)

    if not vectors:
        return canvas

    currentPoint = originPoint
    halfWidth = max(1.0, widthPx / 2.0)
    joints = []

    for vector in vectors:
        dx = float(vector['x']) * vecToPixel
        dy = -float(vector['y']) * vecToPixel
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
        joints.append((p1l, p1r))

        currentPoint = nextPoint

    jointThickness = max(1, min(jointThicknessMax, int(round(widthPx * jointThicknessScale))))
    for p1l, p1r in joints:
        jointStart = (int(round(p1l[0])), int(round(p1l[1])))
        jointEnd = (int(round(p1r[0])), int(round(p1r[1])))
        cv2.line(canvas, jointStart, jointEnd, (0, 0, 0, 255), jointThickness, cv2.LINE_AA)

    return canvas

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
    vectorsNp = np.asarray(vectors, dtype=float)

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

    jointPairs = []

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

        if l1 is not None and r1 is not None:
            jointPairs.append((l1, r1))

    for l1, r1 in jointPairs:
        jointStart = (int(round(l1[0])), int(round(l1[1])))
        jointEnd = (int(round(r1[0])), int(round(r1[1])))
        pixelWidth = float(np.linalg.norm(np.array(l1) - np.array(r1)))
        jointThickness = max(1, int(round(pixelWidth * jointThicknessScale)))
        cv2.line(overlay, jointStart, jointEnd, (0, 0, 0, jointAlpha), jointThickness, cv2.LINE_AA)

def viewRandomItem(datasetDir, calibrationRoot=None, cameraName="camera_front_wide_120fov"):
    """
    Loads a random item from the dataset and visualizes the trajectory.

    Args:
        datasetDir: The root directory of the dataset.
    """
    imagesDir = os.path.join(datasetDir, "images")
    labelsDir = os.path.join(datasetDir, "labels")

    imagesize = tuple(datasetDir.split('framesize')[1].split('x'))
    if "(" in imagesize[1]:
        imagesize = imagesize[0], imagesize[1].split('(')[0]
    
    imagesize = int(imagesize[0]), int(imagesize[1])
    print("imagesize: ", imagesize)

    if not os.path.exists(labelsDir):
        print(f"Labels directory not found at: {labelsDir}")
        return

    print(f"Loading dataset from: {datasetDir}...")

    labelFiles = [f for f in os.listdir(labelsDir) if f.endswith(".txt")]
    if not labelFiles:
        print("No label files found in the directory.")
        return
    
    print(f"Total datapoints: {len(labelFiles)}")
    
    labelFiles.sort()
    currentIndex = random.randint(0, len(labelFiles) - 1)

    calibrationCache = {}

    while True:
        try:
            labelFile = labelFiles[currentIndex]
            baseName = os.path.splitext(labelFile)[0]
            metadataPath = os.path.join(labelsDir, labelFile)

            prevImagePath = os.path.join(imagesDir, f"{baseName}_prev.png")
            currentImagePath = os.path.join(imagesDir, f"{baseName}_current.png")

            # Read metadata
            vectors = []
            vectorTimes = []
            speed, acceleration, turnRate = 0, 0, 0
            videoPath = None
            prevFrameIndex = None
            frameIndex = None
            with open(metadataPath, 'r') as f:
                for line in f:
                    if line.startswith('vectors'):
                        vectors = parseVectorLine(line, vectorTimes if vectorTimes else None)
                    elif line.startswith('vectorTimes'):
                        vectorTimes = parseVectorTimes(line)
                        if vectors:
                            for i, timeValue in enumerate(vectorTimes[:len(vectors)]):
                                vectors[i]['time'] = timeValue
                    elif line.startswith('speed'):
                        speed = parseSpeed(line)
                    elif line.startswith('acceleration'):
                        acceleration = parseAcceleration(line)
                    elif line.startswith('turnRate'):
                        turnRate = parseTurnRate(line)
                    elif line.startswith('video'):
                        videoPath = line.split(' : ', 1)[1].strip()
                    elif line.startswith('prevFrameIndex'):
                        prevFrameIndex = int(line.split(' : ')[1].strip())
                    elif line.startswith('frameIndex'):
                        frameIndex = int(line.split(' : ')[1].strip())

            # Load images
            downscale = 1.0
            if videoPath and prevFrameIndex is not None and frameIndex is not None:
                prevRaw = getVideoFrame(videoPath, prevFrameIndex)
                currentRaw = getVideoFrame(videoPath, frameIndex)
                if prevRaw is not None and currentRaw is not None and imagesize[0] > 0:
                    downscale = prevRaw.shape[1] / float(imagesize[0])
                prevFrame = cv2.resize(prevRaw, imagesize) if prevRaw is not None else None
                currentFrame = cv2.resize(currentRaw, imagesize) if currentRaw is not None else None
            else:
                prevFrame = cv2.imread(prevImagePath)
                currentFrame = cv2.imread(currentImagePath)

            if prevFrame is None or currentFrame is None:
                print(f"Could not load images for {baseName}")
                currentIndex = (currentIndex + 1) % len(labelFiles)
                continue

            putTextWithOutline(prevFrame, "Previous Frame", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # --- Visualization ---
            visFrame = currentFrame.copy()
            putTextWithOutline(visFrame, "Current Frame", (225, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            originPoint = (visFrame.shape[1] // 2, visFrame.shape[0] - 5)

            trajectoryOverlay = np.zeros((visFrame.shape[0], visFrame.shape[1], 4), dtype=np.uint8)
            intrinsics = None
            extrinsics = None
            vehicleDims = None
            if calibrationRoot and videoPath:
                nvidiaInfo = resolveNvidiaEgomotionPaths(videoPath)
                if nvidiaInfo:
                    clipUuid = nvidiaInfo["clipUuid"]
                    if clipUuid not in calibrationCache:
                        calibrationCache[clipUuid] = loadCalibrationData(calibrationRoot, clipUuid, cameraName)
                    intrinsics, extrinsics, vehicleDims = calibrationCache[clipUuid]

            if intrinsics is not None and extrinsics is not None and vectors:
                vectorsArray = [[v['x'], v['y']] for v in vectors]
                rigPoints = buildRigPointsFromVectors(vectorsArray, 1.0)
                carWidth = vehicleDims.get('width', 2.0) if vehicleDims else None
                if carWidth:
                    drawProjectedRibbonFromRig(trajectoryOverlay, rigPoints, intrinsics, extrinsics, downscale, carWidth, (*grayColor, 178))
                else:
                    projected = projectRigPoints(rigPoints, intrinsics, extrinsics, downscale)
                    drawProjectedPolyline(trajectoryOverlay, projected, (*grayColor, 255), vectorThickness)
            else:
                drawImageSpaceRibbon(trajectoryOverlay, vectors, originPoint, vectorThickness, grayColor)

            alpha = trajectoryOverlay[..., 3] / 255.0
            visFrame = ((1 - alpha[..., None]) * visFrame + alpha[..., None] * trajectoryOverlay[..., :3]).astype(np.uint8)

            # Display info
            speedText = f"Speed: {speed:.2f} m/s"
            accelerationText = f"Acceleration: {acceleration:.2f} m.s-2"
            turnRateText = f"Turn Rate: {turnRate:.2f} deg/s"
            infoTexts = [speedText, accelerationText, turnRateText]
            for i, text in enumerate(infoTexts):
                putTextWithOutline(visFrame, text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Concatenate previous and current frames for display
            combinedDisplay = np.hstack([prevFrame, visFrame])

            cv2.imshow("Dataset Viewer - Press 'q' to quit, any other key for next", combinedDisplay)

            topDownCanvas = drawTopDownRibbonFromVectors(vectors, grayColor, vectorThickness)
            topDownCarWidth = vehicleDims.get('width', None) if vehicleDims else None
            topDownCarLength = vehicleDims.get('length', None) if vehicleDims else None
            drawTopDownCar(topDownCanvas, topDownCarWidth, topDownCarLength)
            topDownScale = 0.6
            topDownResized = cv2.resize(
                topDownCanvas,
                (int(topDownCanvas.shape[1] * topDownScale), int(topDownCanvas.shape[0] * topDownScale)),
            )
            cv2.imshow("Top Down Trajectory", topDownResized)

            loadTime = time.time()
            key = -1
            while True:
                pressed = cv2.waitKeyEx(10)
                if pressed == -1:
                    continue
                # Ignore keys that arrive immediately after load to avoid lockups.
                if time.time() - loadTime < 0.15:
                    continue
                key = pressed
                break

            if key == ord('q'):
                break
            if key == 2424832:  # Left arrow
                currentIndex = (currentIndex - 1 + len(labelFiles)) % len(labelFiles)
            elif key == 2555904:  # Right arrow
                currentIndex = (currentIndex + 1) % len(labelFiles)
            else: # Any other key for random
                currentIndex = random.randint(0, len(labelFiles) - 1)
        except Exception as exc:
            print(f"Viewer error for {labelFiles[currentIndex]}: {exc}")
            currentIndex = (currentIndex + 1) % len(labelFiles)
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    datasetOutputDir = r"F:\Projects\Autopilot\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360(1)"
    calibrationRoot = r"F:\Projects\Autopilot\nvidia_dataset\calibration"
    viewRandomItem(datasetOutputDir, calibrationRoot=calibrationRoot)
