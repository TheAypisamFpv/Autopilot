import os
import json
import time
import math
import warnings
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import runModel as runModelModule
from runModel import (
    resolveNvidiaEgomotionPaths,
    loadCalibrationData,
    normalizeVectorTimes,
    kinematicStatesToDisplacementVectors,
    safeTimestampFromUs,
    visualizeFrame,
)
from model.CreateModel import TrajectoryModel
from dataset.generator import (
    loadEgomotionParquet,
    loadCameraTimestampsParquet,
    interpolateEgomotionState,
    calculateFutureTrajectoryEgomotion,
)


imageMean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imageStd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def totalVariationLoss(imageTensor):
    diffX = torch.mean(torch.abs(imageTensor[:, :, :, 1:] - imageTensor[:, :, :, :-1]))
    diffY = torch.mean(torch.abs(imageTensor[:, :, 1:, :] - imageTensor[:, :, :-1, :]))
    return diffX + diffY


def denormalizeToBgr(imageTensor, outputSize):
    imageTensor = imageTensor.detach().cpu()
    imageTensor = imageTensor * imageStd + imageMean
    imageTensor = imageTensor.clamp(0.0, 1.0)
    imageNp = imageTensor.squeeze(0).permute(1, 2, 0).numpy()
    imageNp = (imageNp * 255.0).astype(np.uint8)
    imageBgr = cv2.cvtColor(imageNp, cv2.COLOR_RGB2BGR)
    if outputSize is not None:
        imageBgr = cv2.resize(imageBgr, outputSize)
    return imageBgr


def invertFrameFromTrajectory(
    model,
    targetTraj,
    prevSynthTensor,
    egoHistoryInput,
    outputsKinematicStates,
    vectorTimes,
    device,
    inputImageSize,
    steps,
    learningRate,
    l2Weight,
    tvWeight,
    temporalWeight,
    initTensor=None,
):
    height, width = inputImageSize
    if initTensor is None:
        currentTensor = torch.randn((1, 3, height, width), device=device)
    else:
        currentTensor = initTensor.detach().clone().to(device)

    if prevSynthTensor is None:
        prevSynthTensor = currentTensor.detach().clone()
    else:
        prevSynthTensor = prevSynthTensor.detach().clone().to(device)

    currentTensor.requires_grad_(True)

    optimizer = torch.optim.Adam([currentTensor], lr=learningRate)
    lossFn = torch.nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        preds, _, _ = model(currentTensor, prevSynthTensor, egoHistory=egoHistoryInput)
        predictionForLoss = preds
        if outputsKinematicStates:
            predictionForLoss = kinematicStatesToDisplacementVectors(preds, vectorTimes)
        loss = lossFn(predictionForLoss.squeeze(0), targetTraj)

        if l2Weight > 0.0:
            loss = loss + l2Weight * torch.mean(currentTensor ** 2)
        if tvWeight > 0.0:
            loss = loss + tvWeight * totalVariationLoss(currentTensor)
        if temporalWeight > 0.0:
            loss = loss + temporalWeight * torch.mean((currentTensor - prevSynthTensor) ** 2)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            currentTensor.clamp_(-3.0, 3.0)

    with torch.no_grad():
        preds, _, attnMaps = model(currentTensor, prevSynthTensor, egoHistory=egoHistoryInput)

    lastAttnMap = None
    if attnMaps:
        lastAttnMap = attnMaps[-1].squeeze().cpu().numpy()

    return currentTensor.detach(), preds.detach(), lastAttnMap


def runInvertedModel(
    modelPath,
    videoPath,
    calibrationRoot,
    temporalContextTimeWindow=0.1,
    inversionSteps=20,
    inversionLearningRate=0.05,
    l2Weight=1e-4,
    tvWeight=1e-4,
    temporalWeight=1e-3,
    useRealInit=True,
):
    device = torch.device("cuda" if USEGPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paramsPath = modelPath.replace('best_model.pth', 'training_params.json')
    if os.path.exists(paramsPath):
        with open(paramsPath, 'r') as f:
            trainingParams = json.load(f)
        featDim = trainingParams.get('featDim', 512)
        hiddenDim = trainingParams.get('hiddenDim', 1024)
        baseChannels = trainingParams.get('baseChannels', 32)
        numHeads = trainingParams.get('numHeads', 4)
        numLayers = trainingParams.get('numLayers', 2)
        predSteps = trainingParams.get('predSteps', 12)
        intervalSeconds = trainingParams.get('intervalSeconds', 0.25)
        vectorTimes = trainingParams.get('vectorTimes', None)

        modelName = trainingParams.get('modelName', 'unknown_model')
        availableModelName = TrajectoryModel(
            featDim=featDim,
            hiddenDim=hiddenDim,
            predSteps=predSteps,
            intervalSeconds=intervalSeconds,
            vectorTimes=vectorTimes,
            baseChannels=baseChannels,
            numHeads=numHeads,
            numLayers=numLayers,
        ).to(device).name
        if modelName and modelName != availableModelName:
            warnings.warn(
                f"Model name mismatch: checkpoint declares '{modelName}', local code reports '{availableModelName}'. "
                "Proceeding with local architecture and loading compatible keys."
            )

        print(
            f"Loaded training params from {paramsPath}: featDim={featDim}, hiddenDim={hiddenDim}, baseChannels={baseChannels}, numHeads={numHeads}, numLayers={numLayers}, predSteps={predSteps}, intervalSeconds={intervalSeconds}"
        )
    else:
        warnings.warn(f"{paramsPath} not found, using defaults (This may cause errors if model architecture mismatches.)\n")
        featDim, hiddenDim, predSteps = 512, 1024, 12
        baseChannels, numHeads, numLayers = 32, 4, 2
        intervalSeconds = 0.25
        vectorTimes = None

    model = TrajectoryModel(
        featDim=featDim,
        hiddenDim=hiddenDim,
        predSteps=predSteps,
        intervalSeconds=intervalSeconds,
        vectorTimes=vectorTimes,
        baseChannels=baseChannels,
        numHeads=numHeads,
        numLayers=numLayers,
    ).to(device)
    stateDict = torch.load(modelPath, map_location=device)
    loadResult = model.load_state_dict(stateDict, strict=False)
    if loadResult.missing_keys:
        warnings.warn(f"Missing checkpoint keys: {loadResult.missing_keys[:8]}{' ...' if len(loadResult.missing_keys) > 8 else ''}")
    if loadResult.unexpected_keys:
        warnings.warn(f"Unexpected checkpoint keys: {loadResult.unexpected_keys[:8]}{' ...' if len(loadResult.unexpected_keys) > 8 else ''}")
    model.eval()

    modelRuntimeName = getattr(model, "name", "")
    outputsKinematicStates = (
        (isinstance(modelName, str) and "kinematic" in modelName.lower())
        or (isinstance(modelRuntimeName, str) and "kinematic" in modelRuntimeName.lower())
    )

    predSteps = model.outputSpec.get('num_vectors', 12) if hasattr(model, 'outputSpec') else 12
    interval = model.outputSpec.get('intervalSeconds', intervalSeconds) if hasattr(model, 'outputSpec') else intervalSeconds
    vectorTimes = normalizeVectorTimes(
        vectorTimes if vectorTimes is not None else getattr(model, "vectorTimes", None),
        predSteps,
        fallbackInterval=interval,
    )
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

    # Detect GoPro video
    isGopro = "GOPR" in os.path.basename(videoPath) or "gopro" in videoPath.lower()
    cameraName = "gopro" if isGopro else "camera_front_wide_120fov"
    usegroundtruth = not isGopro and USEGROUNDTRUTH  # For GoPro, don't use ground truth

    if usegroundtruth:
        nvidiaInfo = resolveNvidiaEgomotionPaths(videoPath)
        if not nvidiaInfo:
            print("NVIDIA egomotion data not found for this video.")
            return

        egoData = loadEgomotionParquet(nvidiaInfo["egomotionPath"])
        frameTimestamps = loadCameraTimestampsParquet(nvidiaInfo["timestampsPath"])
        if frameTimestamps is None or len(frameTimestamps) == 0:
            print(f"Camera timestamps missing or empty: {nvidiaInfo['timestampsPath']}")
            return
    else:
        nvidiaInfo = None
        egoData = None
        frameTimestamps = None

    if frameTimestamps is None:
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Cannot open video file.")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # Create dummy timestamps starting from now
        start_time = datetime.now()
        frameTimestamps = [int((start_time + timedelta(seconds=i / fps)).timestamp() * 1e6) for i in range(frame_count)]

    intrinsics, extrinsics, vehicleDims = loadCalibrationData(calibrationRoot, nvidiaInfo["clipUuid"] if nvidiaInfo else "dummy", cameraName)

    if intrinsics is None or extrinsics is None:
        print("Calibration data missing. Make sure calibrationRoot points to the folder containing camera_intrinsics, sensor_extrinsics, vehicle_dimensions.")
        return

    carWidth = vehicleDims.get('width', 2.0) if vehicleDims is not None else 2.0
    trackWidth = vehicleDims.get('track_width', carWidth) if vehicleDims is not None else carWidth
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

    lastSynthTensor = None
    viewMode = 0
    egoHistoryBuffer = deque(maxlen=8)
    startupEgoHistoryLength = 8
    requireEgoWarmup = outputsKinematicStates and usegroundtruth
    

    print()
    cv2.startWindowThread()
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
                    if egoData is not None:
                        currentEgoState = interpolateEgomotionState(egoData, currentFrameTimestampUs)

            if currentEgoState is not None:
                vx = float(currentEgoState['vx'])
                vy = float(currentEgoState['vy'])
                speed = math.sqrt(vx * vx + vy * vy)

                ax = float(currentEgoState['ax'])
                ay = float(currentEgoState['ay'])
                acceleration = math.sqrt(ax * ax + ay * ay)

                curvature = float(currentEgoState['curvature'])
                yawRateRad = curvature * speed
                turnRate = math.degrees(curvature * speed)

                egoHistoryBuffer.append(np.array([speed, acceleration, yawRateRad], dtype=np.float32))

                currentTelemetry = {
                    'timestamp': safeTimestampFromUs(currentFrameTimestampUs),
                    'speed': speed,
                    'acceleration': acceleration,
                    'turnRate': turnRate,
                }

            frameBuffer.append(frame)

            egoWarmupReady = (not requireEgoWarmup) or (len(egoHistoryBuffer) >= startupEgoHistoryLength)

            labels = None
            if currentFrameTimestampUs is not None and currentEgoState is not None:
                if vectorTimes:
                    targetTimesUs = [currentFrameTimestampUs + int(offset * 1e6) for offset in vectorTimes]
                else:
                    intervalUs = int(interval * 1e6)
                    targetTimesUs = [currentFrameTimestampUs + intervalUs * (i + 1) for i in range(predSteps)]
                futureTrajectory = calculateFutureTrajectoryEgomotion(egoData, currentFrameTimestampUs, targetTimesUs)

                validVectors = [v for v in futureTrajectory if v is not None]
                if len(validVectors) >= predSteps:
                    labelsList = [[v['x'], v['y']] for v in validVectors]
                    labels = torch.tensor(labelsList[:predSteps], dtype=torch.float32, device=device)

            synthFrame = None
            synthPrediction = None
            synthAttnMap = None

            modelTimeMs = 0.0
            vizTimeMs = 0.0

            if len(frameBuffer) >= frameBufferSize and egoWarmupReady:
                prevFrameOrig = frameBuffer[0]
                currentFrameOrig = frameBuffer[-1]
                frameBuffer.pop(0)
                
                #dropping the resolution of the input frames by 1/10
                inputFrameDownscale = 10
                blurKernel = 99
                
                prevFrameOrig = cv2.resize(
                    cv2.resize(prevFrameOrig,(prevFrameOrig.shape[1] // inputFrameDownscale, prevFrameOrig.shape[0] // inputFrameDownscale)),
                    (prevFrameOrig.shape[1] * inputFrameDownscale, prevFrameOrig.shape[0] * inputFrameDownscale)
                )
                prevFrameOrig = cv2.GaussianBlur(prevFrameOrig, (blurKernel, blurKernel), 0)
            
                currentFrameOrig = cv2.resize(
                    cv2.resize(currentFrameOrig,(currentFrameOrig.shape[1] // inputFrameDownscale, currentFrameOrig.shape[0] // inputFrameDownscale)),
                    (currentFrameOrig.shape[1] * inputFrameDownscale, currentFrameOrig.shape[0] * inputFrameDownscale)
                )
                currentFrameOrig = cv2.GaussianBlur(currentFrameOrig, (blurKernel, blurKernel), 0)

                initTensor = None
                prevInitTensor = None
                if useRealInit:
                    initImage = Image.fromarray(cv2.cvtColor(currentFrameOrig, cv2.COLOR_BGR2RGB))
                    initTensor = transform(initImage).unsqueeze(0).to(device)
                    prevInitImage = Image.fromarray(cv2.cvtColor(prevFrameOrig, cv2.COLOR_BGR2RGB))
                    prevInitTensor = transform(prevInitImage).unsqueeze(0).to(device)

                prevInputTensor = lastSynthTensor
                if prevInputTensor is None and prevInitTensor is not None:
                    prevInputTensor = prevInitTensor

                if egoHistoryBuffer:
                    historyValues = list(egoHistoryBuffer)
                    if len(historyValues) < 8:
                        padValue = historyValues[0]
                        historyValues = [padValue.copy() for _ in range(8 - len(historyValues))] + historyValues
                    else:
                        historyValues = historyValues[-8:]
                    egoHistoryInput = torch.tensor(np.stack(historyValues), dtype=prevInputTensor.dtype if prevInputTensor is not None else torch.float32, device=device).unsqueeze(0)
                else:
                    egoHistoryInput = torch.zeros((1, 8, 3), dtype=prevInputTensor.dtype if prevInputTensor is not None else torch.float32, device=device)

                if labels is not None:
                    print("\nInverting frame from trajectory...", end="", flush=True)
                    modelStart = time.perf_counter()
                    lastSynthTensor, synthPrediction, synthAttnMap = invertFrameFromTrajectory(
                        model,
                        labels,
                        prevInputTensor,
                        egoHistoryInput,
                        outputsKinematicStates,
                        vectorTimes,
                        device,
                        inputImageSize,
                        inversionSteps,
                        inversionLearningRate,
                        l2Weight,
                        tvWeight,
                        temporalWeight,
                        initTensor=initTensor,
                    )
                    modelTimeMs = (time.perf_counter() - modelStart) * 1000.0
                    print(f" done ({modelTimeMs:.1f} ms)")
                    synthFrame = denormalizeToBgr(lastSynthTensor, (frame.shape[1], frame.shape[0]))
                    prevDisplayTensor = prevInputTensor if prevInputTensor is not None else lastSynthTensor
                    prevSynthFrame = denormalizeToBgr(prevDisplayTensor, (frame.shape[1], frame.shape[0]))
                    if DOWNSCALE > 1:
                        synthFrame = cv2.resize(
                            synthFrame,
                            (synthFrame.shape[1] // DOWNSCALE, synthFrame.shape[0] // DOWNSCALE)
                        )
                        prevSynthFrame = cv2.resize(
                            prevSynthFrame,
                            (prevSynthFrame.shape[1] // DOWNSCALE, prevSynthFrame.shape[0] // DOWNSCALE)
                        )
                    cv2.imshow("Predicted Prev Frame", prevSynthFrame)
                    cv2.imshow("Predicted Current Frame", synthFrame)

            if frame is not None:
                realFrame = frame
                if DOWNSCALE > 1:
                    realFrame = cv2.resize(
                        frame,
                        (frame.shape[1] // DOWNSCALE, frame.shape[0] // DOWNSCALE)
                    )
                cv2.imshow("Real Video", realFrame)

            if synthFrame is not None and synthPrediction is not None:
                vizStart = time.perf_counter()
                overlayAttention = (viewMode == 2)
                viewModeLabel = "Original View" if viewMode == 0 else "Model View" if viewMode == 1 else "Internal Model View"
                synthPredictionVectors = (
                    kinematicStatesToDisplacementVectors(synthPrediction, vectorTimes)
                    if outputsKinematicStates
                    else synthPrediction
                )
                predictedSpeedTextValueMs = None
                if outputsKinematicStates and synthPrediction is not None and synthPrediction.numel() > 0:
                    predictedSpeedTextValueMs = float(synthPrediction[0, 0, 0].detach().cpu().item())
                visualizeFrame(
                    synthFrame,
                    synthPredictionVectors.squeeze(0),
                    labels.detach().cpu() if labels is not None else None,
                    synthAttnMap,
                    None,
                    currentTelemetry,
                    interval,
                    vectorTimes,
                    intrinsics,
                    extrinsics,
                    carWidth,
                    trackWidth,
                    carLength,
                    rearAxleToCenter,
                    viewModeLabel,
                    predictedSpeedMs=predictedSpeedTextValueMs,
                    overlayAttention=overlayAttention,
                )
                vizTimeMs = (time.perf_counter() - vizStart) * 1000.0

            totalMs = (time.perf_counter() - loopStart) * 1000.0
            fpsText = 1000.0 / totalMs if totalMs > 1e-6 else 0.0
            if len(frameBuffer) >= frameBufferSize and egoWarmupReady:
                print(
                    f"\rmodel:{modelTimeMs:5.1f}ms - viz:{vizTimeMs:5.1f}ms | total:{totalMs:6.1f}ms - fps:{fpsText:5.1f}",
                    end="",
                    flush=True,
                )
            elif requireEgoWarmup:
                print(
                    f"\rWarming up ego history {len(egoHistoryBuffer)}/{startupEgoHistoryLength}...",
                    end="",
                    flush=True,
                )

            nextFrameTime += timePerFrame

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('v'):
            viewMode = (viewMode + 1) % 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    DOWNSCALE = 2
    SHOWATTENTION = True
    SHOWORIGINALTRAJ = True
    USEGPU = False
    DEBUG = False
    USEGROUNDTRUTH = True

    runModelModule.DOWNSCALE = DOWNSCALE
    runModelModule.SHOWATTENTION = SHOWATTENTION
    runModelModule.SHOWORIGINALTRAJ = SHOWORIGINALTRAJ
    runModelModule.USEGPU = USEGPU
    runModelModule.DEBUG = DEBUG
    runModelModule.USEGROUNDTRUTH = USEGROUNDTRUTH

    modelPath = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\training\run21\best_model.pth"
    videoPath = r"C:\Users\Aypisam\Videos\Autopilot_Videos\Camera\camera_front_wide_120fov"
    calibrationRoot = r"C:\Users\Aypisam\Videos\Autopilot_Videos\calibration"

    if isinstance(videoPath, list):
        for vidPath in videoPath:
            runInvertedModel(modelPath, vidPath, calibrationRoot)
    else:
        if os.path.isdir(videoPath):
            for fileName in os.listdir(videoPath):
                if fileName.lower().endswith('.mp4'):
                    vidPath = os.path.join(videoPath, fileName)
                    runInvertedModel(modelPath, vidPath, calibrationRoot)
        else:
            runInvertedModel(modelPath, videoPath, calibrationRoot)
