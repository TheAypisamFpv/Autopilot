import gc
import os
import time
from datetime import datetime

import torch

from model.train import trainModel

try:
    import psutil
except ImportError:
    psutil = None


def formatHour(hourFloat):
    hours = int(hourFloat)
    minutes = int(round((hourFloat - hours) * 60))
    if minutes == 60:
        hours = (hours + 1) % 24
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


def isNoTrainingWindow(now, enableNoTrainingHours, noTrainingStartHour, noTrainingEndHour, weekendsTraining):
    if not enableNoTrainingHours:
        return False

    isWeekday = now.weekday() < 5
    applyNoTrainingWindow = isWeekday or (not weekendsTraining)
    if not applyNoTrainingWindow:
        return False

    currentHour = now.hour + now.minute / 60.0 + now.second / 3600.0
    if noTrainingStartHour <= noTrainingEndHour:
        return noTrainingStartHour <= currentHour < noTrainingEndHour

    # Supports windows crossing midnight.
    return currentHour >= noTrainingStartHour or currentHour < noTrainingEndHour


def getCpuRamSnapshotMb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    rssMb = process.memory_info().rss / (1024 ** 2)
    vmsMb = process.memory_info().vms / (1024 ** 2)
    return rssMb, vmsMb


def getGpuSnapshotMb():
    if not torch.cuda.is_available():
        return None

    allocatedMb = torch.cuda.memory_allocated() / (1024 ** 2)
    reservedMb = torch.cuda.memory_reserved() / (1024 ** 2)
    freeBytes, totalBytes = torch.cuda.mem_get_info()
    freeMb = freeBytes / (1024 ** 2)
    totalMb = totalBytes / (1024 ** 2)
    return allocatedMb, reservedMb, freeMb, totalMb


def logResourceSnapshot(prefix):
    cpuSnapshot = getCpuRamSnapshotMb()
    gpuSnapshot = getGpuSnapshotMb()

    if cpuSnapshot is None:
        print(f"{prefix} RAM snapshot unavailable (install psutil for detailed CPU/RAM metrics).")
    else:
        rssMb, vmsMb = cpuSnapshot
        print(f"{prefix} RAM RSS={rssMb:.2f} MB | VMS={vmsMb:.2f} MB")

    if gpuSnapshot is None:
        print(f"{prefix} GPU snapshot: CUDA not available.")
    else:
        allocatedMb, reservedMb, freeMb, totalMb = gpuSnapshot
        print(
            f"{prefix} GPU allocated={allocatedMb:.2f} MB | reserved={reservedMb:.2f} MB | "
            f"free={freeMb:.2f}/{totalMb:.2f} MB"
        )


def freeTrainingResources():
    print("Starting resource freeup...")
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    print("Resource freeup finished.")


def buildStopCallback(enableNoTrainingHours, noTrainingStartHour, noTrainingEndHour, weekendsTraining):
    def stopAfterSubepochCallback(subEpochIndex, numSubEpochs):
        now = datetime.now()
        shouldPause = isNoTrainingWindow(
            now,
            enableNoTrainingHours=enableNoTrainingHours,
            noTrainingStartHour=noTrainingStartHour,
            noTrainingEndHour=noTrainingEndHour,
            weekendsTraining=weekendsTraining,
        )
        if shouldPause:
            print("\n====Stopping training during working hours to free resources====")
            print(
                f"Scheduler requested stop at {now.strftime('%Y-%m-%d %H:%M:%S')} "
                f"after sub-epoch {subEpochIndex}/{numSubEpochs}."
            )
        return shouldPause

    return stopAfterSubepochCallback


if __name__ == "__main__":
    datasetPath = r"C:\Users\Projet_3NC\Desktop\SC-ADS\dataset_output\output_NVIDIA_12_3.0_0.1_framesize640x360"
    datasetMaxSize = None
    numEpochs = 2_000
    patience = 100
    batchSize = 24
    gradAccumSteps = 1
    trainValSplit = 0.8
    trainSamplesPerEpoch = 8_000
    valSamplesPerEpoch = trainSamplesPerEpoch * 2
    seed = 42
    splitSeed = 42
    learningRate = 5e-5
    featDim = 384
    hiddenDim = 768
    baseChannels = 48
    numHeads = 8
    numLayers = 4
    useAuxDyn = False
    resumeModelPath = r"C:\Users\Projet_3NC\Desktop\SC-ADS\Autopilot\training\run23\last_model.pth"

    enableNoTrainingHours = True
    noTrainingStartHour = 7.50  # 07:30
    noTrainingEndHour = 17.50   # 18:00
    weekendsTraining = True     # True: no-training window applies only on weekdays

    schedulerCheckIntervalSeconds = 60
    restartDelaySeconds = 15

    if enableNoTrainingHours:
        print(
            "No-training window enabled: "
            f"{formatHour(noTrainingStartHour)}-{formatHour(noTrainingEndHour)} | "
            f"weekendsTraining={weekendsTraining}"
        )
    else:
        print("No-training window disabled: training will run at all hours.")

    hasPrintedBlockedMessage = False
    resourcesFreed = False

    while True:
        now = datetime.now()
        inNoTrainingWindow = isNoTrainingWindow(
            now,
            enableNoTrainingHours=enableNoTrainingHours,
            noTrainingStartHour=noTrainingStartHour,
            noTrainingEndHour=noTrainingEndHour,
            weekendsTraining=weekendsTraining,
        )

        if inNoTrainingWindow:
            if not hasPrintedBlockedMessage:
                print("\n====Stopping training during working hours to free resources====")
                print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(
                    "Training is paused by schedule "
                    f"({formatHour(noTrainingStartHour)}-{formatHour(noTrainingEndHour)})."
                )
                hasPrintedBlockedMessage = True

            # Only perform the expensive freeup once when entering the blocked window
            if not resourcesFreed:
                logResourceSnapshot("Before freeup:")
                freeTrainingResources()
                logResourceSnapshot("After freeup:")
                resourcesFreed = True

            time.sleep(max(1, schedulerCheckIntervalSeconds))
            continue

        if hasPrintedBlockedMessage:
            print("\n====Restarting training outside working hours====")
            print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

        hasPrintedBlockedMessage = False
        resourcesFreed = False

        stopCallback = buildStopCallback(
            enableNoTrainingHours=enableNoTrainingHours,
            noTrainingStartHour=noTrainingStartHour,
            noTrainingEndHour=noTrainingEndHour,
            weekendsTraining=weekendsTraining,
        )

        result = trainModel(
            datasetDir=datasetPath,
            numEpochs=numEpochs,
            batchSize=batchSize,
            learningRate=learningRate,
            trainValSplit=trainValSplit,
            patience=patience,
            datasetMaxSize=datasetMaxSize,
            gradAccumSteps=gradAccumSteps,
            trainSamplesPerEpoch=trainSamplesPerEpoch,
            valSamplesPerEpoch=valSamplesPerEpoch,
            seed=seed,
            splitSeed=splitSeed,
            useAuxDyn=useAuxDyn,
            featDim=featDim,
            hiddenDim=hiddenDim,
            baseChannels=baseChannels,
            numHeads=numHeads,
            numLayers=numLayers,
            resumeModelPath=resumeModelPath,
            stopAfterSubepochCallback=stopCallback,
        )

        if isinstance(result, dict):
            nextResumePath = result.get("resumeModelPath")
            if nextResumePath:
                resumeModelPath = nextResumePath

            stoppedByScheduler = bool(result.get("stoppedByScheduler", False))
            if stoppedByScheduler:
                logResourceSnapshot("Before freeup:")
                freeTrainingResources()
                logResourceSnapshot("After freeup:")
                continue

        print("Training run ended outside no-training schedule. Waiting before restart...")
        logResourceSnapshot("Before freeup:")
        freeTrainingResources()
        logResourceSnapshot("After freeup:")
        time.sleep(max(1, restartDelaySeconds))
