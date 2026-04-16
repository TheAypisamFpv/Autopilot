import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LineSeriesSpec:
    yColumnName: str
    legendLabel: str
    lineWidth: float = 1.6


@dataclass(frozen=True)
class PlotSpec:
    csvPath: str
    xColumnName: Optional[str]
    ySeriesSpecs: Sequence[LineSeriesSpec]
    xScale: str
    yScale: str
    xLabel: str
    yLabel: str
    plotTitle: str
    outputFilePath: str
    figureSize: Tuple[float, float] = (10.5, 5.8)
    dpi: int = 220
    gridAlpha: float = 0.28


def ensureFileExists(filePath):
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Missing file: {filePath}")


def ensureDirectoryExists(directoryPath):
    if directoryPath and not os.path.exists(directoryPath):
        os.makedirs(directoryPath, exist_ok=True)


def loadHistoryFrame(filePath):
    ensureFileExists(filePath)
    return pd.read_csv(filePath)


def validateScaleName(scaleName, axisName):
    supportedScales = {"linear", "log", "symlog"}
    if scaleName not in supportedScales:
        raise ValueError(
            f"Unsupported {axisName} scale '{scaleName}'. Supported: {sorted(supportedScales)}"
        )


def getNumericColumnValues(historyFrame, columnName):
    if columnName not in historyFrame.columns:
        raise ValueError(f"Missing required column '{columnName}' in CSV")

    columnValues = historyFrame[columnName].to_numpy(dtype=float)
    if not np.all(np.isfinite(columnValues)):
        raise ValueError(f"Column '{columnName}' contains non-finite values")
    return columnValues


def validateScaleCompatibility(values, scaleName, valueName):
    if scaleName == "log" and np.any(values <= 0):
        raise ValueError(f"{valueName} contains values <= 0, invalid for log scale")


def buildXAxisValues(historyFrame, xColumnName):
    if xColumnName is None:
        return np.arange(1, len(historyFrame) + 1, dtype=float)
    return getNumericColumnValues(historyFrame, xColumnName)


def createPlotFromSpec(plotSpec):
    validateScaleName(plotSpec.xScale, "x-axis")
    validateScaleName(plotSpec.yScale, "y-axis")

    historyFrame = loadHistoryFrame(plotSpec.csvPath)
    xAxisValues = buildXAxisValues(historyFrame, plotSpec.xColumnName)
    validateScaleCompatibility(xAxisValues, plotSpec.xScale, "x-axis values")

    plt.figure(figsize=plotSpec.figureSize)
    for seriesSpec in plotSpec.ySeriesSpecs:
        yAxisValues = getNumericColumnValues(historyFrame, seriesSpec.yColumnName)
        validateScaleCompatibility(
            yAxisValues,
            plotSpec.yScale,
            f"Column '{seriesSpec.yColumnName}'",
        )
        plt.plot(
            xAxisValues,
            yAxisValues,
            label=seriesSpec.legendLabel,
            linewidth=seriesSpec.lineWidth,
        )

    plt.xscale(plotSpec.xScale)
    plt.yscale(plotSpec.yScale)
    plt.title(plotSpec.plotTitle)
    plt.xlabel(plotSpec.xLabel)
    plt.ylabel(plotSpec.yLabel)
    plt.grid(alpha=plotSpec.gridAlpha)
    plt.legend()
    plt.tight_layout()

    ensureDirectoryExists(os.path.dirname(plotSpec.outputFilePath))
    plt.savefig(plotSpec.outputFilePath, dpi=plotSpec.dpi)
    plt.close()
    return plotSpec.outputFilePath


def createPlots(plotSpecs):
    generatedPlotPaths = []
    for plotSpec in plotSpecs:
        generatedPlotPaths.append(createPlotFromSpec(plotSpec))
    return generatedPlotPaths


def buildDefaultRunLossPlotSpecs(projectRoot, runName):
    trainingHistoryPath = os.path.join(projectRoot, "training", runName, "training_history.csv")
    visualizationDirectory = os.path.dirname(__file__)

    commonSeriesSpecs = [
        LineSeriesSpec(yColumnName="train_loss", legendLabel="train_loss"),
        LineSeriesSpec(yColumnName="val_loss", legendLabel="val_loss"),
    ]

    return [
        PlotSpec(
            csvPath=trainingHistoryPath,
            xColumnName=None,
            ySeriesSpecs=commonSeriesSpecs,
            xScale="linear",
            yScale="linear",
            xLabel="Sub-epoch",
            yLabel="Loss",
            plotTitle=f"Training Loss Curves - {runName}",
            outputFilePath=os.path.join(visualizationDirectory, f"{runName}_loss.png"),
        ),
        PlotSpec(
            csvPath=trainingHistoryPath,
            xColumnName=None,
            ySeriesSpecs=commonSeriesSpecs,
            xScale="linear",
            yScale="log",
            xLabel="Sub-epoch",
            yLabel="Loss",
            plotTitle=f"Training Loss Curves (log Y) - {runName}",
            outputFilePath=os.path.join(visualizationDirectory, f"{runName}_lossLog10y.png"),
        ),
        PlotSpec(
            csvPath=trainingHistoryPath,
            xColumnName=None,
            ySeriesSpecs=commonSeriesSpecs,
            xScale="log",
            yScale="log",
            xLabel="Sub-epoch",
            yLabel="Loss",
            plotTitle=f"Training Loss Curves (log X and Y) - {runName}",
            outputFilePath=os.path.join(visualizationDirectory, f"{runName}_lossLog10y&x.png"),
        ),
    ]


def main():
    projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    runName = "run23"

    plotSpecs = buildDefaultRunLossPlotSpecs(projectRoot, runName)
    # Add more PlotSpec entries here to batch-generate additional figures.
    generatedPlotPaths = createPlots(plotSpecs)

    for generatedPlotPath in generatedPlotPaths:
        print(f"Saved figure: {generatedPlotPath}")


if __name__ == "__main__":
    main()
