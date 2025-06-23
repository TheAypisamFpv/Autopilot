# Custom Multi-Modal

An advanced vehicle trajectory prediction system leveraging deep learning to forecast vehicle path over a 3-second horizon based on visual input and GPS data.


## Overview

This project implements a multi-modal network that predicts the future trajectory of the vehicle using:
- 2 RGB images from a front-facing GoPro camera (480x270 pixels, both images are separated by a 0.1s time interval for temporal context)
- 3 scalar values derived from real-time GPS:
    - Speed (m/s)
    - Acceleration (m/s²)
    - Turn rate (deg/s)

The model outputs 12 two-dimensional vectors representing the predicted travel distance (x, y) in the 2D road plane for each 0.5-second interval.

## Data Visualization

The following visualization shows an example of processed GPS data used for training and evaluation:

![GPS Data Visualization](images/visualization/dataVis.png)

## Model Architecture

The network uses a multi-modal input approach with two processing branches that are fused to make accurate trajectory predictions:

PDF version of the architecture diagram can be found [here](model/architecture/model_architecture.pdf).

![Model Architecture](model/architecture/modelArchitecture.mmd.png)



## Dataset

.mp4 video files with extracted GPS data from a GoPro hero 5.

## Training
You'll have to wait, sorry.