# Custom Multi-Modal

An advanced vehicle trajectory prediction system leveraging deep learning to forecast vehicle path over a 3-second horizon from Vision only.

<a href="https://x.com/THEAYPISAMFPV/status/1982888965666681123">Video demo on 𝕏</a>


## Overview

This project implements a vision-based network that predicts the future trajectory of the vehicle using:
- 2 RGB frames from a front-facing camera (360x640 pixels, both frames are separated by a 0.1s time interval for temporal context)

The model outputs 12 two-dimensional vectors representing the predicted travel path (x, y) in the 2D road plane over a 3-second interval, using a non-uniform timing schedule (0.1s x 6, 0.25s x 4, 0.7s x 2).

## Data Visualization

The following visualization shows examples of processed pose-derived trajectory data from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset used for training and evaluation:

| Frames View | Top-Down |
|:---:|:---:|
| ![NVIDIA Dataset Visualization (Left)](images/visualization/dataVisAFrames.png) | ![NVIDIA Dataset Visualization (Right)](images/visualization/dataVisATopDown.png) |

## Model Architecture

![Model architecture diagram](model\architecture\trajectory_model_arch.svg)

The network uses a motion-aware encoder with a lightweight transformer decoder:

- **Motion Backbone + FPN Fusion**: Two shared CNN backbones extract multi-scale features from the current and previous frames. Per-scale features are concatenated with their temporal differences, fused through 1x1 convolutions, and combined with a top-down FPN-style merge.

- **Spatial Coordinate Channels**: Normalized x/y coordinate channels are appended to the fused feature map to provide explicit spatial context.

- **Vector Transformer Decoder**: A transformer decoder with learned query embeddings and a time-step MLP predicts the future displacement vectors in parallel.

- **Optional Auxiliary Head**: A small MLP on pooled features can predict auxiliary dynamics (2D output).

By default, the model predicts 12 vectors over a 3-second horizon with a non-uniform timing schedule (0.1s x 6, 0.25s x 4, 0.7s x 2). You can override the time steps at runtime. *(model in training, so changing the time steps has an unknown impact on performance)*


## Dataset

### New way (NVIDIA PhysicalAI-Autonomous-Vehicles)

The current dataset used is NVIDIA’s PhysicalAI-Autonomous-Vehicles dataset:
*https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles*

The data generation process for this dataset is similar in spirit to the old GoPro pipeline, but uses the dataset's per-frame ego-pose and timestamps to build the trajectory labels:

1. Sample consecutive RGB frames separated by 0.1s (same as the model input).
2. Read the ego vehicle pose for each frame (position + orientation) from the dataset.
3. For each input pair, collect future ego poses over the next 3 seconds.
4. Convert those future poses into local displacements relative to the current frame (vehicle-centric x/y).
5. Downsample or resample to 12 vectors using the non-uniform timing schedule (0.1s x 6, 0.25s x 4, 0.7s x 2).

This produces the same kind of 12-step (x, y) displacement targets the model expects, but derived from the dataset's provided pose stream instead of GPS.

#### **License/Terms of Use**
This dataset is governed by the NVIDIA Autonomous Vehicle Dataset License Agreement. This means that i can't share any statistics or details about how models trained on this dataset perform, sorry :/
you can find more details about the license here:
*https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf*.


### Old way (GoPro with GPS)

The original dataset consists of processed frames from .mp4 video files captured with a GoPro Hero 5 camera, paired with its GPS data. The data generation process involves:

1. Extracting GPS data (position, speed, timestamps) from the GoPro's metadata
2. Processing video frames at specific intervals (640x360 pixel resolution)
3. For each frame pair:
   - Two consecutive frames separated by 0.1 seconds are extracted
   - GPS data is validated for accuracy (fix type 3* and accuracy < 3.0m)
   - Future trajectory vectors are calculated for the next 3 seconds
   - Random adjustments to exposure, gamma, brightness, and contrast are applied for a better generalization (hopefully)

The generated dataset is over 130Go with 432501 datapoints.

![Dataset Size Overview](images/dataset/25.06.2025_size.png)


*fix type 3 means the GPS has a 3D fix, which is the most accurate fix type available. The accuracy threshold of < 3.0m ensures that the GPS data is reliable for trajectory prediction, tho even with this fix type GPS data can still be inacurate.*

*Filtering based on map data could be a potential improvement for the dataset quality.*

### Example Scenarios

| Lane Changes | Left Turns | Right Turns |
|:---:|:---:|:---:|
| ![Lane Change](images/dataset/dataset_exemple_laneChange.png) | ![Left Turn](images/dataset/dataset_exemple_leftTurn.png) | ![Right Turn](images/dataset/dataset_exemple_rightTurn.png) |

| Roundabout Entry | In Roundabout | Roundabout Exit |
|:---:|:---:|:---:|
| ![Roundabout Entry](images/dataset/dataset_exemple_roundaboutEnter.png) | ![In Roundabout](images/dataset/dataset_exemple_inRoundabout.png) | ![Roundabout Exit](images/dataset/dataset_exemple_roundaboutExit.png) |

Each sample in the dataset includes the image pair and the ground truth trajectory vectors representing the vehicle's future path. (Legacy records may include speed metadata, but the current model is vision-only.)

## Training

### Current run in training: **run13**
100 hours of training as of now at epoch 20.

The new architecture reduced parameters from 23M to 8.8M while achieving better performance at equivalent training time.

![run13 Training Loss](training/run13/Losschart_up_to_20.png)

![run13 ADE & FDE Metrics](training/run13/ADE&FDEchart_up_to_20.png)

Check out a video demonstration of the model in action (at epoch 19):

<a href="https://x.com/THEAYPISAMFPV/status/1982888965666681123">View the video demonstration on 𝕏</a>

## Q&A

### Will the model's weights be available?

**No**, i'm not planning on releasing them for now (they are shitty anyway). If a good model is trained, then i'll consider it.

### How can I help?
By sending me a better GPU for AI training then an rtx2060 6G, thanks :)