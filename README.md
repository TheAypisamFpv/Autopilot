# Vision-Only Autonomous Vehicle Model

Vision-based trajectory planning from two consecutive front camera frames, with optional ego-history context and a non-uniform 3-second prediction horizon.

[![Legacy video demo](https://github.com/TheAypisamFpv/Autopilot/blob/0a340a3f47bce3185cd542dd45415e015f0389b0/images/visualization/Autopilot_TrajectoryModel_EarlyFusion_AttentiveGRU_epoch19_timelapse.gif)](https://x.com/THEAYPISAMFPV/status/1982888965666681123)

The GIF above is a legacy EarlyFusion+AttentiveGRU demo. The current primary model in code is Motion-FPN + Cross-Attention Kinematic V2.

## Current Status Snapshot

- Main model class: `TrajectoryModel` (alias of `SmoothKinematicTrajectoryModel` in `model/CreateModel.py`).
- Runtime model name: `TrajectoryModel_MotionFpn_CrossAttentionKinematic_V2`.
- Model input: two RGB frames (360x640) separated by about 0.1s, plus optional ego history `(B, 8, 3)`.
- Model output: 12 future kinematic states `(speed_mps, yawRate_radps)`.
- Default horizon times (`vectorTimes`): `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.85, 1.1, 1.35, 1.6, 2.3, 3.0]`.
- Latest run folder present in this repository for the current architecture family: `training/run23`.

## Data Visualization

Examples of processed trajectory labels projected from the NVIDIA PhysicalAI-Autonomous-Vehicles pose stream:

| Frames View | Top-Down |
|:---:|:---:|
| ![NVIDIA Dataset Visualization Frames](images/visualization/dataVisAFrames.png) | ![NVIDIA Dataset Visualization TopDown](images/visualization/dataVisATopDown.png) |

## Model Architecture

![Model architecture diagram](model/architecture/trajectory_model_arch.svg)

Core blocks:

1. MotionBackbone (shared weights on frame t and t-1)
2. MotionFpnEncoder
3. EgoStateEncoder (GRU on 8-step ego history)
4. CrossAttentionDeltaKinematicDecoder

Technical details:

- Temporal fusion is built from `[feat_t, feat_t-1, feat_t - feat_t-1]` at two scales, then fused with an FPN-style merge.
- Normalized x/y coordinate channels are appended before decoding.
- Decoder performs per-step cross-attention over spatial memory, not a single pooled read.
- Decoder internally predicts delta updates and accumulates state autoregressively.
- Teacher forcing is one-step lagged to avoid current-step label leakage.
- Ego-context dropout (`egoDropoutProb=0.15`) is used during training to reduce ego-only shortcut behavior.

Trajectory integration used by training/inference utilities:

$$
\psi_t = \psi_{t-1} + \dot{\psi}_t\,\Delta t_t
$$

$$
x_t = x_{t-1} + v_t\sin(\psi_t)\,\Delta t_t, \qquad
y_t = y_{t-1} + v_t\cos(\psi_t)\,\Delta t_t
$$

## Dataset Pipelines

### NVIDIA Pipeline (Current)

Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

Implemented in `dataset/generator.py` (`generateDatasetNvidiaClip`):

1. Read camera frames + timestamp parquet.
2. Read egomotion parquet.
3. Build frame pairs with temporal context window (default 0.1s).
4. Interpolate ego state at target timestamps.
5. Convert world motion to local incremental vectors (`x=right`, `y=forward`).
6. Save labels (and optionally images).

`labelsOnly=True` is supported. In that mode, labels store video path + frame indices and `model/train.py` reconstructs frames from source videos on the fly.

Label file format:

```text
vectors : x1,y1 x2,y2 ...
vectorTimes : t1 t2 ...
speed : ...
acceleration : ...
turnRate : ...
video : ...                 # only in labels-only mode
prevFrameIndex : ...        # only in labels-only mode
frameIndex : ...            # only in labels-only mode
```

### GoPro/GPS Pipeline (Legacy)

Legacy scripts are still present:

- `dataset/GPSExtract.py`
- `dataset/GPSTestViewer.py`
- `dataset/generator.py` (`generateDataset` for `.MP4 + .json` flow)

Legacy dataset figure:

![Legacy Dataset Size Overview](images/dataset/25.06.2025_size.png)

### Dataset Balancing

`dataset/datasetBalancer.py` builds `balanced.json` with class buckets:

- `straight`
- `left`
- `right`
- `s`
- `still`

`model/train.py` automatically consumes `balanced.json` when present.

## Training

Two training entry styles are available:

1. `runTraining.py` (scheduler wrapper)
2. `model/train.py` (direct training entry)

`runTraining.py` also supports scheduled pauses. Current defaults pause training on weekdays between 07:30 and 17:30, free resources, then resume automatically.

### What the training code currently does

- Clip-aware train/val split (not random frame-level split).
- Sub-epoch schedule (`computeCleanSubepochSchedule`) for long-running training control.
- AMP enabled when CUDA is available.
- Teacher forcing ratio decays from 0.9 to 0.0 over first 30 sub-epochs.
- Kinematic loss combines weighted SmoothL1, integration consistency, and jerk penalty.
- Saves `training_params.json`, `training_history.csv`, `best_model.pth`, and `last_model.pth`.

### Current practical defaults in scripts

`runTraining.py` defaults (editable constants in script):

- `numEpochs=2000`
- `batchSize=24`
- `learningRate=5e-5`
- `gradAccumSteps=1`
- `trainSamplesPerEpoch=8000`
- `valSamplesPerEpoch=16000`
- `featDim=384`, `hiddenDim=768`, `baseChannels=48`, `numHeads=8`, `numLayers=4`

`training/run23/training_params.json` (recorded run config in this repo):

- `batchSize=12`
- `gradAccumSteps=2`
- `trainSamplesPerEpoch=5000`
- `valSamplesPerEpoch=10000`
- `predSteps=12`, `vectorTimes=[0.1, ..., 3.0]`
- `modelName=TrajectoryModel_MotionFpn_CrossAttentionKinematic_V2`

### Legacy Curves (run13)

These plots are kept for historical context from the legacy architecture track:

![run13 Training Loss](training/run13/Losschart_up_to_20.png)

![run13 ADE and FDE Metrics](training/run13/ADE&FDEchart_up_to_20.png)

- ADE: mean L2 distance over predicted steps.
- FDE: final-step L2 distance.

## Inference and Visualization

### runModel.py

- Loads checkpoint + `training_params.json`.
- Handles NVIDIA clip mode and GoPro mode.
- Projects trajectories in camera fisheye space and top-down space.
- Can display attention overlays and motion-map overlays.

Keyboard controls:

- `q`: quit
- `v`: cycle views (original, model input view, attention view, motion-encoder view)
- `j` / `l`: previous/next attention step while in attention view

### runInvertedModel.py

Frame inversion utility that optimizes image pixels to match a target trajectory with:

- reconstruction loss
- total variation loss
- temporal consistency loss

Useful for interpreting model sensitivity and trajectory-conditioned image features.

## Project Utilities

- `dataset/datasetViewer.py`: random sample inspection + top-down/projected trajectory visualization.
- `combineVideos.py`: concatenates GoPro segments + merges JSON metadata.
- `ADS presentation/diagrams`: presentation-ready architecture and pipeline diagrams.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Most scripts currently use hardcoded local paths in their `__main__` blocks, so edit those path constants before running.

## License and Sharing Constraints

Read `LEGAL.md` before sharing project outputs.

Important point: for NVIDIA PhysicalAI-Autonomous-Vehicles derived data, results, charts, checkpoints, or visualizations may be restricted by the dataset license agreement. Keep NVIDIA-derived artifacts internal unless you have explicit rights to share them.

## Q&A

### Will model weights be released?

No public release is planned at the moment.
