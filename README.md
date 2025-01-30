# Autopilot AI (YoloV9)

Uses `YoloV9c-seg` to detect `drivable surfaces`, `lines` and `obstacles`.
It is trained on human-labeled images. Later I will try to integrate auto labeling.

<p align="left">Best Weight <a href="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-AI/Best%20Weights/Yolo9_custom.pt" rel="noopener">here</a>
    <br>
</p>

# Images
## Labeled images

Here are some examples of labeled images.

`Drivable surfaces` are sections of the road where the car should drive, like right hand side of non mark roads, or between the lines on a road with lines.

`Lines` are the lines on the road, like the white lines on the side of the road or the yellow lines in case of construction work or the edge of the road.

`Obstacles` are objects that the car should avoid, like cars, pedestrians, or "real" obstacles like cones or road signs.

<br>

### Validation batch labeled images:

<p align="center">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/tree/Autopilot-AI/images/val_batch0_labels.jpg" alt="Labeled images">
</p>

<br>

### Validation batch predicted images:

<p align="center">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/tree/Autopilot-AI/images/val_batch0_pred.jpg" alt="Predicted images">
</p>
