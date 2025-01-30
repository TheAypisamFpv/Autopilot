<h1 align="center">Autopilot AI (YoloV9)</h1>

<p align="left">Uses `YoloV9c-seg` to detect `drivable surfaces`, `lines` and `obstacles`.
It is trained on human-labeled images. Later I will try to integrate auto labeling.</p>

<p align="left">Best Weight <a href="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20AI/Best%20Weights/Yolo9_custom.pt" rel="noopener">here</a>
    <br>
</p>

# Images
## Labeled images

<p align="left">Here are some examples of labeled images. `Drivable surfaces` are sections of the road where the car should drive, like right hand side of non mark roads, or between the lines on a road with lines. `Lines` are the lines on the road, like the white lines on the side of the road or the yellow lines in case of construction work or the edge of the road. `Obstacles` are objects that the car should avoid, like cars, pedestrians, or "real" obstacles like cones or road signs.
    <br>
</p>

<p align="left">Validation batch labeled images:
    <br> 
</p>

<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20AI/images/val_batch0_labels.jpg" alt="Labeled images"></a>
</p>


<p align="left">Validation batch predicted images:
    <br>
</p>

<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20AI/images/val_batch0_pred.jpg" alt="Predicted images"></a>
</p>

<br>
<br>

---
<h1 align="center">Autopilot without AI (latest version: V6)</h1>



<p align="left"> Capable of lane centering, lane following, and lane departure warning, the autopilot is able to activate as long as it detects one line. The second line will be predicted using the lane width, which is the average distance between the left and right lines when both are detected and stable over time (e.g., during a straight line). This can be seen in the "Lane departure warning" image, where the right line is highlighted in red but not detected. By using the position of the left line, the system determines that the right line is too far to the left and triggers a right lane departure warning.
A line is considered not detected when it is not visible or not stable enough to be trusted. More about the inner working later !
    <br> 
</p>

# Images
## Autopilot active

<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20non%20AI/images/V6/autopilotV6%20active%201.png" alt="Autopilot active1"></a>
</p>
<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20non%20AI/images/V6/autopilotV6%20active%202.png" alt="Autopilot active2"></a>
</p>

## Lane prediction
<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20non%20AI/images/V6/autopilotV6%20lane%20prediction.png" alt="Lane prediction"></a>
</p>

## Lane departure warning
<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20non%20AI/images/V6/autopilotV6%20lane%20departure.png" alt="lane departure warning"></a>
</p>

## Autopilot unavailable
<p align="center"> 
  <a href="" rel="noopener">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/master/autopilot%20non%20AI/images/V6/autopilotV6%20unavailable.png" alt="Autopilot unavailable"></a>
</p>
