# Autopilot without AI (latest version: V6)



Capable of lane centering, lane following, and lane departure warning, the autopilot is able to activate as long as it detects one line.

The second line will be predicted using the lane width, which is the average distance between the left and right lines when both are detected and stable over time (e.g., during a straight line).

This can be seen in the "Lane departure warning" image, where the right line is highlighted in red but not detected. By using the position of the left line, the system determines that the right line is too far to the left and triggers a right lane departure warning.

A line is considered not detected when it is not visible or not stable enough to be trusted.


# Images
## Autopilot active

<p align="center">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-no-AI/images/V6/autopilotV6%20active%201.png" alt="Autopilot active1">
</p>
<p align="center"> 
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-no-AI/images/V6/autopilotV6%20active%202.png" alt="Autopilot active2">
</p>

## Lane prediction
<p align="center">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-no-AI/images/V6/autopilotV6%20lane%20prediction.png" alt="Lane prediction">
</p>

## Lane departure warning
<p align="center"> 
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-no-AI/images/V6/autopilotV6%20lane%20departure.png" alt="lane departure warning">
</p>

## Autopilot unavailable
<p align="center">
 <img width=800px height=auto src="https://github.com/TheAypisamFpv/Autopilot/blob/Autopilot-no-AI/images/V6/autopilotV6%20unavailable.png" alt="Autopilot unavailable">
</p>
