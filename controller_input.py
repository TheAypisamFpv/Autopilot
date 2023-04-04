# import pyvjoy
import time

"""------------------|USE beamngpy|------------------"""

steering_full_roation = 3*360/2
steering_ration = 5#to 1
wheel_full_rotation = steering_full_roation / steering_ration

# create a joystick object
# j = pyvjoy.VJoyDevice(rID=1)

# x_axis = pyvjoy.HID_USAGE_X
# y_axis = pyvjoy.HID_USAGE_Y


def wheel_angle_to_steering_angle(angle:float):
    """
    convert the wheel angle to steering wheel angle then to joystick x pos
    """
    return angle * steering_ration


def steering_angle_to_joystick_pos(angle:float):
    """
    convert the angle of the steering wheel to a joystick position
    """
    return int((angle / wheel_full_rotation) * 32767)


def update_joystick(angle:float):
    """
    input: wheel_angle (not steering wheel angle)
    update the joystick x axis
    """
    steering_wheel_angle = wheel_angle_to_steering_angle(angle)
    x_pos = steering_angle_to_joystick_pos(steering_wheel_angle)
    # j.set_axis(x_axis, x_pos)
    # j.update()
    return x_pos

