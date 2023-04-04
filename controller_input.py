import pyvjoy
import time

"""------------------|USE beamngpy|------------------"""

steering_full_roation = 3*360/2
steering_ration = 5#to 1
wheel_full_rotation = steering_full_roation / steering_ration

# create a joystick object
j = pyvjoy.VJoyDevice(rID=1)

x_axis = pyvjoy.HID_USAGE_X
y_axis = pyvjoy.HID_USAGE_Y


def map(value, In_Min, In_Max, Out_Min, Out_Max) -> float:
    """
    map 'value' from 'In_Min' - 'In_Max' to 'Out_Min' - 'Out_Max'
    """
    ## Figure out how 'wide' each range is
    In_Span = In_Max - In_Min
    Out_Span = Out_Max - Out_Min

    ## Convert the left range into a 0-1 range (float)
    valueScaled = float(value - In_Min) / float(In_Span)

    ## Convert the 0-1 range into a value in the right range.
    return Out_Min + (valueScaled * Out_Span)



def convert_angle_to_pos(angle:float):
    """
    convert the wheel angle to steering wheel angle then to joystick x pos
    """
    steering_wheel_angle = angle * steering_ration
    x_pos = int((steering_wheel_angle / wheel_full_rotation) * 32767)

    return steering_wheel_angle, x_pos


def update_joystick(angle:float):
    """
    update the joystick x axis
    """
    steering_wheel_angle, x_pos = convert_angle_to_pos(angle)
    j.set_axis(x_axis, x_pos)
    j.update()
    return steering_wheel_angle, x_pos

