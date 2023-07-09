import cv2 as cv
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import controller_input


""" Blue Green Red """
AVAILABLE_COLOR        = (255, 255, 255), (232, 39, 33)
UNAVAILABLE_COLOR      = (100, 100, 100), (0  , 0 , 0 )
ALERTE_DEPARTURE_COLOR = (39 , 33 , 232), (0  , 0 , 0 )

autopilot_available       = False
autopilot_available_color = UNAVAILABLE_COLOR

height = 480
width  = 720


lane_center        = int(width/2) + 0 # + offset
car_hood           = 75
lower_lane_width   = width+150
detection_distance = 110
distance_center    = lane_center +5 # + offset
distance_width     = 200

upper_left  = [int(distance_center - distance_width/2),int(height - car_hood - detection_distance)]
upper_right = [int(distance_center + distance_width/2),int(height - car_hood - detection_distance)]

lower_left  = [int(lane_center - lower_lane_width/2), int(height-car_hood)]
lower_right = [int(lane_center + lower_lane_width/2), int(height-car_hood)]

pts1 = np.float32([upper_left, lower_left, upper_right, lower_right])
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

cv.namedWindow('Trackbars')
default_canny_low  = 200
default_canny_high = 225

cv.createTrackbar('Canny_low' , 'Trackbars', default_canny_low , 255, lambda x: None)
cv.createTrackbar('Canny_high', 'Trackbars', default_canny_high, 255, lambda x: None)

# cv.createTrackbar("Box_height", "Trackbars", 25, 100, lambda x: None)
# cv.createTrackbar("Box_width", "Trackbars", 100, 200, lambda x: None)
cv.createTrackbar('Target FPS', 'Trackbars', 60, 150, lambda x: None)



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



def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]



def pad_img_to_fit_bbox(img, x1:int, x2:int, y1:int, y2:int):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)), (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2



def detect_lane_width(frame, displayed_frame, prev_left_line_x_pos, prev_right_line_x_pos):
    """
    return the line's position (lower and upper) and if they are detected (both lower and upper)
    """

    ## modifiy contrast and brightness
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # white_frame = cv.threshold(gray_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    alpha = 1
    beta = -50
    corected_frame = cv.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)
    # corected_frame = cv.bitwise_and(gray_frame, white_frame)

    ## blur the frame
    blur_frame = cv.GaussianBlur(corected_frame, (51, 51), 0)

    ## combine the frames
    combined_frame = cv.add(blur_frame, corected_frame)

    ## bird's eye view of th frame
    M = cv.getPerspectiveTransform(pts1, pts2)
    displayed_frame = cv.cvtColor(combined_frame, cv.COLOR_GRAY2BGR)
    displayed_frame = cv.warpPerspective(displayed_frame, M, (width, height))

    ## canny edge detection
    canny_low  = default_canny_low
    canny_high = default_canny_high

    canny_low  = cv.getTrackbarPos('Canny_low' , 'Trackbars')
    canny_high = cv.getTrackbarPos('Canny_high', 'Trackbars')

    canny_frame = cv.Canny(combined_frame, canny_low, canny_high)
    canny_frame = cv.warpPerspective(canny_frame, M, (width, height))
    canny_frame = cv.dilate(canny_frame, None, iterations=1)
    canny_blur_frame = cv.GaussianBlur(canny_frame, (3, 71), 0, 0, cv.BORDER_CONSTANT, 0)

    canny_frame = cv.add(canny_frame, canny_blur_frame)

    ## convert to black and white
    bw_canny_frame = cv.threshold(canny_frame, 0, 255, cv.THRESH_BINARY)[1]

    ## detect start of the lines
    lower_left_found = False
    lower_right_found = False

    detecting_height = 50
    center_detecting_ignore = 50

    ## projection based lane detecting
    for pixel in range(center_detecting_ignore, int(width/2)-10, 5):
        if bw_canny_frame[height-detecting_height:height, int(width/2 - pixel)].any() > 0 and not lower_left_found:
            left_base   = int(width/2 - pixel)-15
            lower_left_found  = True
        elif not lower_left_found:
            left_base   = int(prev_left_line_x_pos)

        if bw_canny_frame[height-detecting_height:height, int(width/2 + pixel)].any() > 0 and not lower_right_found:
            right_base  = int(width/2 + pixel)+15
            lower_right_found = True
        elif not lower_right_found:
            right_base  = int(prev_right_line_x_pos)

        if lower_left_found and lower_right_found:
            break    


    ## histogram based lane detecting

    # histogram = np.sum(bw_canny_frame[height-detecting_height:height, :], axis=0)


    ## center = lane_center
    # exterior_threshold = 10
    # interior_threshold = 10
    # left_base  = np.argmax(histogram[       exterior_threshold       :lane_center - interior_threshold]) + exterior_threshold + 20
    # right_base = np.argmax(histogram[lane_center + interior_threshold:width       - exterior_threshold]) + lane_center + interior_threshold

    # if left_base == exterior_threshold + 20:
    #     left_base = int(prev_left_line_x_pos)
    #     left_found = False
    # else:
    #     left_found = True

    # if right_base == lane_center + interior_threshold:
    #     right_base = int(prev_right_line_x_pos)
    #     right_found = False
    # else:
    #     right_found = True



    ## sliding boxs
    left_boxs_x = [left_base]
    left_boxs_y = [height]
    right_boxs_x = [right_base]
    right_boxs_y = [height]

    box_height = 25
    # box_height_trackbar = cv.getTrackbarPos('Box_height', 'Trackbars')
    # box_height = box_height_trackbar if box_height_trackbar > 0 else box_height

    box_width = 100
    # box_width_trackbar = cv.getTrackbarPos('Box_width', 'Trackbars')
    # box_width = box_width_trackbar if box_width_trackbar > 0 else box_width

    y = height - box_height

    while y > 0:

        ## left threshold
        img = bw_canny_frame[y-box_height:y, left_base -int(box_width/2):left_base + int(box_width/2)]
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            moment = cv.moments(cnt)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                left_boxs_x.append(cx + left_base - int(box_width/2))
                left_boxs_y.append(cy + y)
                left_base = left_base - int(box_width/2) + cx
                lower_left_found = True



        ## right threshold
        img = bw_canny_frame[y-box_height:y, right_base -int(box_width/2):right_base + int(box_width/2)]
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            moment = cv.moments(cnt)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                right_boxs_x.append(cx + right_base - int(box_width/2))
                right_boxs_y.append(cy + y)
                right_base = right_base - int(box_width/2) + cx
                lower_right_found = True

        cv.rectangle(bw_canny_frame, (left_base  - int(box_width/2), y-box_height), (left_base  + int(box_width/2), y), (255, 255, 255), 2)
        cv.rectangle(bw_canny_frame, (right_base - int(box_width/2), y-box_height), (right_base + int(box_width/2), y), (255, 255, 255), 2)

        y -= box_height


    lines_x = [left_boxs_x, right_boxs_x]
    lines_y = [left_boxs_y, right_boxs_y]

    for boxs in range(len(lines_x)):
        boxs_x = lines_x[boxs]
        for index in range(len(boxs_x)-1):
            x_1 = boxs_x[index]
            y_1 = lines_y[boxs][index]

            x_2 = boxs_x[index + 1]
            y_2 = lines_y[boxs][index + 1]

            cv.line(bw_canny_frame, (x_1, y_1), (x_2, y_2), (0  , 0  , 0  ), 12)
            cv.line(bw_canny_frame, (x_1, y_1), (x_2, y_2), (255, 255, 255), 8 )

    left_line_lower_x_pos  =  left_boxs_x[ 0]
    right_line_lower_x_pos = right_boxs_x[ 0]
    

    left_line_upper_x_pos  =  left_boxs_x[-1]
    right_line_upper_x_pos = right_boxs_x[-1]

    upper_left_found  = True if (len(left_boxs_x)  > 1) and (left_boxs_y[ -1] < height/3) else False
    upper_right_found = True if (len(right_boxs_x) > 1) and (right_boxs_y[-1] < height/3) else False


    return displayed_frame, bw_canny_frame, left_line_lower_x_pos, left_line_upper_x_pos, lower_left_found, upper_left_found, right_line_lower_x_pos, right_line_upper_x_pos, lower_right_found, upper_right_found



def lane_departure(left_line_x_pos, left_found, left_line_predicted, right_line_x_pos, right_found, right_line_predicted) -> tuple[int,int]:
    """
    detect lane departure
    return if there is a lane departure on left, right line
    """

    left_line  = int(   lane_center   - left_line_x_pos)
    right_line = int(right_line_x_pos -   lane_center  )

    left_line_departure  = 0
    right_line_departure = 0

    left_line_departure_threshold  = 100
    right_line_departure_threshold = 100

    if left_line <= left_line_departure_threshold and (left_found or (left_line_predicted and right_found)) :
        left_line_departure = 1
    
    if right_line <= right_line_departure_threshold and (right_found or (right_line_predicted and left_found)):
        right_line_departure = 1

    return left_line_departure, right_line_departure



def main():
    
    ## lane positions init


    """-----------------------------------Left----------------------------------"""
    left_line_lower_x_pos      = (width/2 - lower_lane_width/2)
    left_line_lower_x_pos_prev = left_line_lower_x_pos

    left_line_upper_x_pos      = left_line_lower_x_pos
    left_line_upper_x_pos_prev = left_line_upper_x_pos

    predicted_lower_left_line_x_pos      = left_line_lower_x_pos
    predicted_lower_left_line_x_pos_prev = predicted_lower_left_line_x_pos

    predicted_upper_left_line_x_pos_prev = left_line_upper_x_pos

    left_line_lower_incertitude      = 0
    left_line_lower_incertitude_prev = 0

    left_line_upper_incertitude      = 0
    left_line_upper_incertitude_prev = 0


    left_line_angle      = 0
    left_line_angle_prev = 0

    left_color_prev = UNAVAILABLE_COLOR[0]

    upper_left_found = False

    """----------------------------------Right----------------------------------"""
    right_line_lower_x_pos = (width/2 - lower_lane_width/2)
    right_line_lower_x_pos_prev = right_line_lower_x_pos

    right_line_upper_x_pos      = right_line_lower_x_pos
    right_line_upper_x_pos_prev = right_line_upper_x_pos

    predicted_lower_right_line_x_pos      = right_line_lower_x_pos
    predicted_lower_right_line_x_pos_prev = predicted_lower_right_line_x_pos

    predicted_upper_right_line_x_pos_prev = right_line_upper_x_pos

    right_line_lower_incertitude      = 0
    right_line_lower_incertitude_prev = 0
    
    right_line_upper_incertitude      = 0
    right_line_upper_incertitude_prev = 0


    right_line_angle      = 0
    right_line_angle_prev = 0

    right_color_prev = UNAVAILABLE_COLOR[0]

    upper_right_found = False

    """----------------------------------Lane-----------------------------------"""
    pre_lower_lane_width = 290
    upper_lane_width     = 290
    pre_upper_lane_width = 290

    lane_keeping_angle_prev = 0
    last_lane_keeping_angle = 0
    
    lane_angle           = 0
    lane_angle_prev      = 0
    last_good_lane_angle = 0

    lane_color_prev = UNAVAILABLE_COLOR[0]

    steering_wheel_color_prev = UNAVAILABLE_COLOR[0]
    steering_color            = UNAVAILABLE_COLOR[0]
    steering_color_prev       = UNAVAILABLE_COLOR[0]
    steering_angle = 0

    autopilot_available = False

    """----------------------------------Other----------------------------------"""
    lane_keeping_I = 0
    error_prev = 0
    
    setting = ''
    fps = 60
    prev_fps = fps
    average_fps = fps


    ## Video here
    # cap = cv.VideoCapture(1)
    cap = cv.VideoCapture('Test drive\\07.04.2023\\full.MP4')
    

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_id = cap.get(1)

        if frame_id % 2 or True:
            if ret:
                start_time = time.time()
                frame_width = int(cap.get(3))
                
                if frame_width != width:
                  frame = cv.resize(frame, (width, height))
                  
                displayed_frame = frame.copy()
                
                # frame[:, :, 0] = 0
                # frame[:, :, 1] = 0

                corected_frame, bw_canny_frameed_frame, true_left_line_x_pos, left_line_upper_x_pos, lower_left_found, upper_left_found, true_right_line_x_pos, right_line_upper_x_pos, lower_right_found, upper_right_found = detect_lane_width(frame, displayed_frame, left_line_lower_x_pos_prev, right_line_lower_x_pos_prev)

                video_fps = cv.getTrackbarPos('Target FPS', 'Trackbars')  # cap.get(cv.CAP_PROP_FPS)

                video_fps = video_fps if video_fps > 0 else 0.5

                
                ## smooth the lines and calculate the incertitudes
                incertitude_threshold = 8

                left_line_lower_x_pos  = (true_left_line_x_pos   +  left_line_lower_x_pos_prev *  2) / 3
                left_line_upper_x_pos  = (left_line_upper_x_pos  +  left_line_upper_x_pos_prev * 5) / 6


                left_line_lower_incertitude = abs(left_line_lower_x_pos - left_line_lower_x_pos_prev)
                left_line_lower_incertitude = (left_line_lower_incertitude  +  left_line_lower_incertitude_prev * 5) / 6


                left_line_upper_incertitude = abs(left_line_upper_x_pos - left_line_upper_x_pos_prev)
                left_line_upper_incertitude = (left_line_upper_incertitude  +  left_line_upper_incertitude_prev * 2) / 3


                lower_left_found = False if left_line_lower_incertitude > incertitude_threshold else lower_left_found
                upper_left_found = False if left_line_upper_incertitude > incertitude_threshold else upper_left_found




                right_line_lower_x_pos = (true_right_line_x_pos  + right_line_lower_x_pos_prev *  2) / 3
                right_line_upper_x_pos = (right_line_upper_x_pos + right_line_upper_x_pos_prev * 5) / 6


                right_line_lower_incertitude = abs(right_line_lower_x_pos - right_line_lower_x_pos_prev)
                right_line_lower_incertitude = (right_line_lower_incertitude + right_line_lower_incertitude_prev * 5) / 6


                right_line_upper_incertitude = abs(right_line_upper_x_pos - right_line_upper_x_pos_prev)
                right_line_upper_incertitude = (right_line_upper_incertitude  +  right_line_upper_incertitude_prev * 2) / 3


                lower_right_found = False if right_line_lower_incertitude > incertitude_threshold else lower_right_found
                upper_right_found = False if right_line_upper_incertitude > incertitude_threshold else upper_right_found



                left_line_lower_x_pos_prev  = left_line_lower_x_pos
                left_line_upper_x_pos_prev  = left_line_upper_x_pos

                right_line_lower_x_pos_prev = right_line_lower_x_pos
                right_line_upper_x_pos_prev = right_line_upper_x_pos


                left_line_lower_incertitude_prev  = left_line_lower_incertitude
                left_line_upper_incertitude_prev  = left_line_upper_incertitude

                right_line_lower_incertitude_prev = right_line_lower_incertitude
                right_line_upper_incertitude_prev = right_line_upper_incertitude



                
                ## calculate the true lane width
                min_lane_width = 200
                max_lane_width = 500
                true_lower_lane_width = int(right_line_lower_x_pos - left_line_lower_x_pos) if (lower_left_found and lower_right_found) else int(pre_lower_lane_width)
                true_upper_lane_width = int(right_line_upper_x_pos - left_line_upper_x_pos) if (upper_left_found and upper_right_found) else int(pre_upper_lane_width)

                average_lower_lane_width = (true_lower_lane_width + pre_lower_lane_width*150) / 151 if (true_lower_lane_width > min_lane_width) and (true_lower_lane_width < max_lane_width) else pre_lower_lane_width
                average_upper_lane_width = (true_upper_lane_width + pre_upper_lane_width*150) / 151 if (true_upper_lane_width > min_lane_width) and (true_upper_lane_width < max_lane_width) else pre_upper_lane_width
                
                pre_lower_lane_width = average_lower_lane_width
                pre_upper_lane_width = average_upper_lane_width

                
                ## predict the lines using the average lane width
                predicted_lower_left_line_x_pos  = right_line_lower_x_pos - average_lower_lane_width
                predicted_lower_right_line_x_pos = left_line_lower_x_pos  + average_lower_lane_width

                predicted_upper_left_line_x_pos  = right_line_upper_x_pos - average_upper_lane_width
                predicted_upper_right_line_x_pos = left_line_upper_x_pos  + average_upper_lane_width

                
                ## if the lines are not found, use the predicted lines
                left_line_lower_x_pos , lower_left_line_predicted  = (int((predicted_lower_left_line_x_pos  + predicted_lower_left_line_x_pos_prev *5)/6), True) if not lower_left_found  else (int((left_line_lower_x_pos  + predicted_lower_left_line_x_pos_prev *5)/6), False)
                right_line_lower_x_pos, lower_right_line_predicted = (int((predicted_lower_right_line_x_pos + predicted_lower_right_line_x_pos_prev*5)/6), True) if not lower_right_found else (int((right_line_lower_x_pos + predicted_lower_right_line_x_pos_prev*5)/6), False)

                left_line_upper_x_pos , upper_left_line_predicted  = (int((predicted_upper_left_line_x_pos  + predicted_upper_left_line_x_pos_prev *5)/6), True) if not upper_left_found  else (int((left_line_upper_x_pos  + predicted_upper_left_line_x_pos_prev *5)/6), False)
                right_line_upper_x_pos, upper_right_line_predicted = (int((predicted_upper_right_line_x_pos + predicted_upper_right_line_x_pos_prev*5)/6), True) if not upper_right_found else (int((right_line_upper_x_pos + predicted_upper_right_line_x_pos_prev*5)/6), False)

                predicted_lower_left_line_x_pos_prev  = left_line_lower_x_pos
                predicted_lower_right_line_x_pos_prev = right_line_lower_x_pos

                predicted_upper_left_line_x_pos_prev = left_line_upper_x_pos
                predicted_upper_right_line_x_pos_prev = right_line_upper_x_pos

                ## clamp the lines
                center_threshold = 5

                left_line_lower_x_pos  = np.clip(left_line_lower_x_pos  ,           0              , width/2 - center_threshold)
                right_line_lower_x_pos = np.clip(right_line_lower_x_pos ,width/2 + center_threshold,            width          )

                predicted_lower_left_line_x_pos  = np.clip(predicted_lower_left_line_x_pos  , 0, width/2 - center_threshold    )
                predicted_lower_right_line_x_pos = np.clip(predicted_lower_right_line_x_pos , width/2 + center_threshold, width)


                ## check for lane departure
                left_line_departure, right_line_departure = lane_departure(left_line_lower_x_pos, lower_left_found , lower_left_line_predicted, right_line_lower_x_pos, lower_right_found, lower_right_line_predicted)

                if left_line_departure and right_line_departure:
                    left_line_departure = 0
                    right_line_departure = 0


                lane_departure_amplifier = 1.2 if (left_line_departure or right_line_departure) else 1



                ## determine the lane position
                lane_lower_position = ((left_line_lower_x_pos + right_line_lower_x_pos + width/2) / 3)
                lane_keeping_angle = np.arctan((lane_lower_position - width/2) / height)

                lane_keeping_angle = ((lane_keeping_angle + lane_keeping_angle_prev*5) / 6) * lane_departure_amplifier
                lane_keeping_angle_prev = lane_keeping_angle

                

                ## if none of the lines are found, use the last coherent angle
                if lower_left_found and lower_right_found:
                    last_lane_keeping_angle = lane_keeping_angle
                    lane_keeping = True
                elif lower_left_found or lower_right_found:
                    lane_keeping = True
                elif not lower_left_found and not lower_right_found:
                    lane_keeping_angle = last_lane_keeping_angle
                    lane_keeping = False



                ## lane keeping PID
                Kp = 0.5
                Ki = 0.01
                Kd = 0.1
                error = (left_line_lower_x_pos + right_line_lower_x_pos)/2 - width/2
                
                lane_keeping_P = error*Kp
                lane_keeping_I += error*Ki
                lane_keeping_D = (error - error_prev)*Kd

                pid_correction = (lane_keeping_P + lane_keeping_I + lane_keeping_D)*0.01

                print(f"error = {error:.5f}, P = {lane_keeping_P:.5f}, I = {lane_keeping_I:.5f}, D = {lane_keeping_D:.5f}, correction = {pid_correction:.5f}")
                error_prev = error




                if lower_left_line_predicted and not lower_right_line_predicted:
                    left_lower_color = AVAILABLE_COLOR[1]
                elif lower_left_found:
                    left_lower_color = AVAILABLE_COLOR[0]
                else:
                    left_lower_color = UNAVAILABLE_COLOR[0]
                if left_line_departure:
                    left_lower_color = ALERTE_DEPARTURE_COLOR[0]


                if upper_left_line_predicted and not upper_right_line_predicted:
                    left_upper_color = AVAILABLE_COLOR[1]
                elif upper_left_found:
                    left_upper_color = AVAILABLE_COLOR[0]
                else:
                    left_upper_color = UNAVAILABLE_COLOR[0]
                



                if lower_right_line_predicted and not lower_left_line_predicted:
                    right_lower_color = AVAILABLE_COLOR[1]
                elif lower_right_found:
                    right_lower_color = AVAILABLE_COLOR[0]
                else:
                    right_lower_color = UNAVAILABLE_COLOR[0]
                if right_line_departure:
                    right_lower_color = ALERTE_DEPARTURE_COLOR[0]

                
                if upper_right_line_predicted and not upper_left_line_predicted:
                    right_upper_color = AVAILABLE_COLOR[1]
                elif upper_right_found:
                    right_upper_color = AVAILABLE_COLOR[0]
                else:
                    right_upper_color = UNAVAILABLE_COLOR[0]



                ## smooth left and right line color transition
                b, g, r = left_lower_color
                pre_b, pre_g, pre_r = left_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                left_lower_color = (b, g, r)
                left_color_prev = left_lower_color

                b, g, r = right_lower_color
                pre_b, pre_g, pre_r = right_color_prev
                
                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                right_lower_color = (b, g, r)
                right_color_prev = right_lower_color



                if lane_keeping:
                    lane_color = AVAILABLE_COLOR[0]
                else:
                    lane_color = UNAVAILABLE_COLOR[0]

                b, g, r = lane_color
                pre_b, pre_g, pre_r = lane_color_prev

                b = int((b + pre_b*10) / 11)
                g = int((g + pre_g*10) / 11)
                r = int((r + pre_r*10) / 11)

                lane_color = (b, g, r)
                lane_color_prev = lane_color

                ## lane following     
                lane_upper_position = ((left_line_upper_x_pos + right_line_upper_x_pos) / 2)

                
                ## calculate the angle of the left line knowing x1 = left_line_x_pos, x2 = left_line_upper_x_pos, and the 2 point have detection_distance of seperation in y
                lane_following_incertitude_threshold = 2

                left_line_angle = np.arctan((left_line_lower_x_pos  -  left_line_upper_x_pos) / detection_distance)
                left_line_angle = (left_line_angle  +  left_line_angle_prev*5) / 6
            
                left_line_accurate = True if left_line_upper_incertitude < lane_following_incertitude_threshold else False


                right_line_angle = np.arctan((right_line_lower_x_pos - right_line_upper_x_pos) / detection_distance)
                right_line_angle = (right_line_angle + right_line_angle_prev*5) / 6

                right_line_accurate = True if right_line_upper_incertitude < lane_following_incertitude_threshold else False


                ## calculate the lane angle with both line angle, tho only if that line if detected or predicted
                left_good  = int((lower_left_found  or lower_left_line_predicted)  and left_line_accurate)
                right_good = int((lower_right_found or lower_right_line_predicted) and right_line_accurate)

                lane_angle = (left_line_angle*left_good + right_line_angle*right_good) / (left_good + right_good) if (left_good or right_good) else lane_angle
                last_good_lane_angle = lane_angle if (left_good and right_good) else last_good_lane_angle

                lane_angle = last_good_lane_angle if not(left_good or right_good) else lane_angle
                lane_angle = (lane_angle + lane_angle_prev*5) / 6
                lane_angle_prev = lane_angle


                ## print for both lane if they are detected or predicted
                # print(f"steering_angle: {int(steering_angle): =03d}°     ||     left: lower: {int(width/2-left_line_lower_x_pos): =03d}  ({int(lower_left_found)})   upper: {int(width/2-left_line_upper_x_pos): =03d}  ({int(upper_left_found)})    |   right: lower: {int(right_line_lower_x_pos-width/2): =03d}  ({int(lower_right_found)})   upper: {int(right_line_upper_x_pos-width/2): =03d}  ({int(upper_right_found)})     ||     lane width: lower: {average_lower_lane_width:.2f}  upper: {average_upper_lane_width:.2f}")
                
                ## draw the predicted lines
                detecting_height = 20
                """their foreground"""
                cv.line(corected_frame, (int(predicted_lower_left_line_x_pos ), height-detecting_height), (int(predicted_lower_left_line_x_pos ), height                 ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                         /2), height                 ), (int(predicted_lower_left_line_x_pos ), height-detecting_height), (255, 0, 0), 2)
                cv.line(corected_frame, (int(predicted_lower_right_line_x_pos), height-detecting_height), (int(predicted_lower_right_line_x_pos), height                 ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                         /2), height                 ), (int(predicted_lower_right_line_x_pos), height-detecting_height), (255, 0, 0), 2)

                cv.line(corected_frame, (int(predicted_upper_left_line_x_pos ), detecting_height), (int(predicted_upper_left_line_x_pos ), 0               ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                         /2), 0               ), (int(predicted_upper_left_line_x_pos ), detecting_height), (255, 0, 0), 2)
                cv.line(corected_frame, (int(predicted_upper_right_line_x_pos), detecting_height), (int(predicted_upper_right_line_x_pos), 0               ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                         /2), 0               ), (int(predicted_upper_right_line_x_pos), detecting_height), (255, 0, 0), 2)


                
                ## draw the lines

                ## drawing the line representing the lane
                ## their background
                detecting_height = 40
                cv.line(corected_frame, (int(left_line_lower_x_pos ), height-detecting_height), (int(left_line_lower_x_pos ), height                 ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width               /2), height                 ), (int(left_line_lower_x_pos ), height-detecting_height), (0, 0, 0), 6)
                cv.line(corected_frame, (int(right_line_lower_x_pos), height-detecting_height), (int(right_line_lower_x_pos), height                 ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width               /2), height                 ), (int(right_line_lower_x_pos), height-detecting_height), (0, 0, 0), 6)

                cv.line(corected_frame, (int(left_line_upper_x_pos   ), detecting_height), (int(left_line_upper_x_pos  ), 0               ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width                 /2), 0               ), (int(left_line_upper_x_pos  ), detecting_height), (0, 0, 0), 6)
                cv.line(corected_frame, (int(right_line_upper_x_pos  ), detecting_height), (int(right_line_upper_x_pos ), 0               ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width                 /2), 0               ), (int(right_line_upper_x_pos ), detecting_height), (0, 0, 0), 6)

                ## lines (from lower to upper)
                cv.line(corected_frame, (int(left_line_lower_x_pos ), height), (int(left_line_upper_x_pos ), 0), (0, 0, 0), 8)
                cv.line(corected_frame, (int(right_line_lower_x_pos), height), (int(right_line_upper_x_pos), 0), (0, 0, 0), 8)


                ## draw the lines connected to the 4 points used for the perspective transform
                # cv.line(displayed_frame, upper_left  , upper_right , (0, 0, 255), 2)
                # cv.line(displayed_frame, upper_right , lower_right , (0, 0, 255), 2)
                # cv.line(displayed_frame, lower_right , lower_left  , (0, 0, 255), 2)
                # cv.line(displayed_frame, lower_left  , upper_left  , (0, 0, 255), 2)



                
                ## draw the lines on displayed_frame taking into account the perspective change
                ## map the x position from (0 to width) to (lower_left to lower_right)
                displayed_frame_left_lower_line_x_pos   = map(left_line_lower_x_pos , 0, width, lower_left[0], lower_right[0])
                displayed_frame_right_lower_line_x_pos  = map(right_line_lower_x_pos, 0, width, lower_left[0], lower_right[0])

                displayed_frame_left_upper_line_x_pos    = map(left_line_upper_x_pos , 0, width, upper_left[0], upper_right[0])
                displayed_frame_right_upper_line_x_pos   = map(right_line_upper_x_pos, 0, width, upper_left[0], upper_right[0])

                ## mask of the lane according to the left and right lines
                lane_msk = np.zeros_like(displayed_frame)
                cv.fillPoly(lane_msk, np.array([[(int(displayed_frame_left_lower_line_x_pos  ), height-car_hood), (int(displayed_frame_left_upper_line_x_pos  ), height-car_hood-detection_distance), (int(displayed_frame_right_upper_line_x_pos ), height-car_hood-detection_distance), (int(displayed_frame_right_lower_line_x_pos ), height-car_hood)]], dtype=np.int32), steering_color)
                displayed_frame = cv.addWeighted(displayed_frame, 1, lane_msk, 0.5, 0)


                displayed_frame_center_lower_line_x_pos = map(lane_lower_position   , 0, width, lower_left[0], lower_right[0])
                displayed_frame_center_upper_line_x_pos = map(int((left_line_upper_x_pos + right_line_upper_x_pos)/2)   , 0, width, upper_left[0], upper_right[0])

                ## display the lines
                ## their background
                # cv.line(displayed_frame, (int(displayed_frame_left_lower_line_x_pos  ), height-car_hood), (int(displayed_frame_left_upper_line_x_pos  ), height-car_hood-detection_distance), (0, 0, 0), 3)
                # cv.line(displayed_frame, (int(displayed_frame_right_lower_line_x_pos ), height-car_hood), (int(displayed_frame_right_upper_line_x_pos ), height-car_hood-detection_distance), (0, 0, 0), 3)
                # cv.line(displayed_frame, (int(displayed_frame_center_lower_line_x_pos), height-car_hood), (int(displayed_frame_center_upper_line_x_pos), height-car_hood-detection_distance), (0, 0, 0), 3)

                # # ## their foreground
                # cv.line(displayed_frame, (int(displayed_frame_left_lower_line_x_pos  ), height-car_hood), (int(displayed_frame_left_upper_line_x_pos  ), height-car_hood-detection_distance), left_lower_color , 2)
                # cv.line(displayed_frame, (int(displayed_frame_right_lower_line_x_pos ), height-car_hood), (int(displayed_frame_right_upper_line_x_pos ), height-car_hood-detection_distance), right_lower_color, 2)
                # cv.line(displayed_frame, (int(displayed_frame_center_lower_line_x_pos), height-car_hood), (int(displayed_frame_center_upper_line_x_pos), height-car_hood-detection_distance), lane_color , 1)

                



                ## their foreground
                cv.line(corected_frame, ( int(width/2               ) , height                 ), (int(left_line_lower_x_pos ), height-detecting_height), (0,   255,   0  ), 4)
                cv.line(corected_frame, ( int(left_line_lower_x_pos ) , height-detecting_height), (int(left_line_lower_x_pos ), height                 ), left_lower_color , 4)
                cv.line(corected_frame, ( int(width/2               ) , height                 ), (int(right_line_lower_x_pos), height-detecting_height), (0,    0 ,   255), 4)
                cv.line(corected_frame, ( int(right_line_lower_x_pos) , height-detecting_height), (int(right_line_lower_x_pos), height                 ), right_lower_color, 4)

                cv.line(corected_frame, (int(width               /2) , 0               ), (int(left_line_upper_x_pos ), detecting_height), (0,   255,   0  ), 4)
                cv.line(corected_frame, (int(left_line_upper_x_pos ) , detecting_height), (int(left_line_upper_x_pos ), 0               ), left_upper_color , 4)
                cv.line(corected_frame, (int(width               /2) , 0               ), (int(right_line_upper_x_pos), detecting_height), (0,    0 ,   255), 4)
                cv.line(corected_frame, (int(right_line_upper_x_pos) , detecting_height), (int(right_line_upper_x_pos), 0               ), right_upper_color, 4)

                cv.line(corected_frame, (int(left_line_lower_x_pos ) , height), (int(left_line_upper_x_pos ), 0),  left_lower_color, 3)
                cv.line(corected_frame, (int(right_line_lower_x_pos) , height), (int(right_line_upper_x_pos), 0), right_lower_color, 3)


                ## calculate angle in degrees of the line bewteen the displayed_frame_center_lower_line_x_pos and displayed_frame_center_upper_line_x_pos
                center_upper_x_pos = (left_line_upper_x_pos + right_line_upper_x_pos)/2
                center_lower_x_pos = (left_line_lower_x_pos + right_line_lower_x_pos)/2

                cv.line(corected_frame, (int(center_lower_x_pos), height), (int(center_upper_x_pos), 0), (0, 0, 0), 3)
                
                lane_angle_true = np.arctan((center_upper_x_pos - center_lower_x_pos)/height)
                steering_angle_true = lane_angle_true
                # print(steering_angle_true)

                cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(lane_center + np.tan(lane_angle_true)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance*np.cos(lane_angle_true))), (0,0,0), 10)
                cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(lane_center + np.tan(lane_angle_true)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance*np.cos(lane_angle_true))), lane_color, 5)

                
                ## add legend for the vertical lines' colors (grey if the line is not found, white if it is found, blue if it is predicted)
                # cv.putText(displayed_frame, "Color legend:"                 , (10, height-90),  cv.FONT_HERSHEY_SIMPLEX, 0.45, (0   , 0   , 0   ), 3)
                # cv.putText(displayed_frame, "Color legend:"                 , (10, height-90),  cv.FONT_HERSHEY_SIMPLEX, 0.45, AVAILABLE_COLOR[0], 1)

                # cv.putText(displayed_frame, "- Grey : The line is not found", (25, height-70 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0   , 0   , 0   )  , 3)
                # cv.putText(displayed_frame, "- Grey : The line is not found", (25, height-70 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, UNAVAILABLE_COLOR[0], 1)

                # cv.putText(displayed_frame, "- White: The line is found"    , (25, height-50 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0   , 0   , 0   ), 3)
                # cv.putText(displayed_frame, "- White: The line is found"    , (25, height-50 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, AVAILABLE_COLOR[0], 1)

                # cv.putText(displayed_frame, "- Blue : The line is predicted", (25, height-30 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0   , 0   , 0   ), 3)
                # cv.putText(displayed_frame, "- Blue : The line is predicted", (25, height-30 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, AVAILABLE_COLOR[1], 1)

                # cv.putText(displayed_frame, "- Red  : Lane departure"       , (25, height-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0   , 0   , 0   )       , 3)
                # cv.putText(displayed_frame, "- Red  : Lane departure"       , (25, height-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.45, ALERTE_DEPARTURE_COLOR[0], 1)
                    

                wheel_angle =( np.degrees(lane_keeping_angle)*0.5 + np.degrees(steering_angle_true))

                steering_angle = map(wheel_angle, -25, 25, -30, 30)

                joystick_pos = controller_input.steering_angle_to_joystick_pos(steering_angle)

                # print(f"wheel angle: {int(wheel_angle): 3d}°  -  steering wheel angle: {int(steering_angle_true): =4d}°   |   joystick position: {' ' if (joystick_pos > 0) else '-'} {abs(joystick_pos)/32767:.2f}")


                if lane_color[0] >= 240:
                    autopilot_available = True
                elif lane_color[0] <= 130:
                    autopilot_available = False

           

                if autopilot_available:
                    steering_wheel_color = AVAILABLE_COLOR[0]
                    steering_color = AVAILABLE_COLOR[1]
                else:
                    steering_wheel_color = UNAVAILABLE_COLOR[0]
                    steering_color = UNAVAILABLE_COLOR[1]

                if left_line_departure or right_line_departure:
                    steering_color = ALERTE_DEPARTURE_COLOR[0]
                    steering_wheel_color = AVAILABLE_COLOR[0]

                b    , g    , r     = steering_wheel_color
                pre_b, pre_g, pre_r = steering_wheel_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                steering_wheel_color      = (b, g, r)
                steering_wheel_color_prev = (b, g, r)



                b     , g    , r     = steering_color
                pre_b , pre_g, pre_r = steering_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                steering_color      = (b, g, r)
                steering_color_prev = (b, g, r)



                
                ## steering wheel (rotate with the steering angle) (made with 1 circle and 2 lines)
                steering_wheel_size   = 35
                steering_wheel_center = (int(width/2), int(height/3))
                cv.circle(displayed_frame, steering_wheel_center, steering_wheel_size + int(steering_wheel_size/2 ),    steering_color   ,                         -1)
                cv.circle(displayed_frame, steering_wheel_center, steering_wheel_size + int(steering_wheel_size/10), steering_wheel_color, int(steering_wheel_size/7))
                cv.circle(displayed_frame, (int(steering_wheel_center[0] - (steering_wheel_size/6.5)*math.sin(math.radians(steering_angle))), int(steering_wheel_center[1] + (steering_wheel_size/6.5)*math.cos(math.radians(steering_angle)))), int(steering_wheel_size/2.2), steering_wheel_color, -1)
                
                cv.line(displayed_frame, steering_wheel_center, (int(steering_wheel_center[0]-steering_wheel_size*math.sin(math.radians(steering_angle))), int(steering_wheel_center[1]+steering_wheel_size*math.cos(math.radians(steering_angle)))), steering_wheel_color, int(steering_wheel_size/5))
                steering_wheel_x_lenght = int(steering_wheel_size*math.cos(math.radians(steering_angle)))
                steering_wheel_y_lenght = int(steering_wheel_size*math.sin(math.radians(steering_angle)))
                cv.line(displayed_frame, (steering_wheel_center[0]-steering_wheel_x_lenght, steering_wheel_center[1]-steering_wheel_y_lenght), (steering_wheel_center[0]+steering_wheel_x_lenght, steering_wheel_center[1]+steering_wheel_y_lenght), steering_wheel_color, int(steering_wheel_size/5))

                
                
                ## show FPS
                cv.putText(displayed_frame, f"FPS: {int(fps): =03d} {setting}"  , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0  , 0  , 0  ), 4)
                cv.putText(displayed_frame, f"FPS: {int(fps): =03d} {setting}"  , (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv.putText(displayed_frame, f"AVG FPS: {int(average_fps): =03d}", (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0  , 0  , 0  ), 4)
                cv.putText(displayed_frame, f"AVG FPS: {int(average_fps): =03d}", (5, 35), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            

                cv.imshow('displayed_frame'     , displayed_frame       )
                cv.imshow('corected_frame'      , corected_frame        )
                cv.imshow('bw_canny_frame_frame', bw_canny_frameed_frame)

                

                end_time = time.time()

                cal_time = end_time - start_time
                fps = 1/cal_time
                
                ## make th FPS constant at video_fps
                setting = ""
                if fps > video_fps:
                    time.sleep(1/video_fps - cal_time)
                    setting = f"(caped at {int(video_fps): =03d})"# by video settings)"
                    fps = video_fps
                else:
                    setting = "(max fps)"

                average_fps = (fps + prev_fps*50) / 51
                prev_fps = average_fps
                average_fps = int(average_fps+0.5)

                if cv.waitKey(1) == 27:
                    break


def simulation(angle:float):
    return controller_input.update_joystick(angle)
    


main()