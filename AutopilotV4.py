import cv2 as cv
import numpy as np
import math
import time

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
lane_width         = width
detection_distance = 100
distance_center    = lane_center +5 # + offset
distance_width     = 110

upper_left  = [int(distance_center - distance_width/2),int(height - car_hood - detection_distance)]
upper_right = [int(distance_center + distance_width/2),int(height - car_hood - detection_distance)]

lower_left  = [int(lane_center - lane_width/2), int(height-car_hood)]
lower_right = [int(lane_center + lane_width/2), int(height-car_hood)]

pts1 = np.float32([upper_left, lower_left, upper_right, lower_right])
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

cv.namedWindow('Trackbars')
default_canny_low  = 200
default_canny_high = 225

cv.createTrackbar('Canny_low' , 'Trackbars', default_canny_low , 255, lambda x: None)
cv.createTrackbar('Canny_high', 'Trackbars', default_canny_high, 255, lambda x: None)




def map(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)



def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)), (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def detect_lane_width(frame, displayed_frame):

    # """
    # modifiy contrast and brightness
    # """
    # alpha = 1.0
    # beta = 0
    # corected_frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    corected_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    """
    blur the frame
    """
    blur_frame = cv.GaussianBlur(corected_frame, (51, 51), 0)

    """
    combine the frames
    """
    combined_frame = cv.add(blur_frame, corected_frame)

    """
    bird's eye view of th frame
    """
    M = cv.getPerspectiveTransform(pts1, pts2)
    displayed_frame = cv.cvtColor(combined_frame, cv.COLOR_GRAY2BGR)
    displayed_frame = cv.warpPerspective(displayed_frame, M, (width, height))

    """
    canny edge detection
    """
    canny_low  = default_canny_low
    canny_high = default_canny_high

    canny_low  = cv.getTrackbarPos('Canny_low' , 'Trackbars')
    canny_high = cv.getTrackbarPos('Canny_high', 'Trackbars')

    canny_frame = cv.Canny(combined_frame, canny_low, canny_high)
    canny_frame = cv.warpPerspective(canny_frame, M, (width, height))
    canny_frame = cv.dilate(canny_frame, None, iterations=1)
    canny_blur_frame = cv.GaussianBlur(canny_frame, (3, 71), 0, 0, cv.BORDER_CONSTANT, 0)

    canny_frame = cv.add(canny_frame, canny_blur_frame)

    """
    convert to black and white
    """
    bw_canny_frame = cv.threshold(canny_frame, 7, 255, cv.THRESH_BINARY)[1]


    """
    detect start of the lines
    """
    left_line_x_pos  = (width/3 - lane_width/2)
    left_found = False
    right_line_x_pos = (width/3 + lane_width/2)
    right_found = False

    detecting_height = 40


    center_detecting_ignore = 15

    """
    project a line from the center of the car to the left and right
    """
    for pixel in range(center_detecting_ignore, int(width/2)):
        if bw_canny_frame[height-detecting_height:height, int(width/2 - pixel)].any() > 0 and not left_found:
            left_line_x_pos = width/2 - pixel
            left_found      = True

        if bw_canny_frame[height-detecting_height:height, int(width/2 + pixel)].any() > 0 and not right_found:
            right_line_x_pos = width/2 + pixel
            right_found      = True

        if left_found and right_found:
            break    

    

    

    return displayed_frame, bw_canny_frame, left_line_x_pos, left_found, right_line_x_pos, right_found, car_hood


def lane_departure(left_line_x_pos, left_found, right_line_x_pos, right_found):
    """
    detect lane departure
    """
    left_line  = [int(lane_center       - left_line_x_pos), left_found ]
    right_line = [int(right_line_x_pos  -     lane_center), right_found]

    left_line_departure  = 0
    right_line_departure = 0

    left_line_departure_threshold  = 60
    right_line_departure_threshold = 60

    if left_line[0] < left_line_departure_threshold and left_line[1]:
        left_line_departure = 1
    
    if right_line[0] < right_line_departure_threshold and right_line[1]:
        right_line_departure = 1

    return left_line_departure, right_line_departure



def main():
    """
    lane positions
    """
    left_line_x_pos             = (width/2 - lane_width/2)
    left_line_x_pos_prev        = left_line_x_pos
    predicted_left_line_x_pos   = left_line_x_pos
    left_line_p_x_pos_prev      = left_line_x_pos

    right_line_x_pos            = (width/2 + lane_width/2)
    right_line_x_pos_prev       = right_line_x_pos
    predicted_right_line_x_pos  = right_line_x_pos
    right_line_p_x_pos_prev     = right_line_x_pos

    pre_line_width              = int(lane_width/2)

    left_line_incertitude       = 0
    left_line_incertitude_prev  = 0

    right_line_incertitude      = 0
    right_line_incertitude_prev = 0

    lane_keeping_angle_prev     = 0
    last_lane_keeping_angle     = 0

    left_color_prev             = UNAVAILABLE_COLOR[0]
    right_color_prev            = UNAVAILABLE_COLOR[0]

    lane_color_prev             = UNAVAILABLE_COLOR[0]
    steering_wheel_color_prev   = UNAVAILABLE_COLOR[0]
    steering_color_prev         = UNAVAILABLE_COLOR[0]
    autopilot_available         = False

    setting = ''

    """
    Video here
    """
    cap = cv.VideoCapture('Test drive\\30 03 2023\\3.MP4')
    video_fps = 30#cap.get(cv.CAP_PROP_FPS)

    fps = float(f"{float(video_fps):.2f}")

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_id = cap.get(1)

        if frame_id % 2 or True:
            if ret:
                start_time = time.time()
                frame = cv.resize(frame, (width, height))
                displayed_frame = frame.copy()
                corected_frame, masked_frame, true_left_line_x_pos, left_found, true_right_line_x_pos, right_found, car_hood = detect_lane_width(frame, displayed_frame)


                """
                smooth the lines and calculate the incertitude
                """
                incertitude_threshold = 8

                left_line_x_pos       = (true_left_line_x_pos   +  left_line_x_pos_prev*5) / 6
                left_line_incertitude = np.abs(left_line_x_pos - left_line_x_pos_prev)
                left_line_incertitude = (left_line_incertitude  +   left_line_incertitude_prev*5) / 6
                left_found            = False if left_line_incertitude > incertitude_threshold else left_found

                right_line_x_pos       = (true_right_line_x_pos + right_line_x_pos_prev*5) / 6
                right_line_incertitude = np.abs(right_line_x_pos - right_line_x_pos_prev)
                right_line_incertitude = (right_line_incertitude + right_line_incertitude_prev*5) / 6
                right_found            = False if right_line_incertitude > incertitude_threshold else right_found


                left_line_x_pos_prev        = left_line_x_pos
                right_line_x_pos_prev       = right_line_x_pos
                left_line_incertitude_prev  = left_line_incertitude
                right_line_incertitude_prev = right_line_incertitude


                """
                calculate the true lane width
                """
                true_lane_width = int(right_line_x_pos - left_line_x_pos) if (left_found and right_found) else int(pre_line_width)

                average_lane_width = (true_lane_width + pre_line_width*100) / 101
                pre_line_width = average_lane_width

                """
                predict the lines using the average lane width
                """
                predicted_left_line_x_pos  = right_line_x_pos - average_lane_width
                predicted_right_line_x_pos = left_line_x_pos  + average_lane_width

                """
                if the lines are not found, use the predicted lines
                """
                left_line_x_pos , left_line_predicted  = (int((predicted_left_line_x_pos  + left_line_p_x_pos_prev *5)/6), True) if not left_found  else (int((left_line_x_pos  + left_line_p_x_pos_prev *5)/6), False)
                right_line_x_pos, right_line_predicted = (int((predicted_right_line_x_pos + right_line_p_x_pos_prev*5)/6), True) if not right_found else (int((right_line_x_pos + right_line_p_x_pos_prev*5)/6), False)

                left_line_p_x_pos_prev  = left_line_x_pos
                right_line_p_x_pos_prev = right_line_x_pos

                """
                clamp the lines
                """
                center_threshold = 20

                left_line_x_pos            = np.clip(left_line_x_pos            , 0, width/2 - center_threshold       )
                right_line_x_pos           = np.clip(right_line_x_pos           ,    width/2 + center_threshold, width)

                predicted_left_line_x_pos  = np.clip(predicted_left_line_x_pos  , 0, width/2 - center_threshold       )
                predicted_right_line_x_pos = np.clip(predicted_right_line_x_pos ,    width/2 + center_threshold, width)


                """
                check for lane departure
                """
                left_line_departure, right_line_departure = lane_departure(left_line_x_pos, (left_found or left_line_predicted), right_line_x_pos, (right_found or right_line_predicted))

                if left_line_departure and right_line_departure:
                    left_line_departure = 0
                    right_line_departure = 0


                lane_departure_amplifier = 1.1 if (left_line_departure or right_line_departure) else 1



                """
                determine the lane position
                """
                lane_position = ((left_line_x_pos + right_line_x_pos + width/2) / 3)
                lane_keeping_angle = np.arctan((lane_position - width/2) / height)
                lane_keeping_angle = ((lane_keeping_angle + lane_keeping_angle_prev*5) / 6) * lane_departure_amplifier
                lane_keeping_angle_prev = lane_keeping_angle

                

                """
                if none of the lines are found, use the last coherent angle
                """
                if left_found and right_found:
                    last_lane_keeping_angle = lane_keeping_angle
                    lane_keeping = True
                elif left_found or right_found:
                    lane_keeping = True
                elif not left_found and not right_found:
                    lane_keeping_angle = last_lane_keeping_angle
                    lane_keeping = False


                if left_line_predicted and not right_line_predicted:
                    left_color = AVAILABLE_COLOR[1]
                elif left_found:
                    left_color = AVAILABLE_COLOR[0]
                else:
                    left_color = UNAVAILABLE_COLOR[0]
                if left_line_departure:
                    left_color = ALERTE_DEPARTURE_COLOR[0]

                

                if right_line_predicted and not left_line_predicted:
                    right_color = AVAILABLE_COLOR[1]
                elif right_found:
                    right_color = AVAILABLE_COLOR[0]
                else:
                    right_color = UNAVAILABLE_COLOR[0]
                if right_line_departure:
                    right_color = ALERTE_DEPARTURE_COLOR[0]

                """
                smooth left and right line color transition
                """
                b, g, r = left_color
                pre_b, pre_g, pre_r = left_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                left_color = (b, g, r)
                left_color_prev = left_color

                b, g, r = right_color
                pre_b, pre_g, pre_r = right_color_prev
                
                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                right_color = (b, g, r)
                right_color_prev = right_color



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

                

                # print(f"left_line_incertitude: {int(left_line_incertitude): =04d}  |  right_line_incertitude: {int(right_line_incertitude): =04d}  |  lane width: {true_lane_width: =04d}  |  average lane width: {int(average_lane_width): =04d}  |  left_found: {int(left_found)}  |  right_found: {int(right_found)}  |  Left_departure = {left_line_departure}  right_departure = {right_line_departure}  |  frame_id: {int(frame_id): =05d}  |  fps = {fps:.2f}")

                """
                draw the predicted lines
                """
                detecting_height = 20
                """their foreground"""
                cv.line(corected_frame, (int(predicted_left_line_x_pos   ), height-detecting_height), (int(predicted_left_line_x_pos ), height                 ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                     /2), height                 ), (int(predicted_left_line_x_pos ), height-detecting_height), (255, 0, 0), 2)
                cv.line(corected_frame, (int(predicted_right_line_x_pos  ), height-detecting_height), (int(predicted_right_line_x_pos), height                 ), (255, 0, 0), 2)
                cv.line(corected_frame, (int(width                     /2), height                 ), (int(predicted_right_line_x_pos), height-detecting_height), (255, 0, 0), 2)


                """
                draw the lines
                """

                """
                drawing the line representing the lane
                their background
                """
                detecting_height = 40
                cv.line(corected_frame, (int(left_line_x_pos   ), height-detecting_height), (int(left_line_x_pos ), height                 ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width           /2), height                 ), (int(left_line_x_pos ), height-detecting_height), (0, 0, 0), 6)
                cv.line(corected_frame, (int(right_line_x_pos  ), height-detecting_height), (int(right_line_x_pos), height                 ), (0, 0, 0), 6)
                cv.line(corected_frame, (int(width           /2), height                 ), (int(right_line_x_pos), height-detecting_height), (0, 0, 0), 6)

                # """
                # draw the lines connected to the 4 points used for the perspective transform
                # """
                # cv.line(displayed_frame, upper_left, upper_right, (0, 0, 255), 2)
                # cv.line(displayed_frame, upper_right, lower_right, (0, 0, 255), 2)
                # cv.line(displayed_frame, lower_right, lower_left, (0, 0, 255), 2)
                # cv.line(displayed_frame, lower_left, upper_left, (0, 0, 255), 2)



                """
                draw the lines on displayed_frame taking into account the perspective change

                map the x position from (0 to width) to (lower_left to lower_right)
                """
                displayed_frame_left_lower_line_x_pos   = map(left_line_x_pos , 0, width, lower_left[0], lower_right[0])
                displayed_frame_right_lower_line_x_pos  = map(right_line_x_pos, 0, width, lower_left[0], lower_right[0])

                displayed_frame_left_high_line_x_pos    = map(left_line_x_pos , 0, width, upper_left[0], upper_right[0])
                displayed_frame_right_high_line_x_pos   = map(right_line_x_pos, 0, width, upper_left[0], upper_right[0])

                displayed_frame_center_lower_line_x_pos = map(lane_position   , 0, width, lower_left[0], lower_right[0])
                displayed_frame_center_high_line_x_pos  = map(lane_position   , 0, width, upper_left[0], upper_right[0])

                """their background"""
                cv.line(displayed_frame, (int(displayed_frame_left_lower_line_x_pos  ), height-car_hood), (int(displayed_frame_left_high_line_x_pos  ), height-car_hood-detection_distance), (0, 0, 0), 7)
                cv.line(displayed_frame, (int(displayed_frame_right_lower_line_x_pos ), height-car_hood), (int(displayed_frame_right_high_line_x_pos ), height-car_hood-detection_distance), (0, 0, 0), 7)
                
                cv.line(displayed_frame, (int(displayed_frame_center_lower_line_x_pos), height-car_hood), (int(displayed_frame_center_high_line_x_pos), height-car_hood-detection_distance), (0, 0, 0), 5)

                """their foreground"""
                cv.line(displayed_frame, (int(displayed_frame_left_lower_line_x_pos  ), height-car_hood), (int(displayed_frame_left_high_line_x_pos  ), height-car_hood-detection_distance), left_color , 3)
                cv.line(displayed_frame, (int(displayed_frame_right_lower_line_x_pos ), height-car_hood), (int(displayed_frame_right_high_line_x_pos ), height-car_hood-detection_distance), right_color, 3)

                cv.line(displayed_frame, (int(displayed_frame_center_lower_line_x_pos), height-car_hood), (int(displayed_frame_center_high_line_x_pos), height-car_hood-detection_distance), lane_color , 2)



                # """
                # lane keeping (using lane position and the angle from the bottom center)
                # and display a line on the screen
                # """
                # cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(distance_center + np.tan(lane_keeping_angle)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance/2)), (0,0,0), 14)
                # cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(distance_center + np.tan(lane_keeping_angle)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance/2)), lane_color, 10)


                """their foreground"""
                cv.line(corected_frame, (int(width/2        ), height                 ), (int(left_line_x_pos), height-detecting_height), (0, 255, 0), 4)
                cv.line(corected_frame, (int(left_line_x_pos), height-detecting_height), (int(left_line_x_pos), height                 ), left_color , 4)


                cv.line(corected_frame, (int(width/2         ), height                 ), (int(right_line_x_pos), height-detecting_height), (0, 0, 255), 4)
                cv.line(corected_frame, (int(right_line_x_pos), height-detecting_height), (int(right_line_x_pos), height                 ), right_color, 4)


                cv.putText(displayed_frame, "Lane Keeping Only", (int(width/2)-75, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0  , 0  , 0  ), 3)
                cv.putText(displayed_frame, "Lane Keeping Only", (int(width/2)-75, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


                """
                add legend for the vertical lines' colors (grey if the line is not found, white if it is found, blue if it is predicted)
                """
                cv.putText(displayed_frame, "Color legend:"                 , (10, height-70),  cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0)           , 3)
                cv.putText(displayed_frame, "Color legend:"                 , (10, height-70),  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255 , 255, 255)    , 1)

                cv.putText(displayed_frame, "- Grey : The line is not found", (25, height-50 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0)           , 3)
                cv.putText(displayed_frame, "- Grey : The line is not found", (25, height-50 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, UNAVAILABLE_COLOR[0], 1)
  
                cv.putText(displayed_frame, "- White: The line is found"    , (25, height-30 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0)           , 3)
                cv.putText(displayed_frame, "- White: The line is found"    , (25, height-30 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, AVAILABLE_COLOR[0]  , 1)
  
                cv.putText(displayed_frame, "- Blue : The line is predicted", (25, height-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0)           , 3)
                cv.putText(displayed_frame, "- Blue : The line is predicted", (25, height-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.4, AVAILABLE_COLOR[1]  , 1)
                    
        


                steering_angle = np.degrees(lane_keeping_angle)

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
                    steering_wheel_color = ALERTE_DEPARTURE_COLOR[0]
                    steering_color = AVAILABLE_COLOR[0]

                b, g, r = steering_wheel_color
                pre_b, pre_g, pre_r = steering_wheel_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                steering_wheel_color = (b, g, r)
                steering_wheel_color_prev = (b, g, r)

                b, g, r = steering_color
                pre_b, pre_g, pre_r = steering_color_prev

                b = int((b + pre_b*5) / 6)
                g = int((g + pre_g*5) / 6)
                r = int((r + pre_r*5) / 6)

                steering_color = (b, g, r)
                steering_color_prev = (b, g, r)



                """
                steering wheel (rotate with the steering angle) (made with 1 circle and 2 lines)
                """
                steering_wheel_size = 25
                steering_wheel_center = (75, 75)
                cv.circle(displayed_frame, steering_wheel_center, steering_wheel_size + 10, steering_color, -1)
                cv.circle(displayed_frame, steering_wheel_center, steering_wheel_size+int(steering_wheel_size/10), steering_wheel_color, int(steering_wheel_size/7))
                cv.circle(displayed_frame, (int(steering_wheel_center[0] - (steering_wheel_size/6.5)*math.sin(math.radians(steering_angle))), int(steering_wheel_center[1] + (steering_wheel_size/6.5)*math.cos(math.radians(steering_angle)))), int(steering_wheel_size/2.2), steering_wheel_color, -1)
                
                cv.line(displayed_frame, steering_wheel_center, (int(steering_wheel_center[0]-steering_wheel_size*math.sin(math.radians(steering_angle))), int(steering_wheel_center[1]+steering_wheel_size*math.cos(math.radians(steering_angle)))), steering_wheel_color, int(steering_wheel_size/5))
                steering_wheel_x_lenght = int(steering_wheel_size*math.cos(math.radians(steering_angle)))
                steering_wheel_y_lenght = int(steering_wheel_size*math.sin(math.radians(steering_angle)))
                cv.line(displayed_frame, (steering_wheel_center[0]-steering_wheel_x_lenght, steering_wheel_center[1]-steering_wheel_y_lenght), (steering_wheel_center[0]+steering_wheel_x_lenght, steering_wheel_center[1]+steering_wheel_y_lenght), steering_wheel_color, int(steering_wheel_size/5))

                
                """
                show FPS
                """
                cv.putText(displayed_frame, f"FPS: {fps:.2f} {setting}", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0  , 0  , 0  ), 3)
                cv.putText(displayed_frame, f"FPS: {fps:.2f} {setting}", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


                cv.imshow('displayed_frame', displayed_frame)
                cv.imshow('corected_frame', corected_frame)
                cv.imshow('masked_frame', masked_frame)

                end_time = time.time()

                cal_time = end_time - start_time
                """
                make th FPS constant at video_fps
                """
                setting = ""
                # if cal_time < 1/video_fps:
                #     time.sleep(1/video_fps - cal_time)
                #     cal_time = 1/video_fps
                #     setting = f"(caped at {float(video_fps):.2f} by video settings)"

                fps = 1/cal_time


                if cv.waitKey(1) == 27:
                    break


main()
