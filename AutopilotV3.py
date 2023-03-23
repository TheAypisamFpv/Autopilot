import cv2 as cv
import numpy as np
import math


AVAILABLE_COLOR = (255, 255, 255), (255, 0, 72)
UNAVAILABLE_COLOR = (100, 100, 100), (0, 0, 0)
autopilot_available = False
autopilot_available_color = UNAVAILABLE_COLOR

height = 1080
width = 1920


lane_center = int(width/2) + 0  # + offset
car_hood = 230
lane_width = 1500
detection_distance = 250
distance_center = lane_center + 10  # + offset
distance_width = 150

upper_left = [int(distance_center - distance_width/2),
              int(height - car_hood - detection_distance)]
upper_right = [int(distance_center + distance_width/2),
               int(height - car_hood - detection_distance)]

lower_left = [int(lane_center - lane_width/2), int(height-car_hood)]
lower_right = [int(lane_center + lane_width/2), int(height-car_hood)]

pts1 = np.float32([upper_left, lower_left, upper_right, lower_right])
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                       (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def detect_lane(frame, displayed_frame):
    """bird's eye view of display_frame"""
    M = cv.getPerspectiveTransform(pts1, pts2)
    displayed_frame = cv.warpPerspective(displayed_frame, M, (width, height))

    """modifiy contrast and brightness"""
    alpha = 2
    beta = -100
    corected_frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    """blur the frame"""
    blur_frame = cv.GaussianBlur(corected_frame, (5, 5), 0)
    """detecet edges"""
    canny_frame = cv.Canny(blur_frame, 150, 250)
    """Black or white frame"""
    bw_frame = cv.threshold(canny_frame, 127, 255, cv.THRESH_BINARY)[1]

    """mask the area that we are interested in"""
    mask = np.zeros_like(bw_frame)
    points = np.array([upper_right, upper_left, lower_left, lower_right], np.int32)
    cv.fillPoly(mask, pts=[points], color=(255, 255, 255))
    masked_frame = cv.bitwise_and(bw_frame, mask)

    """transform the frame to bird's eye view"""
    M = cv.getPerspectiveTransform(pts1, pts2)
    masked_frame = cv.warpPerspective(masked_frame, M, (width, height))


    """find x axis start of lanes"""
    lane_center = int(width/2) + 0  # + offset
    left_line_x_pos = lane_center
    left_found = False

    right_line_x_pos = lane_center
    right_found = False

    for x in range(int(width/2)):
        left_lane_projection = int(width/2-x)
        right_lane_projection = int(width/2+x)

        detect_width = 50

        """check if there is white pixel (+-10 pixels up and down)"""
        if masked_frame[height-detect_width:height, left_lane_projection].any() != 0 and not left_found:
            if left_lane_projection < lane_center-50:
                left_line_x_pos = left_lane_projection
                left_found = True
        
        if masked_frame[height-detect_width:height, right_lane_projection].any() != 0 and not right_found:
            if right_lane_projection > lane_center+50:
                right_line_x_pos = right_lane_projection
                right_found = True

        if left_found and right_found:
            break


    """if either of the lines is not found, make it the same as the other one"""
    if not left_found and right_found:
        left_line_x_pos = int(lane_center - (lane_width/1.75 - (right_line_x_pos - lane_center)))
        focus_lane = 2
    elif not right_found and left_found:
        right_line_x_pos = int(lane_center + (lane_width/1.75 - (lane_center - left_line_x_pos)))
        focus_lane = 1
    elif right_found and left_found:
        focus_lane = 3
    else:
        focus_lane = 0


    lines_width = 75

    """draw the lines"""
    cv.line(displayed_frame, (lane_center, height-5), (lane_center, height-detect_width), (255, 255, 255), 2)


    cv.line(displayed_frame, (lane_center, height-5), (left_line_x_pos, height-5), (255, 255, 255), 2)
    cv.line(displayed_frame, (left_line_x_pos, height-5), (left_line_x_pos, height-detect_width), (255, 255, 255), 2)

    cv.line(displayed_frame, (left_line_x_pos, height-5), (left_line_x_pos-lines_width, height-5), (255, 255, 255), 2)
    cv.line(displayed_frame, (left_line_x_pos-lines_width, height-5), (left_line_x_pos-lines_width, height-detect_width), (255, 255, 255), 2)


    cv.line(displayed_frame, (lane_center, height-5), (right_line_x_pos, height-5), (255, 255, 255), 2)
    cv.line(displayed_frame, (right_line_x_pos, height-5), (right_line_x_pos, height-detect_width), (255, 255, 255), 2)

    cv.line(displayed_frame, (right_line_x_pos, height-5), (right_line_x_pos+lines_width, height-5), (255, 255, 255), 2)
    cv.line(displayed_frame, (right_line_x_pos+lines_width, height-5), (right_line_x_pos+lines_width, height-detect_width), (255, 255, 255), 2)




    return masked_frame, corected_frame, displayed_frame, left_line_x_pos, right_line_x_pos, lines_width, focus_lane

def main():
    """
    lane positions
    """
    left_line_x_pos = 0
    left_line_x_pos_prev = left_line_x_pos

    right_line_x_pos = width
    right_line_x_pos_prev = right_line_x_pos


    """
    lane keeping variables
    """
    lane_keep = 0
    lane_keeping_active = 0
    pre_lane_keeping_active = lane_keeping_active
    pre_lane_keeping = 1
    last_good_lane_keeping = pre_lane_keeping

    autopilot_active = False

    """
    Video here
    """
    cap = cv.VideoCapture('Test drive\\23_03_2023_02.MP4')
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret:
            displayed_frame = frame.copy()
            """
            Display the points on the frame
            """
            # cv.circle(displayed_frame,(upper_left[0], upper_left[1]), 5, (0, 0, 255), -1)
            # cv.circle(displayed_frame,(upper_right[0], upper_right[1]), 5, (0, 0, 255), -1)
            # cv.circle(displayed_frame,(lower_left[0], lower_left[1]), 5, (0, 0, 255), -1)
            # cv.circle(displayed_frame,(lower_right[0], lower_right[1]), 5, (0, 0, 255), -1)
            """draw the lines"""
            # cv.line(displayed_frame, (upper_left[0], upper_left[1]),(lower_left[0], lower_left[1]), (0, 0, 255), 2)
            # cv.line(displayed_frame, (upper_right[0], upper_right[1]), (lower_right[0], lower_right[1]), (0, 0, 255), 2)
            # cv.line(displayed_frame, (upper_left[0], upper_left[1]),(upper_right[0], upper_right[1]), (0, 0, 255), 2)
            # cv.line(displayed_frame, (lower_left[0], lower_left[1]),(lower_right[0], lower_right[1]), (0, 0, 255), 2)
    
            # cv.line(displayed_frame, (distance_center, height - car_hood -detection_distance), (lane_center, height - car_hood), (0, 0, 255), 2)
            

            canny_frame, corected_frame, displayed2_frame, left_line_x_pos, right_line_x_pos, lines_width, focus_lane = detect_lane(frame, displayed_frame)
            """
            smooth the lines
            """
            left_line_x_pos = int((left_line_x_pos + left_line_x_pos_prev*20)/21)
            right_line_x_pos = int((right_line_x_pos + right_line_x_pos_prev*20)/21)

            """
            Lane keeping
            """
            lane_keep -= 0.2 if lane_keep > 0 else 0

            if focus_lane == 1:
                focused_lane = "left"
            elif focus_lane == 2:
                focused_lane = "right"
            elif focus_lane == 3:
                focused_lane = "both"
                lane_keep = 1 
            else:
                focused_lane = "none"

            lane_keep = (lane_keep+pre_lane_keeping_active*10)/11
            pre_lane_keeping_active = lane_keep
            if lane_keep > 0.75:
                lane_keeping_active = True
            elif lane_keep < 0.60:
                lane_keeping_active = False


            left_line_x_pos_prev = left_line_x_pos
            right_line_x_pos_prev = right_line_x_pos

            lane_keeping = ((left_line_x_pos+right_line_x_pos)/2) / (width/2)
            

            if focus_lane == 3:
                last_good_lane_keeping = lane_keeping
            else:
                lane_keeping = last_good_lane_keeping


            lane_keeping = (lane_keeping + pre_lane_keeping*5)/6
            pre_lane_keeping = lane_keeping

            """
            draw the lines
            """
            cv.line(displayed2_frame, (left_line_x_pos-int(lines_width/2), height-5), (left_line_x_pos-int(lines_width/2), height-lines_width*3), (0, 255, 0), 2)
            cv.line(displayed2_frame, (right_line_x_pos+int(lines_width/2), height-5), (right_line_x_pos+int(lines_width/2), height-lines_width*3), (0, 255, 0), 2)

            cv.line(displayed2_frame, (lane_center, height), (int(lane_keeping*(width/2)), height-lines_width*4), (0, 255, 0), 2)

            cv.line(displayed_frame, (lane_center, height), (int(lane_keeping*(width/2)), height-lines_width*4), (0, 0, 0), 10)
            cv.line(displayed_frame, (lane_center, height), (int(lane_keeping*(width/2)), height-lines_width*4), (0, 255, 0), 2)
            
            autopilot_available = lane_keeping_active  # and trajectory_keeping == 3
            if cv.waitKey(1) == 13 and autopilot_available:
                autopilot_active = not autopilot_active



            if autopilot_active == autopilot_available:
                autopilot_available_color = AVAILABLE_COLOR
            else:
                autopilot_available_color = UNAVAILABLE_COLOR

            steering_angle = (lane_keeping-1)*90

            """
            steering wheel (rotate with the steering angle) (made with 1 circle and 2 lines)
            """
            steering_wheel_center = (int(width/4), int(75))
            cv.circle(displayed_frame, steering_wheel_center, 70, autopilot_available_color[1], -1)
            cv.circle(displayed_frame, steering_wheel_center, 50, autopilot_available_color[0], 10)
            
            cv.line(displayed_frame, steering_wheel_center, (int(steering_wheel_center[0]-50*math.sin(math.radians(steering_angle))), int(steering_wheel_center[1]+50*math.cos(math.radians(steering_angle)))), autopilot_available_color[0], 10)
            steering_wheel_x_lenght = int(50*math.cos(math.radians(steering_angle)))
            steering_wheel_y_lenght = int(50*math.sin(math.radians(steering_angle)))
            cv.line(displayed_frame, (steering_wheel_center[0]-steering_wheel_x_lenght, steering_wheel_center[1]-steering_wheel_y_lenght), (steering_wheel_center[0]+steering_wheel_x_lenght, steering_wheel_center[1]+steering_wheel_y_lenght), autopilot_available_color[0], 10)


            print(
                f"focused lane: {focused_lane} (lane_keep: {lane_keeping_active} ({lane_keep:.2f})) : {focus_lane == 3} -> lane keeping: {lane_keeping:.2f}")

            cv.imshow('displayed_frame', displayed_frame)
            cv.imshow('corected_frame', displayed2_frame)
            cv.imshow('canny_frame', canny_frame)

            if cv.waitKey(1) == 27:
                break


main()
