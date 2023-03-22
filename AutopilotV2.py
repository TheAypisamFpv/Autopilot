import cv2 as cv
import numpy as np
import math


height = 720
width = 1280

center = int(width/2)
car_hood = 150
car_width = 900
distance_center = int(width/2)
detection_distance = 120
distance_width = 130




left_detect_lane_pos = 330
right_detect_lane_pos = 1022

lane_detect_width = 100
lane_detect_height = 250

left_look_at_upper_pt1 = [left_detect_lane_pos - lane_detect_width, height - lane_detect_height]
left_look_at_upper_pt2 = [left_detect_lane_pos + lane_detect_width, height - int(lane_detect_height/2)]

left_look_at_lower_pt1 = [left_detect_lane_pos - lane_detect_width, height - int(lane_detect_height/2)]
left_look_at_lower_pt2 = [left_detect_lane_pos + lane_detect_width, height]


right_look_at_upper_pt1 = [right_detect_lane_pos - lane_detect_width, height - lane_detect_height]
right_look_at_upper_pt2 = [right_detect_lane_pos + lane_detect_width, height - int(lane_detect_height/2)]

right_look_at_lower_pt1 = [right_detect_lane_pos - lane_detect_width, height - int(lane_detect_height/2)]
right_look_at_lower_pt2 = [right_detect_lane_pos + lane_detect_width, height]




def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2





def detect_lines(frame):
    """detect left and right lane by looking at white pixels"""
    left_lane = []
    right_lane = []



    """look at white pixel around left_look_at"""
    thresh = 200
    frame_wb = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)[1]

    left_upper_bbox = left_look_at_upper_pt1 + left_look_at_upper_pt2
    left_lower_bbox = left_look_at_lower_pt1 + left_look_at_lower_pt2

    right_upper_bbox = right_look_at_upper_pt1 + right_look_at_upper_pt2
    right_lower_bbox = right_look_at_lower_pt1 + right_look_at_lower_pt2

    left_upper_wb = imcrop(frame_wb, left_upper_bbox)
    left_lower_wb = imcrop(frame_wb, left_lower_bbox)

    right_upper_wb = imcrop(frame_wb, right_upper_bbox)
    right_lower_wb = imcrop(frame_wb, right_lower_bbox)


    """
    left lane angle calcuation
    """

    left_upper = cv.moments(left_upper_wb)
    left_lower = cv.moments(left_lower_wb)

    if left_upper["m00"] != 0:
        left_upper_pos = int(left_upper["m10"] / left_upper["m00"]), int(left_upper["m01"] / left_upper["m00"]) + lane_detect_height
    else:
        left_upper_pos = 0, 0

    if left_lower["m00"] != 0:
        left_lower_pos = int(left_lower["m10"] / left_lower["m00"]), int(left_lower["m01"] / left_lower["m00"])
    else:
        left_lower_pos = 0, 0

    left_pos = left_upper_pos[0] - left_lower_pos[0], left_upper_pos[1] - left_lower_pos[1]

    left_angle = math.atan2(left_pos[0], left_pos[1]) * 180 / math.pi
    if abs(left_angle) > 45:
        left_angle = 0
        left_detected = 0
    else:
        left_detected = 1


    """
    right lane angle calcuation
    """

    right_upper = cv.moments(right_upper_wb)
    right_lower = cv.moments(right_lower_wb)

    if right_upper["m00"] != 0:
        right_upper_pos = int(right_upper["m10"] / right_upper["m00"]), int(right_upper["m01"] / right_upper["m00"]) + lane_detect_height
    else:
        right_upper_pos = 0, 0

    if right_lower["m00"] != 0:
        right_lower_pos = int(right_lower["m10"] / right_lower["m00"]), int(right_lower["m01"] / right_lower["m00"])
    else:
        right_lower_pos = 0, 0

    right_pos = right_upper_pos[0] - right_lower_pos[0], right_upper_pos[1] - right_lower_pos[1]

    right_angle = math.atan2(right_pos[0], right_pos[1]) * 180 / math.pi
    if abs(right_angle) > 45:
        right_angle = 0
        right_detected = 0
    else:
        right_detected = 1

    
    """
    detected: 0:None , 1:left, 2:right, 3,both
    """

    if right_detected == left_detected:
        if right_detected == 0:
            # print("no lane detected")
            detected = 0
        else:
            # print("both lane detected")
            detected = 3
    else:
        if right_detected == 1:
            # print("right lane detected")
            detected = 2
        else:
            # print("left lane detected")
            detected = 1


    # print(f"lanes angle: {left_angle}° -- {right_angle}°")


    displayed_wb = frame_wb.copy()
    cv.rectangle(displayed_wb, left_look_at_upper_pt1, left_look_at_upper_pt2, (255, 255, 255), 5)
    cv.rectangle(displayed_wb, left_look_at_lower_pt1, left_look_at_lower_pt2, (255, 255, 255), 5)
    cv.rectangle(displayed_wb, right_look_at_upper_pt1, right_look_at_upper_pt2, (255, 255, 255), 5)
    cv.rectangle(displayed_wb, right_look_at_lower_pt1, right_look_at_lower_pt2, (255, 255, 255), 5)


    return displayed_wb, left_angle, right_angle, detected



cap = video_capture = cv.VideoCapture('project_video.mp4')

left_pre_angle = 0
right_pre_angle = 0
true_pre_angle = 0

while (cap.isOpened()):
    success, frame = cap.read()
    displayed_frame = frame.copy()
    """rbg2gray and transform the view to bird eye view"""

    # transform the view to bird eye view
    top_left = [int(distance_center - distance_width/2),
                int(height - car_hood - detection_distance)]
    top_right = [int(distance_center + distance_width/2),
                 int(height - car_hood - detection_distance)]
    bottom_left = [int(center - car_width/2), int(height-car_hood)]
    bottom_right = [int(center + car_width/2), int(height-car_hood)]

    cv.circle(displayed_frame, top_left, 5, (0, 0, 255), -1)
    cv.circle(displayed_frame, top_right, 5, (0, 0, 255), -1)
    cv.circle(displayed_frame, bottom_left, 10, (0, 0, 255), -1)
    cv.circle(displayed_frame, bottom_right, 10, (0, 0, 255), -1)
    

    pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv.warpPerspective(frame, matrix, (width, height))
    frame_gray = cv.cvtColor(transformed_frame, cv.COLOR_RGB2GRAY)


    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    frame_sharpen = cv.filter2D(frame_gray, -1, kernel_sharpen)
    # frame_sharpen = cv.filter2D(frame_sharpen, -1, kernel_sharpen)

    alpha = 3  # Contrast control (1.0-3.0)
    beta = -200  # Brightness control (0-100)
    adjusted = cv.convertScaleAbs(frame_sharpen, alpha=alpha, beta=beta)

    lanes, left_line_angle, right_line_angle, detected = detect_lines(adjusted)  # detected: 0:None , 1:left, 2:right, 3,both

    """smoothing"""
    left_line_angle_delta = abs(left_line_angle-left_pre_angle)
    right_line_angle_delta = abs(right_line_angle-right_pre_angle)



    left_line_angle = ((left_line_angle*1 + left_pre_angle*2)/3)
    right_line_angle = ((right_line_angle*1 + right_pre_angle*2)/3)


    """
    line_focus: 0:None , 1:left, 2:right, 3,both
    """
    limit = 3
    left_certitude = limit * (limit-left_line_angle_delta)/9
    right_certitude = limit * (limit-right_line_angle_delta)/9
    
    if left_certitude < 0: left_certitude = 0
    if right_certitude < 0: right_certitude = 0


    true_angle = ((left_line_angle*left_certitude) + (right_line_angle*right_certitude))/2


    # if left_line_angle_delta > limit and right_line_angle_delta < limit:
    #     line_focus = 2
    #     true_angle = [right_line_angle]
    # elif left_line_angle_delta < limit and right_line_angle_delta > limit:
    #     line_focus = 1
    #     true_angle = [left_line_angle]
    # elif left_line_angle_delta > limit and right_line_angle_delta > limit:
    #     line_focus = 0
    #     true_angle = [true_pre_angle]
    # else:
    #     line_focus = 3
    #     true_angle = [left_line_angle, right_line_angle]

    



    left_arrow_start = (left_detect_lane_pos, height)
    left_arrow_end = (left_detect_lane_pos + int(left_detect_lane_pos * math.sin(left_line_angle/180*math.pi)) , int(height - lane_detect_height*left_certitude))

    right_arrow_start = (right_detect_lane_pos, height)
    right_arrow_end = (right_detect_lane_pos + int(right_detect_lane_pos * math.sin(right_line_angle/180*math.pi)) ,int(height - lane_detect_height*right_certitude))

    # true_angle = np.mean(true_angle)
    true_angle = ((true_angle*1 + true_pre_angle*5)/6)

    true_arrow_start = (int(width/2), height)
    true_arrow_end = (int(width/2) + int(width/2 * math.sin(true_angle/180*math.pi)), height - lane_detect_height)

    print(
        f"lanes angle: {left_line_angle}° -- {right_line_angle}°    -----   {left_line_angle_delta} ({left_certitude})   {right_line_angle_delta} ({right_certitude})")

    cv.line(adjusted, left_arrow_start, left_arrow_end, (0,0,0), 5)
    cv.line(adjusted, right_arrow_start, right_arrow_end, (0, 0, 0), 5)
    cv.line(adjusted, true_arrow_start, true_arrow_end, (0, 0, 0), 7)

    
    cv.line(displayed_frame, true_arrow_start, true_arrow_end, (0, 0, 255), 7)

    cv.imshow('displayed_frame', displayed_frame)
    cv.imshow('WB', lanes)
    cv.imshow('adjusted', adjusted)
    if cv.waitKey(1) == 27:
        break

    left_pre_angle = left_line_angle
    right_pre_angle = right_line_angle
    true_pre_angle = true_angle
    cv.waitKey(10)
