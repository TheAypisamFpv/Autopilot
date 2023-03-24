import cv2 as cv
import numpy as np
import math


AVAILABLE_COLOR = (255, 255, 255), (255, 0, 72)
UNAVAILABLE_COLOR = (100, 100, 100), (0, 0, 0)
autopilot_available = False
autopilot_available_color = UNAVAILABLE_COLOR

height = 1080
width = 1920


lane_center = int(width/2) + 15  # + offset
car_hood = 200
lane_width = 1100
detection_distance = 250
distance_center = lane_center +5   # + offset
distance_width = 150

upper_left = [int(distance_center - distance_width/2), int(height - car_hood - detection_distance)]
upper_right = [int(distance_center + distance_width/2),int(height - car_hood - detection_distance)]

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


def detect_lane_width(frame, displayed_frame):


    """bird's eye view of display_frame"""
    M = cv.getPerspectiveTransform(pts1, pts2)
    displayed_frame = cv.warpPerspective(displayed_frame, M, (width, height))

    """modifiy contrast and brightness"""
    alpha = 1.5
    beta = 0
    corected_frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    

    """blur the frame"""
    blur_frame = cv.GaussianBlur(corected_frame, (9, 9), 0)

    """detecet edges"""
    canny_frame = cv.Canny(blur_frame, 75, 100)

    """Black or white frame"""
    bw_frame = cv.threshold(canny_frame, 127, 255, cv.THRESH_BINARY)[1]

    """mask the area that we are interested in"""
    mask = np.zeros_like(bw_frame)
    points = np.array([upper_right, upper_left, lower_left, lower_right], np.int32)
    cv.fillPoly(mask, pts=[points], color=(255, 255, 255))
    masked_frame = cv.bitwise_and(bw_frame, mask)

    blur_masked_frame = cv.GaussianBlur(masked_frame, (21, 21), 0)


    """transform the frame to bird's eye view"""
    M = cv.getPerspectiveTransform(pts1, pts2)
    masked_frame = cv.warpPerspective(masked_frame, M, (width, height))
    blur_masked_frame = cv.warpPerspective(blur_masked_frame, M, (width, height))
    bw_blur_masked_frame = cv.threshold(blur_masked_frame, 30, 255, cv.THRESH_BINARY)[1]

    masked_frame = bw_blur_masked_frame

    """detect start of the lines"""
    left_line_x_pos = (width/2 - lane_width/2)
    left_found = False
    right_line_x_pos = (width/2 + lane_width/2)
    right_found = False

    detecting_height = 75

    """project a line from the center of the car to the left and right"""
    for pixel in range(100,int(width/2)):
        if masked_frame[height-detecting_height:height, int(width/2 - pixel)].any() > 0 and not left_found:
            left_line_x_pos = width/2 - pixel
            left_found = True
        if masked_frame[height-detecting_height:height, int(width/2 + pixel)].any() > 0 and not right_found:
            right_line_x_pos = width/2 + pixel
            right_found = True

        if left_found and right_found:
            break

    

    """calculate the true lane width"""
    true_lane_width = int(right_line_x_pos - left_line_x_pos)

    return displayed_frame, masked_frame, left_line_x_pos, left_found, right_line_x_pos, right_found, true_lane_width, car_hood

def main():
    """
    lane positions
    """
    left_line_x_pos = (width/2 - lane_width/2)
    left_line_x_pos_prev = left_line_x_pos
    predicted_left_line_x_pos = left_line_x_pos

    right_line_x_pos = (width/2 + lane_width/2)
    right_line_x_pos_prev = right_line_x_pos
    predicted_right_line_x_pos = right_line_x_pos

    pre_line_width = lane_width

    left_line_incertitude = 0
    left_line_incertitude_prev = 0
    right_line_incertitude = 0
    right_line_incertitude_prev = 0

    lane_keeping_angle_prev = 0
    last_lane_keeping_angle = 0

    
    """
    Video here
    """
    cap = cv.VideoCapture('Test drive\\24_03_2023_02.MP4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_id = cap.get(1)

        if frame_id % 2 or True:
            if ret:
                displayed_frame = frame.copy()
                corected_frame, masked_frame, true_left_line_x_pos, left_found, true_right_line_x_pos, right_found, true_lane_width, car_hood = detect_lane_width(frame, displayed_frame)


                """
                smooth the lines and calculate the incertitude
                """
                incertitude_threshold = 8

                left_line_x_pos = (true_left_line_x_pos + left_line_x_pos_prev*15) / 16
                left_line_incertitude = np.abs(left_line_x_pos - left_line_x_pos_prev)
                left_line_incertitude = (left_line_incertitude + left_line_incertitude_prev*5) / 6
                left_found = False if left_line_incertitude > incertitude_threshold else left_found

                right_line_x_pos = (true_right_line_x_pos + right_line_x_pos_prev*15) / 16
                right_line_incertitude = np.abs(right_line_x_pos - right_line_x_pos_prev)
                right_line_incertitude = (right_line_incertitude + right_line_incertitude_prev*5) / 6
                right_found = False if right_line_incertitude > incertitude_threshold else right_found

                left_line_x_pos_prev = left_line_x_pos
                right_line_x_pos_prev = right_line_x_pos
                left_line_incertitude_prev = left_line_incertitude
                right_line_incertitude_prev = right_line_incertitude


                """
                calculate the average lane width
                """
                true_lane_width = int(true_lane_width) if (left_found and right_found) else int(pre_line_width)

                average_lane_width = (true_lane_width + pre_line_width*100) / 101
                pre_line_width = average_lane_width

                


                


                """
                predict the lines using the average lane width
                """
                predicted_left_line_x_pos = right_line_x_pos - average_lane_width
                predicted_right_line_x_pos = left_line_x_pos + average_lane_width


                """
                if the lines are not found, use the predicted lines
                """
                left_line_x_pos = predicted_left_line_x_pos if not left_found else left_line_x_pos
                right_line_x_pos = predicted_right_line_x_pos if not right_found else right_line_x_pos


                """
                clamp the lines
                """
                center_threshold = 100
                left_line_x_pos = np.clip(left_line_x_pos, 0, width/2 - center_threshold)
                right_line_x_pos = np.clip(right_line_x_pos, width/2 + center_threshold, width)

                predicted_left_line_x_pos = np.clip(predicted_left_line_x_pos, 0, width/2 - center_threshold)
                predicted_right_line_x_pos = np.clip(predicted_right_line_x_pos, width/2 + center_threshold, width)

                """
                determine the lane position
                """
                lane_position = ((left_line_x_pos + right_line_x_pos + width/2) / 3)
                lane_keeping_angle = np.arctan((lane_position - width/2) / height)
                lane_keeping_angle = (lane_keeping_angle + lane_keeping_angle_prev*10) / 11
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

                print(f"left_line_incertitude: {int(left_line_incertitude): =04d}  right_line_incertitude: {int(right_line_incertitude): =04d}  lane width: {true_lane_width: =04d}   average lane width: {int(average_lane_width): =04d}  left_found: {int(left_found)}  right_found: {int(right_found)}   frame_id: {frame_id}")

                """
                draw the predicted lines
                """
                detecting_height = 25
                """their foreground"""
                cv.line(corected_frame, (int(predicted_left_line_x_pos), height-detecting_height), (int(predicted_left_line_x_pos), height), (255, 0, 0), 6)
                cv.line(corected_frame, (int(width/2), height), (int(predicted_left_line_x_pos), height-detecting_height), (255, 0, 0), 6)
                cv.line(corected_frame, (int(predicted_right_line_x_pos), height-detecting_height), (int(predicted_right_line_x_pos), height), (255, 0, 0), 6)
                cv.line(corected_frame, (int(width/2), height), (int(predicted_right_line_x_pos), height-detecting_height), (255, 0, 0), 6)


                """
                draw the lines
                """
                """their background"""
                detecting_height = 75
                """
                lane detection
                """
                cv.line(corected_frame, (int(left_line_x_pos), height-detecting_height), (int(left_line_x_pos), height), (0, 0, 0), 12)
                cv.line(corected_frame, (int(width/2), height), (int(left_line_x_pos), height-detecting_height), (0, 0, 0), 12)
                cv.line(corected_frame, (int(right_line_x_pos), height-detecting_height), (int(right_line_x_pos), height), (0, 0, 0), 12)
                cv.line(corected_frame, (int(width/2), height), (int(right_line_x_pos), height-detecting_height), (0, 0, 0), 12)


                """
                lane keeping (using lane position and the angle from the bottom center)
                and display a line on the screen
                """
                cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(distance_center + np.tan(lane_keeping_angle)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance/2)), (0,0,0), 14)


                """their foreground"""
                cv.line(corected_frame, (int(width/2), height), (int(left_line_x_pos), height-detecting_height), (0, 255, 0), 8)
                cv.line(corected_frame, (int(left_line_x_pos), height-detecting_height), (int(left_line_x_pos), height), (AVAILABLE_COLOR[0] if left_found else AVAILABLE_COLOR[1]), 8)

                cv.line(corected_frame, (int(width/2), height), (int(right_line_x_pos), height-detecting_height), (0, 0, 255), 8)
                cv.line(corected_frame, (int(right_line_x_pos), height-detecting_height), (int(right_line_x_pos), height), (AVAILABLE_COLOR[0] if right_found else AVAILABLE_COLOR[1]), 8)

                cv.line(displayed_frame, (int(lane_center), height-car_hood), (int(distance_center + np.tan(lane_keeping_angle)*(car_hood + detection_distance)/3), int(height-car_hood-detection_distance/2)), AVAILABLE_COLOR[0] if lane_keeping else UNAVAILABLE_COLOR[0], 10)
                
                """
                drawing 2 small vertical lines on the bottom of the screen (grey if the line is not found, white if it is found, blue if it is predicted)
                """
                if left_line_x_pos == predicted_left_line_x_pos:
                    left_line_predicted = True
                else:
                    left_line_predicted = False
                
                if right_line_x_pos == predicted_right_line_x_pos:
                    right_line_predicted = True
                else:
                    right_line_predicted = False

                cv.line(displayed_frame, (lane_center-50, height-50), (lane_center-50, height-150), (0,0,0), 9)
                cv.line(displayed_frame, (lane_center+50, height-50), (lane_center+50, height-150), (0,0,0), 9)

                if left_line_predicted and not right_line_predicted:
                    cv.line(displayed_frame, (lane_center-50, height-50), (lane_center-50, height-150), AVAILABLE_COLOR[1], 6)
                else:
                    cv.line(displayed_frame, (lane_center-50, height-50), (lane_center-50, height-150), AVAILABLE_COLOR[0] if left_found else UNAVAILABLE_COLOR[0], 6)

                if right_line_predicted and not left_line_predicted:
                    cv.line(displayed_frame, (lane_center+50, height-50), (lane_center+50, height-150), AVAILABLE_COLOR[1], 6)
                else:
                    cv.line(displayed_frame, (lane_center+50, height-50), (lane_center+50, height-150), AVAILABLE_COLOR[0] if right_found else UNAVAILABLE_COLOR[0], 6)

                cv.putText(displayed_frame, "Lane Keeping Only", (int(width/2)-75, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
                cv.putText(displayed_frame, "Lane Keeping Only", (int(width/2)-75, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


                """
                add legend for the vertical lines' colors (grey if the line is not found, white if it is found, blue if it is predicted)
                """
                cv.putText(displayed_frame, "color legend:", (10, height-110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
                cv.putText(displayed_frame, "color legend:", (10, height-110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv.putText(displayed_frame, "- when grey: line is not found", (25, height-80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
                cv.putText(displayed_frame, "- when grey: line is not found", (25, height-80), cv.FONT_HERSHEY_SIMPLEX, 0.75, UNAVAILABLE_COLOR[0], 2)

                cv.putText(displayed_frame, "- when white: line is found", (25, height-50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
                cv.putText(displayed_frame, "- when white: line is found", (25, height-50), cv.FONT_HERSHEY_SIMPLEX, 0.75, AVAILABLE_COLOR[0], 2)

                cv.putText(displayed_frame, "- when blue: line is predicted", (25, height-20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
                cv.putText(displayed_frame, "- when blue: line is predicted", (25, height-20), cv.FONT_HERSHEY_SIMPLEX, 0.75, AVAILABLE_COLOR[1], 2)
                    



                # cv.line(corected_frame, (int(width/2), height), (int(width/2), 0), (255, 255, 255), 6)


                steering_angle = np.degrees(lane_keeping_angle)

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


                """
                scale down to 720p
                """
                displayed_frame = cv.resize(displayed_frame, (1280, 720))
                corected_frame = cv.resize(corected_frame, (1280, 720))
                masked_frame = cv.resize(masked_frame, (1280, 720))

                cv.imshow('frame', displayed_frame)
                cv.imshow('corected_frame', corected_frame)
                cv.imshow('masked_frame', masked_frame)

                """
                skip half of the frames
                """

                if cv.waitKey(1) == 27:
                    break


main()
