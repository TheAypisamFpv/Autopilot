import cv2 as cv
import numpy as np
import math
import time

""" Blue Green Red """
AVAILABLE_COLOR = (255, 255, 255), (232, 39, 33)
UNAVAILABLE_COLOR = (100, 100, 100), (0, 0, 0)
ALERTE_DEPARTURE_COLOR = (39, 33, 232), (0, 0, 0)

autopilot_available = False
autopilot_available_color = UNAVAILABLE_COLOR

height = 1080
width = 1920


lane_center = int(width/2) + 15  # + offset
car_hood = 180
lane_width = width
detection_distance = 250
distance_center = lane_center + 5   # + offset
distance_width = 190

upper_left = [int(distance_center - distance_width/2),
              int(height - car_hood - detection_distance)]
upper_right = [int(distance_center + distance_width/2),
               int(height - car_hood - detection_distance)]

lower_left = [int(lane_center - lane_width/2), int(height-car_hood)]
lower_right = [int(lane_center + lane_width/2), int(height-car_hood)]

pts1 = np.float32([upper_left, lower_left, upper_right, lower_right])
pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

M = cv.getPerspectiveTransform(pts1, pts2)


def nothing(x):
    pass


default_lower_hue = 163
default_lower_saturation = 170
default_lower_value = 177
default_upper_hue = 255
default_upper_saturation = 255
default_upper_value = 255

cv.namedWindow('Trackbars')

cv.createTrackbar(f"lower_hue", "Trackbars", default_lower_hue, 255, nothing)
cv.createTrackbar(f"lower_sat", "Trackbars", default_lower_saturation, 255, nothing)
cv.createTrackbar(f"lower_val", "Trackbars", default_lower_value, 255, nothing)

cv.createTrackbar(f"upper_hue", "Trackbars", default_upper_hue, 255, nothing)
cv.createTrackbar(f"upper_sat", "Trackbars", default_upper_saturation, 255, nothing)
cv.createTrackbar(f"upper_val", "Trackbars", default_upper_value, 255, nothing)


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
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)), (np.abs(
        np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def detect_lines(frame, prev_left_line_x_pos, prev_right_line_x_pos):

    prev_left_line_x_pos, prev_right_line_x_pos = int(
        prev_left_line_x_pos), int(prev_right_line_x_pos)

    """
    masking frame to detect lines
    """

    lower_hue = default_lower_hue
    lower_saturation = default_lower_saturation
    lower_value = default_lower_value

    upper_hue = default_upper_hue
    upper_saturation = default_upper_saturation
    upper_value = default_upper_value

    """"""
    lower_hue = cv.getTrackbarPos("lower_hue", "Trackbars")
    lower_saturation = cv.getTrackbarPos("lower_sat", "Trackbars")
    lower_value = cv.getTrackbarPos("lower_val", "Trackbars")

    upper_hue = cv.getTrackbarPos("upper_hue", "Trackbars")
    upper_saturation = cv.getTrackbarPos("upper_sat", "Trackbars")
    upper_value = cv.getTrackbarPos("upper_val", "Trackbars")
    """"""

    frame = cv.warpPerspective(frame, M, (width, height))
    """add constrast"""
    frame = cv.addWeighted(frame, 3, frame, 0, -150)

    lower = np.array([lower_hue, lower_saturation, lower_value])
    upper = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv.inRange(frame, lower, upper)

    mask = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    # mask = cv.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    """
    histogram
    """

    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    # center = lane_center
    exterior_threshold = 200
    interior_threshold = 100
    left_base  = np.argmax(histogram[      exterior_threshold         :lane_center - interior_threshold]) + exterior_threshold
    right_base = np.argmax(histogram[lane_center + interior_threshold :   width    - exterior_threshold]) + lane_center + interior_threshold

    if left_base == 0:
        left_base = prev_left_line_x_pos

    if right_base == lane_center:
        right_base = prev_right_line_x_pos

    """
    sliding boxs
    """
    left_boxs_x = [left_base]
    left_boxs_y = [height]
    right_boxs_x = [right_base]
    right_boxs_y = [height]

    box_height = 25
    box_width = 250

    y = height - box_height

    while y > 0:
        """
        left threshold
        """
        img = mask[y-box_height:y, left_base -
                   int(box_width/2):left_base + int(box_width/2)]
        contours, _ = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            moment = cv.moments(cnt)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                left_boxs_x.append(cx + left_base - int(box_width/2))
                left_boxs_y.append(cy + y)
                left_base = left_base - int(box_width/2) + cx

        """
        right threshold
        """
        img = mask[y-box_height:y, right_base -
                   int(box_width/2):right_base + int(box_width/2)]
        contours, _ = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            moment = cv.moments(cnt)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                right_boxs_x.append(cx + right_base - int(box_width/2))
                right_boxs_y.append(cy + y)
                right_base = right_base - int(box_width/2) + cx

        # cv.rectangle(mask, (left_base - int(box_width/2), y-box_height), (left_base + int(box_width/2), y), (255, 255, 255), 2)
        # cv.rectangle(mask, (right_base - int(box_width/2), y-box_height), (right_base + int(box_width/2), y), (255, 255, 255), 2)

        y -= box_height


    lines_x = [left_boxs_x, right_boxs_x]
    lines_y = [left_boxs_y, right_boxs_y]

    for boxs in range(len(lines_x)):
        boxs_x = lines_x[boxs]
        for index in range(1, len(boxs_x)-1):
            x_1 = boxs_x[index]
            y_1 = lines_y[boxs][index]

            x_2 = boxs_x[index + 1]
            y_2 = lines_y[boxs][index + 1]

            cv.line(mask, (x_1, y_1), (x_2, y_2), (0, 0, 0), 12)
            cv.line(mask, (x_1, y_1), (x_2, y_2), (255, 255, 255), 8)


    left_line_x_pos = left_boxs_x[0]
    left_found = True if len(left_boxs_x) > 0 else False

    right_line_x_pos = right_boxs_x[0]
    right_found = True if len(right_boxs_x) > 0 else False


    return mask, left_line_x_pos, left_found, right_line_x_pos, right_found


def lane_departure(left_line_x_pos, left_found, right_line_x_pos, right_found):
    """
    detect lane departure
    """
    left_line = [int(lane_center - left_line_x_pos), left_found]
    right_line = [int(right_line_x_pos - lane_center), right_found]

    left_line_departure = 0
    right_line_departure = 0

    left_line_departure_threshold = 230
    right_line_departure_threshold = 230

    if left_line[0] < left_line_departure_threshold and left_line[1]:
        left_line_departure = 1

    if right_line[0] < right_line_departure_threshold and right_line[1]:
        right_line_departure = 1

    return left_line_departure, right_line_departure


def main():
    """
    lane positions
    """
    left_line_x_pos = 0
    left_line_x_pos_prev = left_line_x_pos
    left_found = False

    right_line_x_pos = 0
    right_line_x_pos_prev = right_line_x_pos
    right_found = False

    setting = ''

    """
    Video here
    """
    cap = cv.VideoCapture('Test drive\\31 03 2023\\3.MP4')
    video_fps = cap.get(cv.CAP_PROP_FPS)

    fps = float(f"{float(video_fps):.2f}")

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_id = cap.get(1)

        if frame_id % 2 or True:
            if ret:
                start_time = time.time()
                displayed_frame = frame.copy()
                lane_frame, true_left_line_x_pos, left_found, true_right_line_x_pos, right_found = detect_lines(
                    frame, left_line_x_pos_prev, right_line_x_pos_prev)

                left_line_x = (true_left_line_x_pos + left_line_x_pos_prev) / 2
                right_line_x = (true_right_line_x_pos +
                                right_line_x_pos_prev) / 2

                left_line_x_pos_prev = left_line_x
                right_line_x_pos_prev = right_line_x

                steering_angle = 0

                steering_color = UNAVAILABLE_COLOR[1]
                steering_wheel_color = AVAILABLE_COLOR[0]

                """
                steering wheel (rotate with the steering angle) (made with 1 circle and 2 lines)
                """
                steering_wheel_size = 60
                steering_wheel_center = (int(width/2.5), int(90))
                cv.circle(displayed_frame, steering_wheel_center,
                          steering_wheel_size + 25, steering_color, -1)
                cv.circle(displayed_frame, steering_wheel_center, steering_wheel_size+int(
                    steering_wheel_size/10), steering_wheel_color, int(steering_wheel_size/7))
                cv.circle(displayed_frame, (int(steering_wheel_center[0] - (steering_wheel_size/6.5)*math.sin(math.radians(steering_angle))), int(
                    steering_wheel_center[1] + (steering_wheel_size/6.5)*math.cos(math.radians(steering_angle)))), int(steering_wheel_size/2.2), steering_wheel_color, -1)

                cv.line(displayed_frame, steering_wheel_center, (int(steering_wheel_center[0]-steering_wheel_size*math.sin(math.radians(steering_angle))), int(
                    steering_wheel_center[1]+steering_wheel_size*math.cos(math.radians(steering_angle)))), steering_wheel_color, int(steering_wheel_size/5))
                steering_wheel_x_lenght = int(
                    steering_wheel_size*math.cos(math.radians(steering_angle)))
                steering_wheel_y_lenght = int(
                    steering_wheel_size*math.sin(math.radians(steering_angle)))
                cv.line(displayed_frame, (steering_wheel_center[0]-steering_wheel_x_lenght, steering_wheel_center[1]-steering_wheel_y_lenght), (
                    steering_wheel_center[0]+steering_wheel_x_lenght, steering_wheel_center[1]+steering_wheel_y_lenght), steering_wheel_color, int(steering_wheel_size/5))

                """
                show FPS
                """
                cv.putText(displayed_frame, f"FPS: {fps:.2f} {setting}", (
                    5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
                cv.putText(displayed_frame, f"FPS: {fps:.2f} {setting}", (
                    5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                """
                scale down to 720p
                """
                displayed_frame = cv.resize(displayed_frame, (1280, 720))
                lane_frame = cv.resize(lane_frame, (1280, 720))

                cv.imshow('displayed_frame', displayed_frame)
                cv.imshow('masked_frame', lane_frame)

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

