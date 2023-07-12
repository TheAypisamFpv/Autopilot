import cv2 as cv
import numpy as np


height = 720
width = 1280

center = int(width/2)
car_hood = 150
car_width = 900
distance_center = int(width/2)
detection_distance = 120
distance_width = 130




def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


cap = video_capture = cv.VideoCapture('Test drive\\31 03 2023\\4.MP4')

while (cap.isOpened()):
    success, frame = cap.read()
    if success:
        displayed_frame = frame.copy()
        """rbg2gray and transform the view to bird eye view"""


        # transform the view to bird eye view
        top_left = [int(distance_center - distance_width/2), int(height - car_hood - detection_distance)]
        top_right = [int(distance_center + distance_width/2), int(height - car_hood - detection_distance)]
        bottom_left = [int(center - car_width/2), int(height-car_hood)]
        bottom_right = [int(center + car_width/2), int(height-car_hood)]

        cv.circle(displayed_frame, top_left, 5, (0, 0, 255), -1)
        cv.circle(displayed_frame, top_right, 5, (0, 0, 255), -1)
        cv.circle(displayed_frame, bottom_left, 10, (0, 0, 255), -1)
        cv.circle(displayed_frame, bottom_right, 10, (0, 0, 255), -1)
        cv.imshow('displayed_frame', displayed_frame)


        frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # frame_blur = cv.GaussianBlur(frame_gray, (5, 5), 0)

        pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

        matrix = cv.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv.warpPerspective(frame_gray, matrix, (width, height))

        kernel_sharpen = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
        frame_sharpen = cv.filter2D(transformed_frame, -1, kernel_sharpen)
        # frame_sharpen = cv.filter2D(frame_sharpen, -1, kernel_sharpen)

        alpha = 2  # Contrast control (1.0-3.0)
        beta = -200  # Brightness control (0-100)
        adjusted = cv.convertScaleAbs(frame_sharpen, alpha=alpha, beta=beta)

        # kernel_edge = np.array([[0, -1, 0],
        #                         [-1, 5, -1],
        #                         [0, -1, 0]])
        # edge_detection = cv.filter2D(adjusted, -1, kernel_edge)

        frame_canny = cv.Canny(adjusted, 100, 250)
        lines = cv.HoughLinesP(frame_canny, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=30)
        frame_lines = display_lines(frame_canny, lines)
        combo_image = cv.addWeighted(adjusted, 0.8, frame_lines, 1, 1)

        cv.imshow('adjusted', adjusted)
        cv.imshow('frame_canny', frame_canny)
        cv.imshow('combo_image', frame_lines)
        if cv.waitKey(1) == 27:
            break

        cv.waitKey(10)