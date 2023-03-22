import cv2 as cv
import numpy as np


def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


cap = video_capture = cv.VideoCapture('project_video.mp4')

while (cap.isOpened()):
    success, frame = cap.read()
    displayed_frame = frame.copy()
    """rbg2gray and transform the view to bird eye view"""
    height = 720
    width = 1280

    # transform the view to bird eye view
    top_left = [560, 450]
    top_right = [750, 450]
    bottom_left = [250, height-70]
    bottom_right = [width-150, height-70]

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

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    frame_sharpen = cv.filter2D(transformed_frame, -1, kernel)


    alpha = 2  # Contrast control (1.0-3.0)
    beta = -200  # Brightness control (0-100)
    adjusted = cv.convertScaleAbs(frame_sharpen, alpha=alpha, beta=beta)


    frame_canny = cv.Canny(adjusted, 100, 150)
    lines = cv.HoughLinesP(frame_canny, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=15)
    frame_lines = display_lines(frame_canny, lines)
    combo_image = cv.addWeighted(frame_lines, 0.8, frame_lines, 1, 1)

    cv.imshow('adjusted', adjusted)
    cv.imshow('combo_image', combo_image)
    if cv.waitKey(1) == 27:
        break

    cv.waitKey(10)