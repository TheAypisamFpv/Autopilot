import os
from ultralytics import YOLO
import cv2

# Set the environment variable to increase read attempts
os.putenv("OPENCV_FFMPEG_READ_ATTEMPTS", "32768")

model = YOLO("autopilot AI\BestWeights\Yolo9_custom.pt")

path = r"D:\VS_Python_Project\Autopilot\Autopilot\autopilot non AI\Test drive\2023.07.04\full.mp4"
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    
    if i % 2 != 0:
        continue

    if not ret:
        print("Error: Could not read frame or end of video.")
        break


    # Perform prediction on the current frame
    results = model.predict(source=frame)

    for result in results:
        annotated_frame = result.plot(boxes=False,
                                      line_width=1,
                                      conf=0.6
                                      )  # Get the annotated frame
        
        cv2.imshow("Prediction", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()