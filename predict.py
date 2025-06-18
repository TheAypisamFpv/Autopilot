import os
from ultralytics import YOLO
import cv2

# Set the environment variable to increase read attempts
os.putenv("OPENCV_FFMPEG_READ_ATTEMPTS", "32768")

model = YOLO("runs\\train3\\weights\\best.pt")

path = r"D:\VS_Python_Project\Autopilot\Autopilot\Test drive\2025.06.18\Full.mp4"
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    
    # if i % 2 != 0:
    #     continue

    if not ret:
        print("Error: Could not read frame or end of video.")
        break

    # Perform prediction on the current frame
    results = model.predict(source=frame)

    # --- Custom drawing logic to control colors ---
    annotated_frame = frame.copy()
    
    # Define colors for each class in BGR format as requested:
    # LeadingCar: cyan
    # Obstacles: gray
    # PredictedPath: blue
    # The order comes from dataset_custom.yaml: ["LeadingCar", "Obstacles", "PredictedPath"]
    colors = [
        (255, 255, 0),   # cyan for LeadingCar
        (0, 0, 255), # red for Obstacles
        (255, 0, 0)      # blue for PredictedPath
    ]
    
    class_names = model.names

    for result in results:
        # Draw masks first
        if result.masks is not None:
            overlay = annotated_frame.copy()
            alpha = 0.4  # Transparency factor

            for i, seg in enumerate(result.masks.xy):
                if i < len(result.boxes):
                    cls_index = int(result.boxes[i].cls[0])
                    if cls_index < len(colors):
                        color = colors[cls_index]
                        
                        pts = seg.astype(int).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color)
            
            annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

        # Then draw bounding boxes and labels on top
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] >= 0.6:  # Filter by confidence, same as original plot
                cls_index = int(box.cls[0])
                
                if cls_index < len(colors):
                    color = colors[cls_index]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw the bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1) # line_width=1
                    
                    # Prepare and draw the label with a background
                    label = f"{class_names[cls_index]} {box.conf[0]:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Prediction", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()