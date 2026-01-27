import cv2
import os
import random
import numpy as np

def parseVectorLine(line):
    """Parses the vector line and returns a list of dictionaries."""
    vectors = []
    try:
        # Extract the part after 'vectors : '
        vector_data = line.split(' : ')[1].strip()
        # Split into individual vector strings
        vector_strings = vector_data.split(' ')
        time_increment = 3.0 / len(vector_strings) if vector_strings else 0

        for i, vec_str in enumerate(vector_strings):
            x_str, y_str = vec_str.split(',')
            vectors.append({
                'x': float(x_str),
                'y': float(y_str),
                'time': (i + 1) * time_increment
            })
    except (IndexError, ValueError) as e:
        print(f"Error parsing vector line: {line.strip()} - {e}")
    return vectors

def parseSpeed(line):
    """Parses the speed line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing speed line: {line.strip()} - {e}")
        return 0.0

def parseAcceleration(line):
    """Parses the acceleration line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing acceleration line: {line.strip()} - {e}")
        return 0.0

def parseTurnRate(line):
    """Parses the turn rate line and returns a float."""
    try:
        return float(line.split(' : ')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error parsing turn rate line: {line.strip()} - {e}")
        return 0.0

def putTextWithOutline(frame, text, org, fontFace, fontScale, color, thickness=1):
    """Draws text with a black outline."""
    # Draw the outline in black
    cv2.putText(frame, text, org, fontFace, fontScale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Draw the main text in the specified color
    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

def viewRandomItem(datasetDir):
    """
    Loads a random item from the dataset and visualizes the trajectory.

    Args:
        datasetDir: The root directory of the dataset.
    """
    imagesDir = os.path.join(datasetDir, "images")
    labelsDir = os.path.join(datasetDir, "labels")

    if not os.path.exists(labelsDir):
        print(f"Labels directory not found at: {labelsDir}")
        return

    labelFiles = [f for f in os.listdir(labelsDir) if f.endswith(".txt")]
    if not labelFiles:
        print("No label files found in the directory.")
        return
    
    print(f"Total datapoints: {len(labelFiles)}")
    
    labelFiles.sort()
    currentIndex = random.randint(0, len(labelFiles) - 1)

    while True:
        labelFile = labelFiles[currentIndex]
        baseName = os.path.splitext(labelFile)[0]
        metadataPath = os.path.join(labelsDir, labelFile)

        prevImagePath = os.path.join(imagesDir, f"{baseName}_prev.png")
        currentImagePath = os.path.join(imagesDir, f"{baseName}_current.png")

        # Read metadata
        vectors = []
        speed, acceleration, turnRate = 0, 0, 0
        with open(metadataPath, 'r') as f:
            for line in f:
                if line.startswith('vectors'):
                    vectors = parseVectorLine(line)
                elif line.startswith('speed'):
                    speed = parseSpeed(line)
                elif line.startswith('acceleration'):
                    acceleration = parseAcceleration(line)
                elif line.startswith('turnRate'):
                    turnRate = parseTurnRate(line)

        # Load images
        prevFrame = cv2.imread(prevImagePath)
        currentFrame = cv2.imread(currentImagePath)

        if prevFrame is None or currentFrame is None:
            print(f"Could not load images for {baseName}")
            currentIndex = (currentIndex + 1) % len(labelFiles)
            continue

        putTextWithOutline(prevFrame, "Previous Frame", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- Visualization ---
        visFrame = currentFrame.copy()
        putTextWithOutline(visFrame, "Current Frame", (225, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        originPoint = (visFrame.shape[1] // 2, visFrame.shape[0] - 5)
        vecToPixel = 3 # Increased for better visibility
        vectorThickness = 2

        # Draw trajectory vectors
        if vectors:
            colorBlend = 255 / len(vectors) if len(vectors) > 0 else 255
            for i, vector in enumerate(vectors):
                x = int(originPoint[0] + vector['x'] * vecToPixel)
                y = int(originPoint[1] - vector['y'] * vecToPixel)
                
                color = (0, 255 - i * colorBlend, i * colorBlend)
                cv2.line(visFrame, originPoint, (x, y), color, vectorThickness)
                # cv2.circle(visFrame, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                putTextWithOutline(visFrame, f"{vector['time']:.2f}s", (x + 10, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                originPoint = (x, y)  # Update origin for next vector

        # Display info
        speedText = f"Speed: {speed:.2f} m/s"
        accelerationText = f"Acceleration: {acceleration:.2f} m.s-2"
        turnRateText = f"Turn Rate: {turnRate:.2f} deg/s"
        infoTexts = [speedText, accelerationText, turnRateText]
        for i, text in enumerate(infoTexts):
            putTextWithOutline(visFrame, text, (10, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Concatenate previous and current frames for display
        combinedDisplay = np.hstack([prevFrame, visFrame])

        cv2.imshow("Dataset Viewer - Press 'q' to quit, any other key for next", combinedDisplay)
        
        key = cv2.waitKeyEx(0)
        if key == ord('q'):
            break
        elif key == 2424832:  # Left arrow
            currentIndex = (currentIndex - 1 + len(labelFiles)) % len(labelFiles)
        elif key == 2555904:  # Right arrow
            currentIndex = (currentIndex + 1) % len(labelFiles)
        else: # Any other key for random
            currentIndex = random.randint(0, len(labelFiles) - 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    datasetOutputDir = r"D:\VS_Python_Project\Autopilot\Autopilot\dataset\NVIDIA_output_12_3.0_0.1_framesize640x360"
    viewRandomItem(datasetOutputDir)
