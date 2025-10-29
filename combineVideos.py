import os
import cv2
import json
import subprocess
from model.progressBar import getProgressBar

if __name__ == '__main__':
    INPUTVIDEOS = [
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GOPR5986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP015986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP025986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP035986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP045986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP055986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP065986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP075986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP085986.MP4",
        # r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP095986.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GOPR5987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP015987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP025987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP035987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP045987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP055987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP065987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP075987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP085987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP095987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP105987.MP4",
        r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GP115987.MP4",
    ]
    OUTPUTVIDEOPATH = r"C:\Users\Aypisam\Documents\VS_Python_Project\Autopilot\test_drive\2025.08.19\GOPR5987_combined.MP4"
    
    
    
    outputJsonPath = os.path.splitext(OUTPUTVIDEOPATH)[0] + '.json'
    print()
    if os.path.exists(OUTPUTVIDEOPATH):
        response = input(f"Output video '{OUTPUTVIDEOPATH}' already exists. Overwrite? (Y/n): ").strip().lower()
        if response not in ('y', 'yes', ''):
            print("Operation cancelled by user.")
            exit(0)
    
    if os.path.exists(OUTPUTVIDEOPATH):
        os.remove(OUTPUTVIDEOPATH)
    if os.path.exists(outputJsonPath):
        os.remove(outputJsonPath)

    # Combine JSON data
    combinedData = []
    print("Combining JSON data...")
    for i, videoPath in enumerate(INPUTVIDEOS):
        jsonPath = videoPath.replace('.MP4', '.json')
        if not os.path.exists(jsonPath):
            print(f"\nWarning: JSON file not found: {jsonPath}\n")
            continue
        
        with open(jsonPath, 'r') as file:
            data = json.load(file)
        
        combinedData.extend(data)
        progress = (i + 1) / len(INPUTVIDEOS)
        print(f"Combining JSON: {getProgressBar(progress)}", end='\r')
    
    print(f"\nTotal combined points: {len(combinedData)}")
    
    with open(outputJsonPath, 'w') as file:
        json.dump(combinedData, file, indent=4)

    # NEW: Use FFmpeg for robust video concatenation
    print("Concatenating videos with FFmpeg...")
    
    # Create a list.txt file with paths to videos
    with open('filelist.txt', 'w') as f:
        for videoPath in INPUTVIDEOS:
            f.write(f"file '{videoPath}'\n")
    
    # Build and execute the FFmpeg command
    try:
        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'filelist.txt',
            '-c', 'copy',
            OUTPUTVIDEOPATH
        ]
        
        # Run the command and capture output
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("\nFFmpeg output:")
        print(process.stdout)
        print(process.stderr)
        
    except subprocess.CalledProcessError as e:
        print("\nError during FFmpeg execution:")
        print(e.stderr)
    finally:
        # Clean up the temporary file
        if os.path.exists('filelist.txt'):
            os.remove('filelist.txt')
    
    print(f"Combined video saved to: '{OUTPUTVIDEOPATH}'")
    print(f"Combined JSON saved to: '{outputJsonPath}'")
