from ultralytics import YOLO

saveDir = r'D:\VS_Python_Project\Autopilot\Autopilot\runs'
modelPath = r'D:\VS_Python_Project\Autopilot\Autopilot\models\yolo11m-seg.pt'

def trainModel():
    model = YOLO(modelPath)

    model.train(data="dataset_custom.yaml",
                imgsz=640,
                batch=9,
                epochs=150,
                workers=1,
                device='cuda',
                save_dir=saveDir,
                project=saveDir)

if __name__ == "__main__":
    trainModel()