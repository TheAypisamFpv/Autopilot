from ultralytics import YOLO

def trainModel():
    model = YOLO("autopilot AI\yolo11m-seg.pt")

    model.train(data="autopilot AI\dataset_custom.yaml",
                imgsz=640,
                batch=9,
                epochs=200,
                workers=1,
                device=0)

if __name__ == "__main__":
    trainModel()