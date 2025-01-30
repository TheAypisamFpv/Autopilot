from ultralytics import YOLO

def trainModel():
    model = YOLO("Yolo9_custom.pt")

    model.train(data="dataset_custom.yaml",
                imgsz=640,
                batch=9,
                epochs=250,
                workers=1,
                device=0)

if __name__ == "__main__":
    trainModel()