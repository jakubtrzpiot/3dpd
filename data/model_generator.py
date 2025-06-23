from ultralytics import YOLO
import shutil

model = YOLO('yolo11n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs/train',
    name='spaghetti_detector'
)

shutil.copy2("runs/train/spaghetti_detector/weights/best.pt", "../model/3dpd.pt")
print("Model saved in /model/3dpd.pt")