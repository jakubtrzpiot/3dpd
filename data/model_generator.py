from ultralytics import YOLO
import shutil
import torch
import os

def main():
    # Check for CUDA availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load YOLO model
    model = YOLO('yolo11n.pt')

    # Train model
    results = model.train(
        data='data.yaml',
        epochs=300,
        imgsz=640,
        batch=64,
        project='runs/train',
        name='spaghetti_detector',
        device=device,
        workers=8,
        half=True if device == 0 else False,  # Use half precision if GPU is available
        amp=True,  # Automatic Mixed Precision for better speed/memory efficiency
        pretrained=True
    )

    # Ensure output path exists
    dest_path = "../model/3dpd.pt"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Copy best model to target directory
    shutil.copy2("runs/train/spaghetti_detector/weights/best.pt", dest_path)
    print("✅ Model saved to:", dest_path)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
