# DFU3.py

from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # We can use yolov8s.pt or others

# Train the model
model.train(
    data='/home/rohanb/Documents/mca/ML Mini Project/dfu/dataset.yaml',
    epochs=100, # configuration of parameters for training
    imgsz=224,
    batch=16
)
