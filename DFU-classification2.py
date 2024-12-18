from ultralytics import YOLO

# Initialize a YOLO model for classification
model = YOLO('yolov8s-cls.pt')  # Use pre-trained YOLOv8 classification model

# Train the model
model.train(
    data='/home/rohanb/Documents/mca/ML Mini Project/provided_datasets/DFUC2021_classification',  # Path to your dataset
    epochs=50,
    imgsz=224,
    batch=32
)
