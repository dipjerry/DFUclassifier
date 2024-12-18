from ultralytics import YOLO
model = YOLO("/home/rohanb/Documents/mca/ML Mini Project/dfu/runs/classify/train3/weights/best.pt") # choose best.pt from test file accordingly

results = model.val(
    data='/home/rohanb/Documents/mca/ML Mini Project/provided_datasets/DFUC2021_classification',
    imgsz=224,
    save=True
)
print(results)  # Outputs metrics like accuracy and confusion matrix

