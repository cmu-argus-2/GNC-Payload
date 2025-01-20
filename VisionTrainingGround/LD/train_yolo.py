from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8m_R17_w106_n100/weights/last.pt')

# Train
results = model.train(
   data='datasets/17R_dataset_w106_n100/dataset.yaml',
   name='yolov8m_R17_w106_n100',
   degrees=180,
   scale=0.3,
   fliplr=0.0,
   imgsz=576,
   mosaic = 0,
   perspective = 0.0001,
   plots=True,
   save=True,
   resume=True,
   epochs=300
)


