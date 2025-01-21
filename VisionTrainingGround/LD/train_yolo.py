from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
   data='/PATH TO DATA/VisionTrainingGround/LD/datasets/#MGRS_REGION#_dataset/dataset.yaml', # TODO: SET PATH TO DATASET
   name='yolov8n_53L_n100',
   degrees=180,
   scale=0.3,
   fliplr=0.0,
   imgsz=576,
   mosaic = 0,
   perspective = 0.0001,
   plots=True,
   save=True,
   resume=True,
   epochs=300,
   device = 'cpu'
)
