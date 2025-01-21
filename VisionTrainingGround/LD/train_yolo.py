from ultralytics import YOLO

# Load pretrained yolo model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
   data='/home/frederik/cmu/GNC-Payload/VisionTrainingGround/LD/datasets/53L_dataset/dataset.yaml',
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
