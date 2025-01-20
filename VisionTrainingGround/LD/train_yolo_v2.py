from ultralytics import YOLO
#from wandb.integration.ultralytics import add_wandb_callback

# Load pretrained yolo model
model = YOLO('runs/detect/yolov8m_R17_w19_n100/weights/last.pt')
#model.cuda()
#add_wandb_callback(model, enable_model_checkpointing=True)

# Train
results = model.train(
   data='datasets/17R_dataset_w91_n100/dataset.yaml',
   name='yolov8m_R17_w19_n100',
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


