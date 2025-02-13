import argparse
import torch
from ultralytics import YOLO

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model with custom name and data path.")
    parser.add_argument('--region', type=str, required=True, help='Region Code')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the model.pt file. The file is saved in save_dir/"<version>_<region>_n<epochs"')
    parser.add_argument('--version', type=str, required=False, default="yolov8n", help='YOLO version')
    parser.add_argument('--epochs', type=int, required=False, default=300, help='YOLO version')
    return parser.parse_args()

# Main function to train the model
def train_yolo():
   args = parse_args()

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   print(f"Using device: {device}")

   model = YOLO(f'{args.version}.pt')
   name = args.version + "_" + args.region + "_" + "n" + str(args.epochs)
   print(args.save_dir)

   # Train
   results = model.train(
      data=args.data,  # Dataset path from argument
      project = args.save_dir, 
      name=name, # The result files are all saved in project/name
      degrees=180,
      scale=0.3,
      fliplr=0.0,
      imgsz=576,
      mosaic=0,
      perspective=0.0001,
      plots=True,
      save=True,
      resume=False,
      epochs=args.epochs,
      device=device  # Set device to cuda or cpu
   )

if __name__ == "__main__":
    train_yolo()
