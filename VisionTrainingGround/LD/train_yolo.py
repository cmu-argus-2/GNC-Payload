"""
This script trains a YOLO model on a custom dataset. It accepts command-line arguments to specify
the region code, dataset path, model save directory, YOLO version, and the number of epochs for training.
The model is trained using the provided dataset, and the results are saved in the specified directory.

Required arguments:
- --region: The region code for naming and saving model results.
- --data: Path to the dataset YAML file.
- --save_dir: Directory to save the trained model file.
Optional arguments:
- --version: YOLO model version (default is "yolov8n").
- --epochs: Number of training epochs (default is 300).
"""

import argparse

import torch
from ultralytics import YOLO


def parse_args():
    """
    Parses command-line arguments to specify configuration for YOLO training.

    Arguments:
    - None

    Returns:
    - args: Namespace object containing parsed arguments with the following attributes:
       - region (str): The region code (required).
       - data (str): Path to the dataset YAML file (required).
       - save_dir (str): Directory to save the model file (required).
       - version (str): YOLO model version (default is "yolov8n").
       - epochs (int): Number of training epochs (default is 300).
    """
    parser = argparse.ArgumentParser(description="Train YOLO model with custom name and data path.")
    parser.add_argument("--region", type=str, required=True, help="Region Code")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help='Path to save the model.pt file. The file is saved in save_dir/"<version>_<region>_n<epochs"',
    )
    parser.add_argument(
        "--version", type=str, required=False, default="yolov8n", help="YOLO version"
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default=300, help="Number of training epochs"
    )
    return parser.parse_args()


def train_yolo():
    """
    Main function to initialize and train a YOLO model using specified command-line arguments.

    This function:
    - Parses the command-line arguments to get training parameters.
    - Determines the computing device (CPU or GPU).
    - Loads the YOLO model based on the version specified.
    - Sets up the training configuration and runs the training process.
    - Saves the trained model to the specified directory.

    Arguments:
    - None

    Returns:
    - None
    """
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(f"{args.version}.pt")
    name = args.version + "_" + args.region + "_" + "n" + str(args.epochs)
    print(args.save_dir)

    # pylint: disable=unused-variable
    results = model.train(
        data=args.data,  # Dataset path from argument
        project=args.save_dir,
        name=name,  # The result files are saved in project/name
        degrees=180,  # Image augmentation parameters
        scale=0.3,
        fliplr=0.0,
        imgsz=576,
        mosaic=0,
        perspective=0.0001,
        plots=True,  # Plot the results
        save=True,  # Save the trained model
        resume=False,  # Do not resume training
        epochs=args.epochs,  # Number of epochs for training
        device=device,  # Set device to cuda or cpu
    )


if __name__ == "__main__":
    train_yolo()
