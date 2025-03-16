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
import os

import torch
from ultralytics import YOLO

from utils.config_utils import load_config
from VisionTrainingGround.LD.prepare_yolo_data import LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME


def parse_args():
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train YOLO model with custom name and data path.")
    parser.add_argument(
        "training_dir",
        type=str,
        help="The main training directory.",
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to run saliency analysis for.",
    )
    parser.add_argument(
        "--skip_regions", type=str, nargs="+", default=[], help="MGRS regions to skip."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--version", type=str, required=False, default="yolov8n", help="YOLO version"
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default=300, help="Number of training epochs"
    )
    return parser.parse_args()


def train_yolo(
        training_dir: str,
        region: str,
        overwrite: bool,
        version: str,
        epochs: int,
) -> None:
    """
    Main function to initialize and train a YOLO model using specified command-line arguments.

    This function:
    - Determines the computing device (CPU or GPU).
    - Loads the YOLO model based on the version specified.
    - Sets up the training configuration and runs the training process.
    - Saves the trained model.

    Arguments:
    - training_dir (str): The main training directory.
    - region (str): The MGRS region to train the model for.
    - overwrite (bool): Whether to overwrite the output file if it exists.
    - version (str): The YOLO model version to use.
    - epochs (int): The number of epochs for training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir = os.path.join(training_dir, region, LD_TRAINING_DIR_NAME)
    name = f"{version}_{region}_n{epochs}"
    output_file = os.path.join(save_dir, f"{name}.pt")
    if os.path.exists(output_file):
        if not overwrite:
            raise FileExistsError(f"Output file {output_file} already exists.")

        os.remove(output_file)

    model = YOLO(f"{version}.pt")
    yolo_config_path = os.path.join(training_dir, region, LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME)
    # pylint: disable=unused-variable
    results = model.train(
        data=yolo_config_path,  # Dataset path from argument
        project=save_dir,
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
        epochs=epochs,  # Number of epochs for training
        device=device,  # Set device to cuda or cpu
    )


def main() -> None:
    """
    Script entry point.
    """
    args = parse_args()
    regions = list(set(args.regions) - set(args.skip_regions))

    for region in regions:
        train_yolo(args.training_dir, region, args.overwrite, args.version, args.epochs)


if __name__ == "__main__":
    train_yolo()
