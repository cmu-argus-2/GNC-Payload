"""
Train landmark detection YOLO models for the specified MGRS regions.

This script expects to find the following contents in the training directory:
- /training_directory
  - /{region}
    - /LD_training
      - dataset.yaml
      - /train
        - /images
          - 00000.png (symlink)
          - ...
        - /labels
          - 00000.txt
          - ...
      - /test
        - ...
      - /val
        - ...

This scipy will generate/overwrite the following contents in the training directory:
- /training_directory
  - /{region}
    - yolo_model_weights.pt
"""

import argparse
import os

import torch
from ultralytics import YOLO

from utils.config_utils import USER_CONFIG_PATH, load_config
from vision_inference.landmark_detector import LandmarkDetector
from VisionTrainingGround.LD.prepare_yolo_data import LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME


def parse_args():
    """
    Parse command-line arguments.

    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train landmark detection YOLO models for the specified MGRS regions."
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=load_config()["vision"]["salient_mgrs_region_ids"],
        help="MGRS regions to train landmark detection YOLO models for.",
    )
    parser.add_argument(
        "--skip_regions",
        type=str,
        nargs="+",
        default=[],
        help="MGRS regions to skip. This takes precedence over --regions.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--version", type=str, required=False, default="yolov8s", help="The YOLO version to use."
    )
    parser.add_argument(
        "--epochs", type=int, required=False, default=300, help="The number of training epochs."
    )
    return parser.parse_args()


def train_yolo(
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
    - region (str): The MGRS region to train the model for.
    - overwrite (bool): Whether to overwrite the output file if it exists.
    - version (str): The YOLO model version to use.
    - epochs (int): The number of epochs for training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    training_dir = load_config(USER_CONFIG_PATH)["training_directory"]
    output_file = os.path.join(
        training_dir, LandmarkDetector.get_LD_model_weights_relative_path(region)
    )
    if os.path.exists(output_file):
        if not overwrite:
            raise FileExistsError(f"Output file {output_file} already exists.")

        os.remove(output_file)

    model = YOLO(f"{version}.pt")
    yolo_config_path = os.path.join(
        training_dir, region, LD_TRAINING_DIR_NAME, YOLO_CONFIG_FILE_NAME
    )
    # pylint: disable=unused-variable
    results = model.train(
        data=yolo_config_path,  # Dataset path from argument
        project=os.path.dirname(os.path.abspath(output_file)),
        name=os.path.splitext(os.path.basename(output_file))[
            0
        ],  # The result files are saved in project/name
        degrees=180,  # Image augmentation parameters
        scale=0.3,
        fliplr=0.0,
        imgsz=LandmarkDetector.IMAGE_SIZE,
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
        train_yolo(region, args.overwrite, args.version, args.epochs)


if __name__ == "__main__":
    main()
