"""
GNC Vision Pipeline Execution Script

This script executes a complete vision processing pipeline for spacecraft navigation by:
1. Running region classification on images to identify MGRS grid regions
2. Running landmark detection within those identified regions
3. Saving all results for orbit identification

The script expects:
- A set of images stored in <output_dir>/<experiment_name>/images/
- A models directory containing YOLO weights for each region and ground truth landmark data
- A valid configuration in the user config file (models_directory, output_dir)

Directory Structure:
What the script expects to find:
- /output_dir
    - /experiment_name
        - /images                  # Input directory containing camera images
            - <timestep>
                - img_<timestep>_<camera_name>.npy
                - img_<timestep>_<camera_name>.npy
                - ...

What the script will generate:
- /output_dir
    - /experiment_name
        - /vis_inf
            -/<timestep>
                - /region_classification
                    - region_to_images.json
                    - image_to_regions.json
                - /landmark_detections
                    - inf_<timestep>_<camera_name>.npz
                - /bearing_vectors
                    - landmarks_<timestep>_<camera_name>.npz

How to use:
    python run_vision.py --name <experiment_name> --timestep <timestep> 
    [--num_workers <n>] [--batch_ld <n>] [--batch_rc <n>]

Required arguments:
    --name: Experiment name used to organize inputs and outputs
    --timestep: Timestep index to process images for

Optional arguments:
    --num_workers: Number of worker processes for data loading (default: 0)
    --batch_ld: Batch size for landmark detection inference (default: 8)
    --batch_rc: Batch size for region classification inference (default: 8)

Output Files:
- region_to_images.json: Maps each region to its images
- image_to_regions.json: Maps each image to its identified regions
- inf_<timestep>_<camera_name>.npz: Binary files containing landmark detections for each image with arrays for:
    - pixel_coordinates: (N,2) array of x,y points
    - latlons: (N,2) array of lat,lon coordinates
    - class_ids: (N,) array of landmark class IDs
    - region_ids: (N,) array of region IDs
    - confidences: (N,) array of detection confidences
- landmarks_<timestep>_<camera_name>.npz: Binary files containing bearing vectors 
    and landmark positions for each image with arrays for:
    - bearing_vectors: (N,3) array of bearing unit vectors in body frame
    - landmark_positions: (N,3) array of landmark positions in ECI frame
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import brahe
import numpy as np
from brahe.epoch import Epoch
from sensors.camera_model import CameraModel, CameraModelManager
from utils.brahe_utils import increment_epoch  # , load_brahe_data_files
from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import lat_lon_to_ecef

from vision_inference.landmark_detector import LandmarkDetections, LandmarkDetector
from vision_inference.logger import Logger
from vision_inference.region_classifier import RegionClassifier

INPUT_DIR = "images"
OUTPUT_DIR = "vis_inf"


def run_region_classification(args: argparse.Namespace, output_dir: str) -> Dict[str, List[str]]:
    """
    Run region classification on images in the specified directory.

    Args:
        args: Command line arguments containing data_dir
        output_dir: Output directory path from config

    Returns:
        Dict mapping image paths (values) to their predicted regions (key)
    """
    # Initialize the region classifier
    classifier = RegionClassifier(load_weights=True)

    # Prepare image directory path
    images_dir = os.path.join(output_dir, args.name, INPUT_DIR, str(args.timestep))

    # Create image directory if it doesn't exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created image directory: {images_dir}")

    # Run batch classification
    predictions = classifier.classify_region_batch(
        images_dir=images_dir, num_workers=args.num_workers if hasattr(args, "num_workers") else 0
    )
    return predictions


def run_landmark_detection(
    args: argparse.Namespace, output_dir: str, rc_results: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Run landmark detection on images based on their classified regions.

    Args:
        args: Command line arguments
        output_dir: Output directory path from config
        RC_results: Results from region classification (region -> list of image paths)

    Returns:
        Dict of landmark detection results mapping (img_name -> LandmarkDetections)
    """
    detections_by_image = defaultdict(list)
    total_landmarks = 0
    total_images = 0

    # Process each region's images
    for region, image_paths in rc_results.items():
        if not image_paths:  # Skip regions with no images
            continue

        try:
            Logger.log("INFO", f"Processing {len(image_paths)} images for region {region}")
            total_images += len(image_paths)

            # Initialize detector for this region
            detector = LandmarkDetector(region_id=region)

            # Run batch detection
            ld_results = detector.png_detect_landmarks(
                png_paths=[
                    os.path.join(output_dir, args.name, INPUT_DIR, str(args.timestep), img_path)
                    for img_path in image_paths
                ],
                batch_size=args.batch_ld,
            )

            # Count landmarks detected in this region
            region_landmark_count = sum(len(detections) for detections in ld_results.values())
            total_landmarks += region_landmark_count
            Logger.log("INFO", f"Detected {region_landmark_count} landmarks in region {region}")

            for img_name, detections in ld_results.items():
                detections_by_image[img_name].append(detections)

        except Exception as e:  # pylint: disable=W0718
            Logger.log("ERROR", f"Failed to process region {region}: {e}")

    for img_name, detections in detections_by_image.items():
        detections_by_image[img_name] = LandmarkDetections.stack(detections)

    Logger.log(
        "INFO",
        f"Landmark detection complete: {total_landmarks} landmarks detected"
        + f" across {total_images} images in {len(detections_by_image)} regions",
    )
    return detections_by_image


def process_landmark_to_bearing(
    detections: LandmarkDetections, camera_model: CameraModel, curr_epoch: Epoch
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single image's landmark detections into bearing unit vectors and landmark positions.

    Args:
        detections: LandmarkDetections object containing the landmark detections
        camera_model: CameraModel object for the camera that took the image
        curr_epoch: Epoch object representing the time the image was taken

    Returns:
        Tuple containing:
        - bearing_unit_vectors_body: Array of shape (N, 3) containing bearing unit vectors in body frame
        - landmark_positions_eci: Array of shape (N, 3) containing landmark positions in ECI frame
    """
    # Calculate the transformation from ECEF to ECI frame
    ecef_R_eci = brahe.frames.rECItoECEF(curr_epoch)

    # Calculate landmark positions and bearing vectors
    landmark_positions_ecef = lat_lon_to_ecef(detections.latlons)
    landmark_positions_eci = (ecef_R_eci.T @ landmark_positions_ecef.T).T

    bearing_unit_vectors_cf = camera_model.pixel_to_bearing_unit_vector(
        detections.pixel_coordinates
    )
    bearing_unit_vectors_body = (camera_model.body_R_camera @ bearing_unit_vectors_cf.T).T

    return bearing_unit_vectors_body, landmark_positions_eci


def save_region_classification_results(
    rc_reg2img: Dict[str, List[str]],
    rc_img2reg: Dict[str, List[str]],
    output_dir: str,
    args: argparse.Namespace,
) -> None:
    """
    Save region classification results to files.

    Args:
        RC_reg2img: Dict mapping regions to image paths
        RC_img2reg: Dict mapping image paths to regions
        output_dir: Base output directory
        args: Command line arguments
    """
    save_dir = os.path.join(
        output_dir, args.name, OUTPUT_DIR, str(args.timestep), "region_classification"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save both dictionaries as JSON
    with open(os.path.join(save_dir, "region_to_images.json"), "w", encoding="utf-8") as f:
        json.dump(rc_reg2img, f)

    with open(os.path.join(save_dir, "image_to_regions.json"), "w", encoding="utf-8") as f:
        json.dump(rc_img2reg, f)

    Logger.log("INFO", f"Region classification results saved to {save_dir}")


def save_landmark_detections(
    ld_results: Dict[str, LandmarkDetections], output_dir: str, args: argparse.Namespace
) -> None:
    """
    Save landmark detection results to binary files efficiently.

    Args:
        LD_results: Dict mapping image filename to LandmarkDetections object
        output_dir: Base output directory
        args: Command line arguments

    Returns:
        None
    """
    save_dir = os.path.join(
        output_dir, args.name, OUTPUT_DIR, str(args.timestep), "landmark_detections"
    )
    os.makedirs(save_dir, exist_ok=True)

    landmarks_detected = False
    for img_name, detections in ld_results.items():
        if len(detections) == 0:
            continue

        landmarks_detected = True
        # Use the basename without extension as the file name
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        # swap the "img" in the base name with "inf"
        base_name = base_name.replace("img", "inf")
        # Create a unique file path for each image
        file_path = os.path.join(save_dir, f"{base_name}.npz")
        # Save all arrays in a single compressed file
        np.savez_compressed(
            file_path,
            pixels=detections.pixel_coordinates,
            latlons=detections.latlons,
            class_ids=detections.class_ids,
            region_ids=detections.region_ids,
            confidences=detections.confidences,
        )

    if landmarks_detected:
        Logger.log(
            "INFO",
            f"Saved landmark detection results for {len(ld_results)} images to {save_dir}",
        )


# pylint: disable=R0913,R0917
def save_bearing_vectors_and_positions(
    bearing_unit_vectors_body: np.ndarray,
    landmark_positions_eci: np.ndarray,
    output_dir: str,
    experiment_name: str,
    timestep: int,
    camera_name: str,
) -> None:
    """
    Save bearing unit vectors and landmark positions to a single NPZ file.

    Args:
        bearing_unit_vectors_body: Array of shape (N, 3) containing bearing unit vectors in body frame
        landmark_positions_eci: Array of shape (N, 3) containing landmark positions in ECI frame
        output_dir: Base output directory
        experiment_name: Name of the experiment
        timestep: Timestep index from the image filename
        camera_name: Camera name (e.g., "x+")
    """
    # Create the output directory structure
    bearing_dir = os.path.join(
        output_dir, experiment_name, OUTPUT_DIR, str(timestep), "bearing_vectors"
    )
    os.makedirs(bearing_dir, exist_ok=True)

    # Create filename based on the timestep and camera name
    base_name = f"{timestep}_{camera_name}"
    file_path = os.path.join(bearing_dir, f"landmarks_{base_name}.npz")

    # Save both arrays in a single compressed file
    np.savez_compressed(
        file_path,
        bearing_vectors=bearing_unit_vectors_body,
        landmark_positions=landmark_positions_eci,
    )

    Logger.log(
        "INFO",
        f"Saved bearing vectors and landmark positions for timestep {timestep}, camera {camera_name}",
    )
    Logger.log("INFO", f"  - Bearing vectors: {len(bearing_unit_vectors_body)}")
    Logger.log("INFO", f"  - Landmark positions: {len(landmark_positions_eci)}")
    return file_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run region classification on a directory of images"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Path to data directory to store the generated files",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--batch_ld", type=int, default=8, help="Batch size for landmark detection inference"
    )
    parser.add_argument(
        "--batch_rc", type=int, default=8, help="Batch size for region classification inference"
    )

    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help="Timestep index to process images for",
    )
    return parser.parse_args()


# pylint: disable=R0914
def main():
    """Main entry point."""
    args = parse_arguments()
    config = load_config(USER_CONFIG_PATH)
    # check if the output directory exists or else raise an error (the dir is config["output_dir"] + args.name)
    output_basedir = os.path.join(config["output_dir"], args.name)

    if not os.path.exists(output_basedir):
        raise ValueError(
            f"Output directory {output_basedir} does not exist. Please create it first."
        )

    rc_reg2img, rc_img2reg = run_region_classification(args, config["output_dir"])

    # Print summary
    Logger.log(
        "INFO", f"Region classification completed. Found {len(rc_reg2img)} images with regions."
    )

    # # Optionally display first few results
    # if len(RC_reg2img) > 0:
    #     Logger.log("INFO", "First few results:")
    #     for i, (image_name, regions) in enumerate(list(RC_reg2img.items())[:5]):
    #         Logger.log("INFO", f"Image: {image_name}, Regions: {', '.join(regions)}")
    #         if i >= 4:
    #             break

    save_region_classification_results(rc_reg2img, rc_img2reg, config["output_dir"], args)

    rc_dir = os.path.join(output_basedir, OUTPUT_DIR, str(args.timestep), "region_classification")

    # Load region to images mapping
    with open(os.path.join(rc_dir, "region_to_images.json"), "r", encoding="utf-8") as f:
        rc_reg2img = json.load(f)
    ld_results = run_landmark_detection(args, config["output_dir"], rc_reg2img)

    # Print landmark detection summary
    if ld_results:
        # Count images with landmarks
        images_with_landmarks = 0
        # Count only images that have at least one landmark detection
        images_with_landmarks = sum(1 for detections in ld_results.values() if len(detections) > 0)

        Logger.log(
            "INFO",
            f"Found landmarks in {images_with_landmarks} images across {len(ld_results)} regions",
        )

        # Print example of landmark detection results
        for img_name, detections in list(ld_results.items())[:2]:
            if len(detections) > 0:
                Logger.log(
                    "INFO",
                    f"Image: {os.path.basename(img_name)}, "
                    f"Region(s): {np.unique(detections.region_ids)}, "
                    f"Landmarks: {len(detections)}",
                )
                break

    save_landmark_detections(ld_results, config["output_dir"], args)

    # Initialize camera models
    camera_manager = CameraModelManager()
    camera_models = {
        "x+": camera_manager["x+"],
        "y+": camera_manager["y+"],
        "x-": camera_manager["x-"],
        "y-": camera_manager["y-"],
    }

    # Load the experiment parameters once
    try:
        with open(f"{output_basedir}/args.json", "r", encoding="utf-8") as jsonfile:
            arg_data = json.load(jsonfile)
    except Exception as e:
        raise ValueError(f"Error loading args.json for experiment {output_basedir}: {e}") from e

    # Get the starting epoch and time step from args
    starting_epoch = Epoch(*brahe.time.mjd_to_caldate(arg_data["start_date"]))
    dt = 1 / arg_data["frequency"]

    # Process each image with detections to generate bearing vectors and positions
    bearing_paths = []

    for image_name, detections in ld_results.items():
        # Skip images with no detections
        if len(detections) == 0:
            continue

        # Extract timestep and camera name from the image filename
        parts = image_name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {image_name}")

        timestep = int(parts[1])
        camera_name = parts[2].split(".")[0]
        camera_model = camera_models[camera_name]

        # Calculate the epoch for this timestep
        curr_epoch = increment_epoch(starting_epoch, timestep * dt)

        # Process the landmarks to get bearing vectors and positions
        bearing_unit_vectors_body, landmark_positions_eci = process_landmark_to_bearing(
            detections, camera_model, curr_epoch
        )

        # Save the bearing vectors and landmark positions
        file_path = save_bearing_vectors_and_positions(
            bearing_unit_vectors_body,
            landmark_positions_eci,
            config["output_dir"],
            args.name,
            timestep,
            camera_name,
        )

        bearing_paths.append(file_path)

    # Print summary
    if len(bearing_paths) > 0:
        Logger.log(
            "INFO", f"Saved bearing vectors and landmark positions for {len(bearing_paths)} images"
        )
    else:
        Logger.log("INFO", "No bearing vectors or landmark positions saved - no detections found")


if __name__ == "__main__":
    # The load_brahe_data_files() function is currently disabled because it is only required
    # in specific scenarios where Brahe data files need to be preloaded. Uncomment this line
    # if preloading is necessary for your use case.

    # load_brahe_data_files()
    main()
