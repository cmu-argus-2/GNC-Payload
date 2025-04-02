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
- /output_dir
    - /experiment_name
        - /images                  # Input directory containing camera images
            - img_<timestep>_<camera_name>.npy
            - img_<timestep>_<camera_name>.npy
            - ...
        - /vis_inf                 # Output directory for vision inference results
            - /region_classification
                - region_to_images.json
                - image_to_regions.json
            - /landmark_detections
                - inf_<timestep>_<camera_name>.npz
                - inf_<timestep>_<camera_name>.npz
                - ...

How to use:
    python run_vision.py --name <experiment_name> [--num_workers <n>] [--batch_ld <n>] [--batch_rc <n>]

Required arguments:
    --name: Experiment name used to organize inputs and outputs

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
"""
from vision_inference.region_classifier import RegionClassifier
from vision_inference.landmark_detector import LandmarkDetector, LandmarkDetections

import argparse
import os
import json
from typing import Dict, List, Any
from utils.config_utils import USER_CONFIG_PATH, load_config
from vision_inference.logger import Logger
import numpy as np
from collections import defaultdict

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
    images_dir = os.path.join(output_dir, args.name, INPUT_DIR)
    
    # Create image directory if it doesn't exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created image directory: {images_dir}")
    
    # Run batch classification
    predictions = classifier.classify_region_batch(
        images_dir=images_dir,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 0
    )
    return predictions

def run_landmark_detection(args: argparse.Namespace, models_dir: str, output_dir: str, RC_results: Dict[str, List[str]]) -> Dict[str, Dict]:
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
    for region, image_paths in RC_results.items():
        if not image_paths:  # Skip regions with no images
            continue
            
        try:
            Logger.log("INFO", f"Processing {len(image_paths)} images for region {region}")
            total_images += len(image_paths)
            
            # Initialize detector for this region
            detector = LandmarkDetector(region_id=region)
            
            # Run batch detection
            LD_results = detector.batch_detect_landmarks(
                npy_paths= [os.path.join(output_dir, args.name, INPUT_DIR, img_path) for img_path in image_paths],
                batch_size=args.batch_ld,
            )
            
            # Count landmarks detected in this region
            region_landmark_count = sum(len(detections) for detections in LD_results.values())
            total_landmarks += region_landmark_count
            Logger.log("INFO", f"Detected {region_landmark_count} landmarks in region {region}")
            
            for img_name, detections in LD_results.items():
                detections_by_image[img_name].append(detections)
            
        except Exception as e:
            Logger.log("ERROR", f"Failed to process region {region}: {e}")

    for img_name, detections in detections_by_image.items():
        detections_by_image[img_name] = LandmarkDetections.stack(detections)

    Logger.log("INFO", f"Landmark detection complete: {total_landmarks} landmarks detected across {total_images} images in {len(detections_by_image)} regions")
    return detections_by_image

def save_region_classification_results(RC_reg2img, RC_img2reg, output_dir, name):
    """
    Save region classification results to files.
    
    Args:
        RC_reg2img: Dict mapping regions to image paths
        RC_img2reg: Dict mapping image paths to regions
        output_dir: Base output directory
        name: Experiment name for subdirectory
    """
    save_dir = os.path.join(output_dir, name, OUTPUT_DIR, "region_classification")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save both dictionaries as JSON
    with open(os.path.join(save_dir, "region_to_images.json"), "w") as f:
        json.dump(RC_reg2img, f)
    
    with open(os.path.join(save_dir, "image_to_regions.json"), "w") as f:
        json.dump(RC_img2reg, f)
    
    Logger.log("INFO", f"Region classification results saved to {save_dir}")

def save_landmark_detections(LD_results, output_dir, name):
    """
    Save landmark detection results to binary files efficiently.
    
    Args:
        LD_results: Dict of dicts mapping region -> {filename -> LandmarkDetections}
        output_dir: Base output directory
        name: Experiment name for subdirectory
    """
    save_dir = os.path.join(output_dir, name, OUTPUT_DIR, "landmark_detections")
    os.makedirs(save_dir, exist_ok=True)
    
    
    for img_name, detections in LD_results.items():
        if len(detections) == 0:
            continue
            
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
            confidences=detections.confidences
        )

    
    Logger.log("INFO", f"Landmark detection results saved to {save_dir}")



def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run region classification on a directory of images")

    parser.add_argument("--name", type=str, required=True, 
                        help="Path to data directory to store the generated files")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading")
    parser.add_argument("--batch_ld", type=int, default=8,
                        help="Batch size for landmark detection inference")
    parser.add_argument("--batch_rc", type=int, default=8,
                        help="Batch size for region classification inference")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    config = load_config(USER_CONFIG_PATH)
    RC_reg2img, RC_img2reg = run_region_classification(args, config["output_dir"])
    
    # Print summary
    Logger.log("INFO", f"Region classification completed. Found {len(RC_reg2img)} images with regions.")
    
    # Optionally display first few results
    if len(RC_reg2img) > 0:
        Logger.log("INFO", "First few results:")
        for i, (image_name, regions) in enumerate(list(RC_reg2img.items())[:5]):
            Logger.log("INFO", f"Image: {image_name}, Regions: {', '.join(regions)}")
            if i >= 4:
                break
    
    save_region_classification_results(RC_reg2img, RC_img2reg, config["output_dir"], args.name)


    LD_results = run_landmark_detection(args, config["models_directory"], config["output_dir"], RC_reg2img)
        
    # Print landmark detection summary
    if LD_results:
        # Count images with landmarks
        images_with_landmarks = 0
        # Count only images that have at least one landmark detection
        images_with_landmarks = sum(1 for detections in LD_results.values() if len(detections) > 0)
        
        Logger.log("INFO", f"Found landmarks in {images_with_landmarks} images across {len(LD_results)} regions")
        
        # Print example of landmark detection results
        for img_name, detections in list(LD_results.items())[:2]:
            if len(detections) > 0:
                Logger.log("INFO", f"Example - Image: {os.path.basename(img_name)}, "
                                   f"Region(s): {np.unique(detections.region_ids)}, "
                                   f"Landmarks: {len(detections)}")
                break

    save_landmark_detections(LD_results, config["output_dir"], args.name)
    
    return


if __name__ == "__main__":
    main()
