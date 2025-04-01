from vision_inference.region_classifier import RegionClassifier
from vision_inference.landmark_detector import LandmarkDetector

import argparse
import os
import json
from typing import Dict, List, Any
from utils.config_utils import USER_CONFIG_PATH, load_config
from vision_inference.logger import Logger
import numpy as np

INPUT_DIR = "images"
OUTPUT_DIR = "vision_inference_output"


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

def run_landmark_detection(args: argparse.Namespace, models_dir: str, output_dir: str, region_results: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Run landmark detection on images based on their classified regions.
    
    Args:
        args: Command line arguments
        output_dir: Output directory path from config
        region_results: Results from region classification (region -> list of image paths)
        
    Returns:
        Dict of landmark detection results by region
    """
    detections_by_image = {}
    total_landmarks = 0
    total_images = 0
    
    # Process each region's images
    for region, image_paths in region_results.items():
        region_dir = os.path.join(models_dir, region)
        if not image_paths:  # Skip regions with no images
            continue
            
        try:
            Logger.log("INFO", f"Processing {len(image_paths)} images for region {region}")
            total_images += len(image_paths)
            
            # Initialize detector for this region
            detector = LandmarkDetector(region_id=region_dir)
            
            # Run batch detection
            region_results = detector.batch_detect_landmarks(
                npy_paths= [os.path.join(output_dir, args.name, INPUT_DIR, img_path) for img_path in image_paths],
                batch_size=args.batch_size if hasattr(args, "batch_size") else 8,
            )
            
            # Count landmarks detected in this region
            region_landmark_count = sum(len(detections) for detections in region_results.values())
            total_landmarks += region_landmark_count
            Logger.log("INFO", f"Detected {region_landmark_count} landmarks in region {region}")
            
            detections_by_image[region] = region_results
            
        except Exception as e:
            Logger.log("ERROR", f"Failed to process region {region}: {e}")
    
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
    
    # Create a lookup file to record what's saved where
    lookup = {}
    
    for region, region_results in LD_results.items():
        # Create region directory
        region_dir = os.path.join(save_dir, region)
        os.makedirs(region_dir, exist_ok=True)
        
        region_lookup = {}
        for img_name, detections in region_results.items():
            if len(detections) == 0:
                continue
                
            # Create a compact numpy file with all detection data
            # Use the basename without extension as the key
            base_name = os.path.splitext(os.path.basename(img_name))[0]
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
            
            region_lookup[base_name] = len(detections)
        
        # Save region lookup with counts
        lookup[region] = region_lookup
    
    # Save the lookup file as JSON
    with open(os.path.join(save_dir, "detection_counts.json"), "w") as f:
        json.dump(lookup, f)
    
    Logger.log("INFO", f"Landmark detection results saved to {save_dir}")



def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run region classification on a directory of images")

    parser.add_argument("--name", type=str, required=True, 
                        help="Path to data directory to store the generated files")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading")
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
        for region, region_data in LD_results.items():
            images_with_landmarks += sum(1 for img_data in region_data.values() if len(img_data) > 0)
        
        Logger.log("INFO", f"Found landmarks in {images_with_landmarks} images across {len(LD_results)} regions")
        
        # Print example of landmark detection results
        for region, region_data in list(LD_results.items())[:1]:
            example_items = list(region_data.items())[:2]
            for img_name, detections in example_items:
                if len(detections) > 0:
                    Logger.log("INFO", f"Example - Image: {os.path.basename(img_name)}, "
                                        f"Region: {region}, Landmarks: {len(detections)}")
                    break

    save_landmark_detections(LD_results, config["output_dir"], args.name)
    
    return


if __name__ == "__main__":
    main()
