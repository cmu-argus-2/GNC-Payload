from vision_inference.region_classifier import RegionClassifier
import argparse
import os
import json
from typing import Dict, List, Any
from utils.config_utils import USER_CONFIG_PATH, load_config


def run_region_classification(args: argparse.Namespace, output_dir: str) -> Dict[str, List[str]]:
    """
    Run region classification on images in the specified directory.
    
    Args:
        args: Command line arguments containing data_dir
        output_dir: Output directory path from config
        
    Returns:
        Dict mapping image names to their predicted regions
    """
    # Initialize the region classifier
    classifier = RegionClassifier(load_weights=True)
    
    # Prepare image directory path
    image_dir = os.path.join(output_dir, "images")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Run batch classification
    predictions = classifier.classify_region_batch(
        image_dir=image_dir,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 0
    )
        
    return predictions


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
    results = run_region_classification(args, config.output_dir)
    
    # Print summary
    print(f"Processed {len(results)} images")
    
    # Optionally display first few results
    if len(results) > 0:
        print("\nSample results:")
        for i, (image_name, regions) in enumerate(list(results.items())[:5]):
            print(f"{image_name}: {regions}")
            if i >= 4:
                break
    
    return results


if __name__ == "__main__":
    main()
