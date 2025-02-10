import os
import random
import argparse
from PIL import Image
import shutil
import re

def convert_tif_to_png(input_path, output_path):
    """
    Converts a .tif image to .jpg format and saves it.

    Args:
        input_path (str): Path to the input .tif file.
        output_path (str): Path to save the converted .jpg file.
    """
    # Open the .tif image
    with Image.open(input_path) as img:
        # Convert to RGB if necessary (to handle images with alpha channel)
        img = img.convert("RGB")
        # Save as .jpg
        img.save(output_path, "PNG")

def split_and_convert_images(root_dir, output_dir, test_ratio=0.2, val_ratio=0.2):
    """
    Processes .tif images, converts them to .jpg, splits into train/val/test, and saves them
    while maintaining the folder structure.

    Args:
        root_dir (str): Path to the root directory containing folders of .tif images.
        output_dir (str): Path to the directory to save train, val, and test splits.
        test_ratio (float): Fraction of images to use for testing.
        val_ratio (float): Fraction of images to use for validation.
    """
    # Get class directories
    classes = os.listdir(root_dir)
    
    # Create train, val, and test directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # Delete existing directories if they exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Create directories for each class
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process each class
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Get all .tif files for this class
        all_tif_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(".tif")]
        
        # Shuffle the files
        random.shuffle(all_tif_files)

        # Compute split indices
        total_count = len(all_tif_files)
        train_count = int(total_count * (1 - (val_ratio + test_ratio)))
        val_count = int(total_count * val_ratio)

        # Split files into train, val, and test sets
        train_files = all_tif_files[:train_count]
        val_files = all_tif_files[train_count:train_count + val_count]
        test_files = all_tif_files[train_count + val_count:]

        # Define output paths for this class
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        # Create class directories in train, val, and test
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Helper function to process files
        def process_files(file_list, target_dir):
            for file_path in file_list:
                # Get the base name of the file (without the path)
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # Check if the file name ends with three digits
                if re.search(r'\d{3}$', file_name):  # Regex to match three digits at the end
                    # Define target path for the .png file
                    target_file_name = file_name + ".png"
                    target_file_path = os.path.join(target_dir, target_file_name)
                    # Convert .tif to .png
                    convert_tif_to_png(file_path, target_file_path)
                else:
                    print(f"Skipping file (does not end with 3 digits): {file_name}")

        # Process the files for each split
        process_files(train_files, train_class_dir)
        process_files(val_files, val_class_dir)
        process_files(test_files, test_class_dir)

        print(f"Processed {len(train_files)} train images, {len(val_files)} val images, and {len(test_files)} test images for class '{class_name}'.")

    print(f"Finished processing. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tif images to .png and create train/val/test splits while maintaining folder structure.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory containing .tif images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for splits.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of images for testing (default: 0.2).")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of images for validation (default: 0.2).")

    args = parser.parse_args()

    # Validate ratios
    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError("The sum of test_ratio and val_ratio must be less than 1.0.")

    # Call the function
    split_and_convert_images(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )
