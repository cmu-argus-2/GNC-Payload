import os
import argparse
from PIL import Image

def convert_tif_to_jpg(source_dir, target_dir):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory {source_dir} does not exist.")
        return
    
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Listing files in {source_dir}:")
    for file in os.listdir(source_dir):
        print(file)

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            # Construct the full file path
            file_path = os.path.join(source_dir, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Convert and save as .jpg in the target directory
                rgb_im = img.convert('RGB')
                jpg_filename = filename[:-4] + '.jpg'
                jpg_path = os.path.join(target_dir, jpg_filename)
                rgb_im.save(jpg_path, quality=95)

            print(f"Converted {filename} to {jpg_filename} in {target_dir}")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert .tif images to .jpg format.")
    parser.add_argument("--source_dir", type=str, help="Path to the source directory containing .tif files.")
    parser.add_argument("--target_dir", type=str, help="Path to the target directory to save .jpg files.")
    
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    convert_tif_to_jpg(args.source_dir, args.target_dir)
