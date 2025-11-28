import os
import argparse
import shutil
from PIL import Image

def convert_tif_to_png(source_dir, target_dir, use_sequential_naming=False):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory {source_dir} does not exist.")
        return
    
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Listing files in {source_dir}:")
    tif_files = sorted([f for f in os.listdir(source_dir) if f.endswith((".tif", ".tiff"))])
    print(f"Found {len(tif_files)} TIF files")
    
    # Loop through all files in the source directory
    for idx, filename in enumerate(tif_files):
        # Construct the full file path
        file_path = os.path.join(source_dir, filename)
        base_name = filename.rsplit('.', 1)[0]
        
        # Open the image
        with Image.open(file_path) as img:
            # Convert and save as .png in the target directory
            rgb_im = img.convert('RGB')
            
            if use_sequential_naming:
                # Use sequential naming like 00000.png, 00001.png, etc.
                png_filename = f"{idx:05d}.png"
                new_base_name = f"{idx:05d}"
            else:
                # Keep original name
                png_filename = base_name + '.png'
                new_base_name = base_name
            
            png_path = os.path.join(target_dir, png_filename)
            rgb_im.save(png_path)

        print(f"Converted {filename} to {png_filename}")
        
        # Also copy/rename the corresponding _lat_lon.npz file if it exists
        npz_filename = base_name + "_lat_lon.npz"
        npz_source_path = os.path.join(source_dir, npz_filename)
        
        if os.path.exists(npz_source_path):
            npz_target_filename = new_base_name + "_lat_lon.npz"
            npz_target_path = os.path.join(target_dir, npz_target_filename)
            shutil.copy2(npz_source_path, npz_target_path)
            print(f"  Copied {npz_filename} to {npz_target_filename}")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert .tif images to .png format.")
    parser.add_argument("--source_dir", type=str, help="Path to the source directory containing .tif files.")
    parser.add_argument("--target_dir", type=str, help="Path to the target directory to save .png files.")
    parser.add_argument("--sequential", action="store_true", help="Use sequential naming (00000.png, 00001.png, etc.)")
    
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    convert_tif_to_png(args.source_dir, args.target_dir, args.sequential)
