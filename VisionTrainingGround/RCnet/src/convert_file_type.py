from PIL import Image
import os

def convert_tif_to_jpg(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    print(f"Listing files in {folder_path}:")
    for file in os.listdir(folder_path):
        print(file)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            print("here")
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Convert and save as .jpg
                rgb_im = img.convert('RGB')
                jpg_filename = filename[:-4] + '.jpg'
                jpg_path = os.path.join(folder_path, jpg_filename)
                rgb_im.save(jpg_path, quality=95)

            print(f"Converted {filename} to {jpg_filename}")

# Example usage
folder_path = '/home/argus-vision/vision/VisionTrainingGround/RCnet/data/test/17R'
fp_2  = '/home/argus-vision/vision/VisionTrainingGround/RCnet/data/test/53S'
fp_3 = '/home/argus-vision/vision/VisionTrainingGround/RCnet/data/test/54S'
convert_tif_to_jpg(folder_path)
convert_tif_to_jpg(fp_2)
convert_tif_to_jpg(fp_3)
