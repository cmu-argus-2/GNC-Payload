import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive

# Create output directory for visualizations
output_dir = 'augmentation_outputs'
os.makedirs(output_dir, exist_ok=True)

# Define the DirectionalBlur transformation
class DirectionalBlur:
    def __init__(self, velocity, angular_velocity):
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def __call__(self, img):
        # Compute blur kernel parameters
        angle = np.degrees(np.arctan2(self.angular_velocity, self.velocity))
        kernel_size = max(1, int(np.hypot(self.velocity, self.angular_velocity) * 10))
        
        # Get original size and create a larger canvas with more padding
        original_size = img.size
        # Work with RGB PIL image
        img = img.convert('RGB')

        # Convert to numpy for robust padding (use reflect to avoid artificial fill color)
        arr = np.array(img)
        h, w = arr.shape[:2]

        # compute padding but ensure it is less than image dims - numpy.reflect needs pad <= dim-1
        requested_pad = int(max(h, w) * 0.75)
        safe_pad = min(requested_pad, max(1, min(h, w) - 1))

        # pad ((top,bottom),(left,right),(0,0)) with reflect mode to mirror content at edges
        padded = np.pad(arr, ((safe_pad, safe_pad), (safe_pad, safe_pad), (0, 0)), mode='reflect')

        # Convert back to PIL Image
        expanded = Image.fromarray(padded)

        # Rotate, blur, rotate back (expand=False since we padded to accommodate rotation)
        expanded = expanded.rotate(-angle, resample=Image.BICUBIC, expand=False)
        expanded = expanded.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        expanded = expanded.rotate(angle, resample=Image.BICUBIC, expand=False)

        # Crop back to original size from center
        left = (expanded.width - w) // 2
        upper = (expanded.height - h) // 2
        right = left + w
        lower = upper + h
        img = expanded.crop((left, upper, right, lower))

        return img


# Define transformations with DirectionalBlur
# Camera and motion parameters
degrees_per_second = 2.0  # Angular rate
exposure_time = 26e-3  # Camera exposure time 
rotation_during_exposure = degrees_per_second * exposure_time  # Amount of rotation during one frame

# Set parameters for pure rotational motion
velocity = 0.001 # No linear velocity
angular_velocity = np.radians(rotation_during_exposure)  # Convert to radians
#angular_velocity = 3  # This was the default. It returns a horrible blur. Don't default to it. 
train_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    DirectionalBlur(velocity, angular_velocity),
    transforms.ToTensor(),
])

# Function to get files within a specified range
def get_files_in_range(folder_path, start_index, end_index, extension=".tif"):
    try:
        print(f"Looking for {extension} files in: {folder_path}")
        all_files = sorted(f for f in os.listdir(folder_path) if f.endswith(extension))
        print(f"Found {len(all_files)} {extension} files")
        if len(all_files) == 0:
            print("No matching files found!")
            return []
        selected_files = all_files[start_index:end_index + 1]
        return [os.path.join(folder_path, file) for file in selected_files]
    except FileNotFoundError:
        print(f"Error: Directory not found: {folder_path}")
        return []
    except Exception as e:
        print(f"Error accessing files: {str(e)}")
        return []

# Function to visualize original and augmented images
def visualize_augmentations(image_paths):
    if not image_paths:
        print("No images to process!")
        return
    for image_path in image_paths:
        # Load the image
        original_image = Image.open(image_path).convert("RGB")
        
        # Apply the transformations
        augmented_image = train_transform(original_image)
        
        # Convert augmented tensor back to PIL image for visualization
        augmented_image = transforms.ToPILImage()(augmented_image)
        
        # Plot side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(augmented_image)
        axes[1].set_title("Augmented Image (With Directional Blur)")
        axes[1].axis("off")

        plt.tight_layout()
        # Save the figure instead of showing it
        output_path = os.path.join(output_dir, f'augmented_{os.path.basename(image_path)}.png')
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
        print(f"Saved visualization to: {output_path}")

# Specify the folder and range
folder_path = "/home/pvijayba/baja_california_training/filled_baja_california"
start_index = 0  # Start index (inclusive)
end_index = 1  # End index (inclusive)

print("\nStarting image processing...")
# Get the list of files in the specified range
image_files = get_files_in_range(folder_path, start_index, end_index)
print(f"Selected files to process: {image_files}")

# Run the visualization
visualize_augmentations(image_files)
