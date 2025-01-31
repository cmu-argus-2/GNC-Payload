import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Define the DirectionalBlur transformation
class DirectionalBlur:
    def __init__(self, velocity, angular_velocity):
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def __call__(self, img):
        # Compute blur kernel parameters
        angle = np.degrees(np.arctan2(self.angular_velocity, self.velocity))
        kernel_size = max(1, int(np.hypot(self.velocity, self.angular_velocity) * 10))
        
        # Expand the canvas to avoid cropping
        original_size = img.size
        expanded_size = (int(original_size[0] * 1.5), int(original_size[1] * 1.5))  # Increase canvas size by 50%
        img = img.resize(expanded_size, Image.BICUBIC)
        
        # Rotate to align blur direction
        img = img.rotate(-angle, resample=Image.BICUBIC, expand=True)
        
        # Apply Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        
        # Rotate back
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        # Crop back to original size
        # left = (img.size[0] - original_size[0]) // 2
        # upper = (img.size[1] - original_size[1]) // 2
        # right = left + original_size[0]
        # lower = upper + original_size[1]
        # img = img.crop((left, upper, right, lower))
        
        return img


# Define transformations with DirectionalBlur
velocity = 3  # Example input velocity
angular_velocity = 0  # Example input angular velocity

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    DirectionalBlur(velocity, angular_velocity),
    transforms.ToTensor(),
])

# Function to get files within a specified range
def get_files_in_range(folder_path, start_index, end_index, extension=".tif"):
    all_files = sorted(f for f in os.listdir(folder_path) if f.endswith(extension))
    selected_files = all_files[start_index:end_index + 1]
    return [os.path.join(folder_path, file) for file in selected_files]

# Function to visualize original and augmented images
def visualize_augmentations(image_paths):
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
        plt.show()

# Specify the folder and range
folder_path = "Landsat_Data/53L"
start_index = 0  # Start index (inclusive)
end_index = 1    # End index (inclusive)

# Get the list of files in the specified range
image_files = get_files_in_range(folder_path, start_index, end_index)

# Run the visualization
visualize_augmentations(image_files)
