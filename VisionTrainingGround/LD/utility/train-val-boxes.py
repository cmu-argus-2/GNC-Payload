import cv2
import os
from PIL import Image, ImageDraw

def draw_landmark_boxes(base_path, image_name):
    # Define the paths
    image_path = os.path.join(base_path, 'val/images', image_name)
    label_path = os.path.join(base_path, 'va/labels', os.path.splitext(image_name)[0] + '.txt')
    
    # Read the image
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)
    
    # Read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cls, center_x, center_y, width, height = map(float, line.split())
            
            # Convert normalized values to actual pixel values
            actual_center_x = center_x * image_width
            actual_center_y = center_y * image_height
            actual_width = width * image_width
            actual_height = height * image_height
            
            # Calculate box corners
            left = actual_center_x - (actual_width / 2)
            top = actual_center_y - (actual_height / 2)
            right = actual_center_x + (actual_width / 2)
            bottom = actual_center_y + (actual_height / 2)
            
            # Draw the box
            draw.rectangle([left, top, right, bottom], outline='red', width=2)
    
    # Display the image
    #image.show()

    # Save the resulting image
    save_path = os.path.join('/home/argus-vision/vision/VisionTrainingGround/LD/utility', image_name)
    image.save(save_path)
    print(f"Image saved to {save_path}")

# Example usage
base_path = '/home/argus-vision/vision/VisionTrainingGround/LD/datasets/17R_dataset'
image_name = 'l8_17R_00040.jpg' # Replace with your actual image name
draw_landmark_boxes(base_path, image_name)
