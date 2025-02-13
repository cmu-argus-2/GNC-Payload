import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_top_n_salient_regions(image_path, top_n=16):
    # Load the image (Earth image in this case)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    # Initialize the Saliency detection model from OpenCV
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Compute the saliency map (returns a saliency map in the range [0, 255])
    success, saliency_map = saliency.computeSaliency(image)

    # Normalize the saliency map to range [0, 1] for better visualization
    saliency_map = saliency_map.astype(np.float32)
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)

    # Convert the saliency map to a binary mask
    _, binary_mask = cv2.threshold(saliency_map, 0.5, 1, cv2.THRESH_BINARY)

    # Find contours in the saliency map
    contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Select top n contours
    top_contours = contours

    # Draw bounding boxes around the top n salient regions
    image_with_boxes = np.copy(image_rgb)
    for contour in top_contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        # Draw rectangle on the image
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # Show the results
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Earth Image")
    plt.axis('off')

    # Image with Highlighted Top N Salient Regions
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_boxes)
    plt.title(f"Top {top_n} Salient Regions")
    plt.axis('off')

    plt.show()

# Example usage
highlight_top_n_salient_regions('worldmap.jpg', top_n=100)  # Replace with your Earth image path
