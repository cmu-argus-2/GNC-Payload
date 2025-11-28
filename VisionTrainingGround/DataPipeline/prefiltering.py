import cv2
import numpy as np
from PIL import Image

def prefilter_image(image_path, color_threshold=30, contrast_threshold=20):
    """
    Prefilter images based on color variety and contrast.
    
    Args:
        image_path: Path to the image file
        color_threshold: Threshold for color standard deviation (default: 30)
        contrast_threshold: Threshold for contrast standard deviation (default: 20)
    
    Returns:
        tuple: (bool, dict) - (Filter result, additional info)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return False, {"error": "Could not load image"}
    
    # Convert to RGB and HSV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate color standard deviation
    color_std = np.std(img_rgb, axis=(0, 1))
    avg_color_std = np.mean(color_std)
    
    # Calculate contrast (luminance standard deviation)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_std = np.std(gray)
    
    # Get dominant color
    avg_color = np.mean(img_rgb, axis=(0, 1))
    avg_hue = np.mean(img_hsv[:, :, 0])
    avg_saturation = np.mean(img_hsv[:, :, 1])
    avg_value = np.mean(img_hsv[:, :, 2])
    
    info = {
        "color_std": float(avg_color_std),
        "contrast_std": float(contrast_std),
        "avg_color_rgb": avg_color.tolist(),
        "avg_hue": float(avg_hue),
        "avg_saturation": float(avg_saturation),
        "avg_value": float(avg_value),
        "is_significant": False,
        "dominant_type": None
    }
    
    # Check if image has significant colors and contrast
    has_variety = avg_color_std > color_threshold and contrast_std > contrast_threshold
    
    # Check for specific colors
    is_blue = 90 < avg_hue < 130 and avg_saturation > 50  # Blue hue range
    is_black = avg_value < 50  # Low brightness
    is_white = avg_value > 200 and avg_saturation < 30  # High brightness, low saturation
    is_green = 35 < avg_hue < 85 and avg_saturation > 50  # Green hue range
    
    # Decision logic
    if has_variety:
        info["is_significant"] = True
        info["dominant_type"] = "varied"
        return True, info
    elif is_blue:
        info["dominant_type"] = "blue"
        return False, info
    elif is_black:
        info["dominant_type"] = "black"
        return False, info
    elif is_white or is_green:
        info["is_significant"] = True
        info["dominant_type"] = "white" if is_white else "green"
        return True, info
    else:
        info["dominant_type"] = "single_color"
        return False, info


# Example usage:
result, info = prefilter_image("prinks/desktop.jpg")
# print(f"Filter result: {result}")
# print(f"Additional info: {info}")