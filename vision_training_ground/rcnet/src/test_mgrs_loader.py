"""
Test MGRS loader.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.earth_utils import get_mgrs_grid


def calculate_sigmoid_params(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    """Calculate the parameters for the sigmoid function based on two points."""
    logit1 = np.log(y1 / (1 - y1))
    logit2 = np.log(y2 / (1 - y2))

    k = (logit2 - logit1) / (x2 - x1)
    x0 = x1 - logit1 / k

    return {"k": k, "x0": x0}


def custom_sigmoid(x: float, sigmoid_params: dict[str, float]) -> float:
    """Apply a custom sigmoid function with the calculated parameters."""
    k = sigmoid_params["k"]
    x0 = sigmoid_params["x0"]
    return 1 / (1 + np.exp(-k * (x - x0)))


def load_lat_lon_arrays(lat_lon_filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load latitude and longitude arrays.
    """
    # Load using memory mapping for efficiency
    start = time.time()
    lat_lon_data: dict[str, np.ndarray] = np.load(lat_lon_filepath, mmap_mode="r")
    print(f"Loaded lat/lon data in {time.time() - start:.2f} seconds")

    # Extract lat and lon arrays in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # pylint: disable=W0108,E1136
        future_lat = executor.submit(lambda: lat_lon_data["lat_lon"][:, :, 0].copy())
        future_lon = executor.submit(lambda: lat_lon_data["lat_lon"][:, :, 1].copy())

        # Wait for both to complete
        lat_array = future_lat.result()
        lon_array = future_lon.result()

    print(f"Extracted lat/lon arrays in {time.time() - start:.2f} seconds total")
    return lat_array, lon_array


# pylint: disable=R0914
def process_image_and_labels(
    img_filepath: str, lat_lon_filepath: str, sal_regions: list[str]
) -> tuple[Image.Image,]:
    """
    Process image and labels.
    """
    # Load image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            # transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ]
    )
    print("Transformations applied")
    img = Image.open(img_filepath).convert("RGB")
    print("Loaded Image")
    img = transform(img)
    print("Transformed Image")
    # Load lat/lon data
    # lat_lon_data = np.load(lat_lon_path, mmap_mode="r")
    print("Loaded lat/lon data")
    # lat_array = lat_lon_data['lat_lon'][:, :, 0]
    # lon_array = lat_lon_data['lat_lon'][:, :, 1]
    lat_array, lon_array = load_lat_lon_arrays(lat_lon_filepath)
    print("Extracted lat/lon arrays")
    total_pixels = lat_array.size
    print(f"Total pixels: {total_pixels}")
    # Define sigmoid parameters (example values, adjust as needed)
    sigmoid_params = calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)
    print(f"Sigmoid parameters: {sigmoid_params}")
    # Get MGRS grid and prepare salient boundaries
    mgrs_grid = get_mgrs_grid()
    salient_boundaries = {}
    for region in sal_regions:
        if region in mgrs_grid:
            min_lon, min_lat, max_lon, max_lat = mgrs_grid[region]
            salient_boundaries[region] = (min_lon, min_lat, max_lon, max_lat)
    # Vectorized approach to count pixels in each region
    region_counts = {}

    # Only process salient regions instead of all regions
    for region, (min_lon, min_lat, max_lon, max_lat) in salient_boundaries.items():
        # Create masks for lat/lon within region bounds
        lat_mask = (lat_array >= min_lat) & (lat_array < max_lat)
        lon_mask = (lon_array >= min_lon) & (lon_array < max_lon)

        # Combined mask for pixels in this region
        region_mask = lat_mask & lon_mask

        # Count pixels in this region
        pixel_count = np.sum(region_mask)

        if pixel_count > 0:
            region_counts[region] = pixel_count
    print(f"Region counts: {region_counts}")
    # Create multi-hot encoded vector with sigmoid transformation
    lbl_vector = torch.zeros(len(sal_regions), dtype=torch.float32)
    salient_region_indices = {region: i for i, region in enumerate(sal_regions)}

    for region, count in region_counts.items():
        if region in salient_region_indices:
            # Calculate fraction of pixels in this region
            fraction = count / total_pixels
            print(f"Fraction for {region}: {fraction}")
            # Apply sigmoid transformation
            idx = salient_region_indices[region]
            lbl_vector[idx] = custom_sigmoid(fraction, sigmoid_params)

    # Test
    print(custom_sigmoid(0.25, sigmoid_params))

    return img, lbl_vector


# Example usage
if __name__ == "__main__":
    # Define paths
    BASE_PATH = "/mnt/sdb2/training2/10S/"
    IMG_PATH = BASE_PATH + "00028.png"
    LAT_LON_PATH = BASE_PATH + "00028_lat_lon.npz"  # "/home/argus/Arvind/00000_lat_lon.npz"

    # Define salient regions
    salient_regions = [
        "05V",
        "09V",
        "10S",
        "10T",
        "11R",
        "12R",
        "14Q",
        "15V",
        "16T",
        "18Q",
        "18S",
        "19J",
        "21H",
        "23L",
        "29Q",
        "30U",
        "32S",
        "32T",
        "33K",
        "33S",
        "33T",
        "35J",
        "36L",
        "37Q",
        "38K",
        "39P",
        "40R",
        "42R",
        "46Q",
        "48M",
        "49S",
        "50M",
        "51J",
        "52S",
        "53L",
        "54S",
        "54U",
        "55J",
        "57V",
        "59G",
    ]
    salient_regions = sorted(salient_regions)

    # Process the image and get labels
    image, label_vector = process_image_and_labels(IMG_PATH, LAT_LON_PATH, salient_regions)

    # Print results
    print(f"Image size: {image.size}")
    print(f"Label vector: {label_vector}")
    non_zero_indices = torch.nonzero(label_vector).squeeze().tolist()

    if isinstance(non_zero_indices, int):
        positive_regions = [salient_regions[non_zero_indices]]
    else:
        positive_regions = [salient_regions[i] for i in non_zero_indices]

    print(f"Positive regions: {positive_regions}")
# import os
# img_path = "/mnt/sdb2/training2/10S/00028.png"
# region = os.path.basename(os.path.dirname(img_path))
# img_id = os.path.splitext(os.path.basename(img_path))[0]
# print(f"Region: {region}")
# print(f"Image ID: {img_id}")
