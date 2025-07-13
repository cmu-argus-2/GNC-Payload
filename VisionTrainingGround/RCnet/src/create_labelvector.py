import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from typing import List
import warnings
import time

from utils.earth_utils import get_MGRS_grid  
from utils.config_utils import load_config, MAIN_CONFIG_PATH

def calculate_sigmoid_params(x1, y1, x2, y2):
    logit1 = np.log(y1 / (1 - y1))
    logit2 = np.log(y2 / (1 - y2))
    k = (logit2 - logit1) / (x2 - x1)
    x0 = x1 - logit1 / k
    return k, x0

def custom_sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def precompute_region_boundaries(salient_regions, mgrs_grid):
    boundaries = {}
    for region in salient_regions:
        if region in mgrs_grid:
            min_lon, min_lat, max_lon, max_lat = mgrs_grid[region]
            boundaries[region] = (min_lon, min_lat, max_lon, max_lat)
    return boundaries

def compute_label(lat_lon_path, salient_boundaries, region_indices, k, x0):
    with np.load(lat_lon_path, mmap_mode='r') as lat_lon_data:
        start = time.time()
        lat_lon_array = lat_lon_data['lat_lon']
        end = time.time()
        print(f"Loaded lat_lon data in {end - start:.2f} seconds")
        total_pixels = lat_lon_array.shape[0] * lat_lon_array.shape[1]
        lat = lat_lon_array[:, :, 0]
        lon = lat_lon_array[:, :, 1]

        label_vector = torch.zeros(len(region_indices), dtype=torch.float32)
        start = time.time()
        for region, (min_lon, min_lat, max_lon, max_lat) in salient_boundaries.items():
            mask = (
                (lat >= min_lat) & (lat < max_lat) &
                (lon >= min_lon) & (lon < max_lon)
            )
            pixel_count = mask.sum()
            if pixel_count > 0:
                idx = region_indices[region]
                fraction = pixel_count.item() / total_pixels
                label_vector[idx] = custom_sigmoid(fraction, k, x0)
        end = time.time()
        print(f"Computed label vector in {end - start:.2f} seconds")
        return label_vector.numpy()

def main(root_dir, salient_regions):
    salient_regions = sorted(salient_regions)
    region_indices = {region: i for i, region in enumerate(salient_regions)}
    mgrs_grid = get_MGRS_grid()
    boundaries = precompute_region_boundaries(salient_regions, mgrs_grid)
    k, x0 = calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)

    for region_dir in tqdm(sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]), desc="Regions"):
        # Check if there already is a vector_labels.npz file
        region_labels_path = os.path.join(root_dir, region_dir, "vector_labels.npz")
        if os.path.isfile(region_labels_path):
            print(f"Skipping {region_dir} as vector_labels.npz already exists.")
            continue
        region_path = os.path.join(root_dir, region_dir)
        if not os.path.isdir(region_path):
            continue
        print("Region:", os.path.basename(region_dir))
        cache = {}
        for file in sorted(os.listdir(region_path)):
            if file.endswith(("_lat_lon.npz")):
                print(f"Processing {file} in region {region_dir}")
                base = file.replace("_lat_lon.npz", "")
                lat_lon_path = os.path.join(region_path, file)
                label = compute_label(lat_lon_path, boundaries, region_indices, k, x0)
                cache[base] = label

        if cache:
            np.savez(os.path.join(region_path, "vector_labels.npz"), **cache)
            print(f"Saved labels for {region_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/mnt/sdb2/training2/", help="Dataset root directory")
    config = load_config(MAIN_CONFIG_PATH)
    args = parser.parse_args()

    main(args.root_dir, config["vision"]["salient_mgrs_region_ids"])
