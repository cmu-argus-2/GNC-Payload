"""
This module defines a custom dataset class for loading and processing images from a directory.
It supports filtering specific classes, applying transformations, and returning images with
multi-hot encoded labels.

Classes:
    CustomImageDataset: A PyTorch Dataset for loading images with optional class filtering
                        and transformations.
"""

import numpy as np
import os
import random
import warnings
import time
from typing import List, Optional, Tuple, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.earth_utils import get_MGRS_grid

class CustomImageDataset(Dataset):
    """
    A custom dataset for loading images from a directory, supporting class selection and
    transformations.

    Attributes:
        root_dir (str): Path to the dataset directory.
        transform (Optional[object]): Image transformations to be applied.
        classes (List[str]): Sorted list of selected class names.
        class_to_idx (dict): Mapping from class names to indices.
        files (List[Tuple[str, str]]): List of image file paths and corresponding labels.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Loads and returns an image along with its multi-hot encoded label.
    """

    def __init__(
        self,
        root_dir: str,
        selected_classes: Optional[List[str]] = None,
        transform: Optional[object] = None,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the dataset directory.
            selected_classes (list, optional): List of class names to include in the classification.
                                               If None, all available classes are used.
            transform: Image transformations.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all available class names
        all_classes = sorted(os.listdir(root_dir))
        print("Selected Classes", selected_classes)
        # If selected_classes is not provided, use all available classes and show a warning
        if not selected_classes:
            warnings.warn("No selected classes provided. Using all available classes.")
            selected_classes = all_classes  # Default to using all classes

        # Ensure selected_classes is a subset of available classes
        self.classes = sorted([cls for cls in selected_classes if cls in all_classes])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print("Using classes:", self.classes)

        # Collect images and their corresponding labels
        self.files = []
        for label in all_classes:  # Iterate over all available classes
            label_path = os.path.join(root_dir, label)
            for f in os.listdir(label_path):
                if f.endswith(".png") or f.endswith(".jpg"):
                    img_path = os.path.join(label_path, f)
                    self.files.append((img_path, label))  # Store file path and label name

        print(f"Total number of images found: {len(self.files)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Create a multi-hot encoded vector
        label_vector = torch.zeros(len(self.classes), dtype=torch.float32)  # Default: all zeros
        if label in self.class_to_idx:
            class_idx = self.class_to_idx[label]
            label_vector[class_idx] = 1  # Set corresponding class to 1 if it's a selected class

        return image, label_vector  # Return image and multi-hot vector

class MGRSImageDataset(Dataset):
    """A custom dataset for loading images with MGRS grid-based multi-hot encoding."""

    def __init__(
        self,
        root_dir: str,
        salient_regions: List[str],
        transform: Optional[object] = None,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> None:
        """
        Args:
            root_dir (str): Path to the dataset directory.
            salient_regions (List[str]): List of MGRS regions to consider.
            transform (Optional[object]): Optional transforms to apply to images.
            split (str): One of 'train', 'val', or 'test'.
            train_ratio (float): Ratio of data to use for training.
            val_ratio (float): Ratio of data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.salient_regions = sorted(salient_regions)
        
        # Get the MGRS grid
        self.mgrs_grid = get_MGRS_grid()
        
        # Create mapping from region to index
        self.salient_region_indices = {region: i for i, region in enumerate(self.salient_regions)}
        
        # Set sigmoid parameters
        self.sigmoid_params = self._calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)

            
        print(f"Sigmoid parameters: k={self.sigmoid_params['k']:.4f}, x0={self.sigmoid_params['x0']:.4f}")
        
        # Collect images and their corresponding lat/lon files
        self.files = []
        self.region_label_cache = {}
        self.region_label_updated = {} # To check if labels are updated
        for region_dir in os.listdir(root_dir):
            region_path = os.path.join(root_dir, region_dir)
            if not os.path.isdir(region_path):
                continue

            cache_path = os.path.join(region_path, f"vector_labels.npz")
            if os.path.exists(cache_path):
                self.region_label_cache[region_dir] = dict(np.load(cache_path))
            else:
                self.region_label_cache[region_dir] = {}
            self.region_label_updated[region_dir] = False
                
            for img_file in os.listdir(region_path):
                if img_file.endswith((".png", ".jpg")):
                    img_path = os.path.join(region_path, img_file)
                    base_name = os.path.splitext(img_file)[0]
                    lat_lon_path = os.path.join(region_path, f"{base_name}_lat_lon.npz")
                        
                    if os.path.exists(lat_lon_path):
                        self.files.append((img_path, lat_lon_path))
                    else:
                        warnings.warn(f"Lat/lon data file not found for {img_file}. Skipping.")
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Shuffle files
        random.shuffle(self.files)
        
        # Calculate split sizes
        total_size = len(self.files)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split the data
        if split == 'train':
            self.files = self.files[:train_size]
        elif split == 'val':
            self.files = self.files[train_size:train_size + val_size]
        else:  # test
            self.files = self.files[train_size + val_size:]
            
        print(f"Total {split} images: {len(self.files)}")
        
        # Precompute region boundaries for fast lookup
        self._precompute_region_boundaries()

    def _parse_region_and_id(self, img_path: str) -> Tuple[str, str]:
        region = os.path.basename(os.path.dirname(img_path))
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return region, img_id

    def _precompute_region_boundaries(self):
        """Precompute region boundaries for vectorized operations."""
        # Extract only salient regions for faster processing
        self.salient_boundaries = {}
        for region in self.salient_regions:
            if region in self.mgrs_grid:
                min_lon, min_lat, max_lon, max_lat = self.mgrs_grid[region]
                self.salient_boundaries[region] = (min_lon, min_lat, max_lon, max_lat)

    def _calculate_sigmoid_params(self, x1, y1, x2, y2):
        """Calculate the parameters for the sigmoid function based on two points."""
        logit1 = np.log(y1 / (1 - y1))
        logit2 = np.log(y2 / (1 - y2))
        
        k = (logit2 - logit1) / (x2 - x1)
        x0 = x1 - logit1 / k
        
        return {'k': k, 'x0': x0}

    def _custom_sigmoid(self, x):
        """Apply a custom sigmoid function with the calculated parameters."""
        k = self.sigmoid_params['k']
        x0 = self.sigmoid_params['x0']
        return 1 / (1 + np.exp(-k * (x - x0)))

    def __len__(self) -> int:
        return len(self.files)
        
    def _compute_label(self, lat_lon_path: str) -> torch.Tensor:
        with np.load(lat_lon_path, mmap_mode='r') as lat_lon_data:
            start = time.time()
            lat_lon_array = lat_lon_data['lat_lon']  # shape: (H, W, 2)
            end = time.time()
            print(f"Loaded lat/lon data in {end - start:.4f} seconds")
            total_pixels = lat_lon_array.shape[0] * lat_lon_array.shape[1]
            lat = lat_lon_array[:, :, 0]
            lon = lat_lon_array[:, :, 1]

            label_vector = torch.zeros(len(self.salient_regions), dtype=torch.float32)
            start = time.time()
            # Vectorized processing for each region
            for region, (min_lon, min_lat, max_lon, max_lat) in self.salient_boundaries.items():
                mask = (
                    (lat >= min_lat) & (lat < max_lat) &
                    (lon >= min_lon) & (lon < max_lon)
                )
                pixel_count = mask.sum()
                if pixel_count > 0:
                    idx = self.salient_region_indices[region]
                    fraction = pixel_count.item() / total_pixels
                    label_vector[idx] = self._custom_sigmoid(fraction)
            end = time.time()
            print(f"Computed label vector in {end - start:.4f} seconds")
            return label_vector



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lat_lon_path = self.files[idx]
        region, img_id = self._parse_region_and_id(img_path)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Try loading label
        if img_id in self.region_label_cache[region]:
            print(f"Loading label from cache for {img_id}")
            label_vector = torch.from_numpy(self.region_label_cache[region][img_id]).float()
        else:
            label_vector = self._compute_label(lat_lon_path)
            vec_np = label_vector.numpy()
            self.region_label_cache[region][img_id] = vec_np
            self.region_label_updated[region] = True

        return image, label_vector
