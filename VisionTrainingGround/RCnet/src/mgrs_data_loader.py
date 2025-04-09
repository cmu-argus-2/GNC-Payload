"""
This module defines a custom dataset class for loading and processing images with MGRS grid-based
multi-hot encoding using lat/lon data.

Classes:
    MGRSImageDataset: A PyTorch Dataset for loading images with MGRS grid-based encoding
                     and transformations.
"""

import os
import warnings
import numpy as np
from typing import List, Optional, Tuple, Dict
from functools import lru_cache

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.earth_utils import get_MGRS_grid


class MGRSImageDataset(Dataset):
    """
    A custom dataset for loading images with MGRS grid-based multi-hot encoding.
    
    This dataset loads images and their corresponding lat/lon data from NPZ files.
    It assigns each pixel to an MGRS grid region and creates a fraction-based
    encoding that is processed through a sigmoid function.
    
    Attributes:
        root_dir (str): Path to the dataset directory.
        transform (Optional[object]): Image transformations to be applied.
        salient_regions (List[str]): List of salient MGRS region identifiers.
        files (List[Tuple[str, str]]): List of image file paths.
        mgrs_grid (Dict): Dictionary mapping MGRS region identifiers to bounding boxes.
        sigmoid_params (Dict): Parameters for the sigmoid function.
    """

    def __init__(
        self,
        root_dir: str,
        salient_regions: List[str],
        transform: Optional[object] = None,
        sigmoid_params: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            root_dir (str): Path to the dataset directory.
            salient_regions (List[str]): List of MGRS region identifiers to include in the classification.
            transform: Image transformations.
            sigmoid_params (Dict, optional): Parameters for the sigmoid function.
                                            Defaults to parameters that make sigmoid(0.2)=0.05,
                                            sigmoid(0.25)=0.5, and sigmoid(0.3)=0.95.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.salient_regions = sorted(salient_regions)
        
        # Get the MGRS grid
        self.mgrs_grid = get_MGRS_grid()
        
        # Set sigmoid parameters
        if sigmoid_params is None:
            # Calculate sigmoid parameters to satisfy the requirements:
            # sigmoid(0.2) = 0.05, sigmoid(0.25) = 0.5, sigmoid(0.3) = 0.95
            # Using the formula: sigmoid(x) = 1 / (1 + exp(-k * (x - x0)))
            # Solving for k and x0
            self.sigmoid_params = self._calculate_sigmoid_params(0.2, 0.05, 0.3, 0.95)
        else:
            self.sigmoid_params = sigmoid_params
            
        print(f"Sigmoid parameters: k={self.sigmoid_params['k']:.4f}, x0={self.sigmoid_params['x0']:.4f}")
        
        # Collect images and their corresponding lat/lon files
        self.files = []
        for f in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, f)):
                for img_file in os.listdir(os.path.join(root_dir, f)):
                    if img_file.endswith(".png") or img_file.endswith(".jpg"):
                        img_path = os.path.join(root_dir, f, img_file)
                        base_name = os.path.splitext(img_file)[0]
                        lat_lon_path = os.path.join(root_dir, f, f"{base_name}_lat_lon.npz")
                        
                        if os.path.exists(lat_lon_path):
                            self.files.append((img_path, lat_lon_path))
                        else:
                            warnings.warn(f"Lat/lon data file not found for {img_file}. Skipping.")

    def _calculate_sigmoid_params(self, x1, y1, x2, y2):
        """
        Calculate the parameters for the sigmoid function based on two points.
        
        Args:
            x1, y1: First point (x1, y1) where sigmoid(x1) = y1
            x2, y2: Second point (x2, y2) where sigmoid(x2) = y2
            
        Returns:
            Dict with keys 'k' and 'x0' for the sigmoid function parameters
        """
        # For the sigmoid function: y = 1 / (1 + exp(-k * (x - x0)))
        # We can derive: logit(y) = k * (x - x0)
        # Where logit(y) = log(y / (1 - y))
        
        # Calculate logit values
        logit1 = np.log(y1 / (1 - y1))
        logit2 = np.log(y2 / (1 - y2))
        
        # Solve for k and x0 using two points
        k = (logit2 - logit1) / (x2 - x1)
        x0 = x1 - logit1 / k
        
        return {'k': k, 'x0': x0}

    def _custom_sigmoid(self, x):
        """
        Apply a custom sigmoid function with the calculated parameters.
        
        Args:
            x: Input value or array
            
        Returns:
            Sigmoid transformed value or array
        """
        k = self.sigmoid_params['k']
        x0 = self.sigmoid_params['x0']
        return 1 / (1 + np.exp(-k * (x - x0)))

    @lru_cache(maxsize=128)
    def _get_mgrs_region(self, lat, lon):
        """
        Get the MGRS region for a given latitude and longitude.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            MGRS region identifier or None if not found
        """
        for region, (min_lon, min_lat, max_lon, max_lat) in self.mgrs_grid.items():
            if min_lon <= lon < max_lon and min_lat <= lat < max_lat:
                return region
        return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lat_lon_path = self.files[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        lat_array = lat_lon_array[:, :, 0]
        lon_array = lat_lon_array[:, :, 1]
        
        # Count pixels in each MGRS region
        region_counts = {}
        total_pixels = lat_array.size
        
        # Flatten arrays for faster processing
        lat_flat = lat_array.flatten()
        lon_flat = lon_array.flatten()
        
        for lat, lon in zip(lat_flat, lon_flat):
            region = self._get_mgrs_region(lat, lon)
            if region is not None:
                region_counts[region] = region_counts.get(region, 0) + 1
        
        # Create multi-hot encoded vector with sigmoid transformation
        label_vector = torch.zeros(len(self.salient_regions), dtype=torch.float32)
        
        for i, region in enumerate(self.salient_regions):
            if region in region_counts:
                # Calculate fraction of pixels in this region
                fraction = region_counts[region] / total_pixels
                # Apply sigmoid transformation
                label_vector[i] = self._custom_sigmoid(fraction)
        
        return image, label_vector
