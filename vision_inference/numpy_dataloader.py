"""
NumPy Image Batch Processing Module

This module provides classes and functions to support batch processing of
numpy image files for inference tasks.

Author: Arvind
Date: April 1, 2025
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Callable, Optional
import cv2
from PIL import Image

from vision_inference.logger import Logger


class NumpyImageDataset(Dataset):
    """
    Dataset class for processing numpy image files in batches.
    """
    
    def __init__(
        self, 
        image_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to numpy image files
            transform: Transformation to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (processed_image, image_path)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load numpy image file
            image = np.load(image_path)
            
            # Convert to PIL Image for transforms

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            
            # Apply transformations if provided
            if self.transform:
                image_tensor = self.transform(pil_image)
            else:
                # Basic conversion to tensor if no transform is provided
                from torchvision import transforms
                image_tensor = transforms.ToTensor()(pil_image)
            
            return image_tensor, image_path
            
        except Exception as e:
            Logger.log("ERROR", f"Error loading image {image_path}: {e}")
            # Return a placeholder tensor and the path
            placeholder = torch.zeros((3, 224, 224))
            if self.transform:
                # Try to match the expected tensor format from transform
                try:
                    dummy_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
                    placeholder = self.transform(dummy_img)
                except:
                    pass
            
            return placeholder, image_path


