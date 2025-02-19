"""
This module defines a custom dataset class for loading and processing images from a directory.
It supports filtering specific classes, applying transformations, and returning images with
multi-hot encoded labels.

Classes:
    CustomImageDataset: A PyTorch Dataset for loading images with optional class filtering
                        and transformations.
"""

import os
import warnings
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


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
