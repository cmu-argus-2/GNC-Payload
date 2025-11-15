"""
Script for adding dead pixels.
"""

# import random

import torch

# import torchvision.transforms as transforms


class AddDeadPixels(torch.nn.Module):
    """
    Class for adding dead pixels.
    """

    def __init__(self, dead_pixel_percentage: float = 0.05):
        super().__init__()
        self.dead_pixel_percentage = dead_pixel_percentage  # Fraction of pixels to be set to zero

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W).

        Returns:
            Tensor: Image with randomly placed dead pixels.
        """
        _, h, w = img.shape
        total_pixels = h * w
        num_dead_pixels = int(total_pixels * self.dead_pixel_percentage)

        # Generate random indices for dead pixels
        dead_pixel_indices = torch.randint(0, total_pixels, (num_dead_pixels,))
        row_indices = dead_pixel_indices // w
        col_indices = dead_pixel_indices % w

        # Set selected pixel locations to zero across all channels
        img[:, row_indices, col_indices] = 0
        return img
