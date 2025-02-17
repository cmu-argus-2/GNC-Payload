"""
Frame Processing Module for Vision-Based Systems

This module defines a FrameProcessor class that preprocesses video frames for different analytical tasks in vision-based systems.
The primary focus of this class is to filter and prepare frames based on specific criteria such as brightness levels, which can
significantly impact the performance and accuracy of downstream processing tasks, such as ml pipelines and star tracking algorithms.
Frames are processed in batches, with each frame being associated with a camera ID.

Author: Eddie
Date: [Creation or Last Update Date]
"""

from typing import List, Sequence

import cv2
import numpy as np

from vision_inference.frame import Frame
from vision_inference.logger import Logger


def process_for_ml_pipeline(
    frames: Sequence[Frame], dark_threshold: float = 0.5, brightness_threshold: int = 60
) -> List[Frame]:
    """
    Processes frames to select those suitable for machine learning pipeline processing, based on darkness level and potentially other criteria.

    Args:
        frames: The frames to process.
        dark_threshold: The threshold for deciding if a frame is too dark for ML processing. Defaults to 0.5.
        brightness_threshold: The pixel intensity threshold below which pixels are considered dark. Defaults to 60.

    Returns:
        The Frame objects suitable for ML pipeline processing.
    """
    suitable_frames = []
    for frame_obj in frames:
        try:
            gray_frame = cv2.cvtColor(frame_obj.image, cv2.COLOR_BGR2GRAY)
            dark_percentage = np.sum(gray_frame < brightness_threshold) / np.prod(gray_frame.shape)
            if dark_percentage <= dark_threshold:
                suitable_frames.append(frame_obj)
        except Exception as e:
            Logger.log(
                "ERROR",
                f"Error converting image to grayscale or processing error: {e}",
            )

    Logger.log("INFO", f"{len(suitable_frames)} frame(s) selected for ML Pipeline.")
    return suitable_frames


def process_for_star_tracker(
    frames: Sequence[Frame], dark_threshold: float = 0.5, brightness_threshold: int = 60
) -> List[Frame]:
    """
    Processes frames to select those potentially suitable for star tracker processing or other uses where high darkness levels are acceptable or required.

    Args:
        frames: The frames to process.
        dark_threshold: The threshold for selecting darker frames suitable for tasks like star tracking. Defaults to 0.5.
        brightness_threshold: The pixel intensity threshold below which pixels are considered dark. Defaults to 60.

    Returns:
        The Frame objects suitable for star tracker processing or similar tasks.
    """
    suitable_frames = []
    for frame_obj in frames:
        try:
            gray_frame = cv2.cvtColor(frame_obj.image, cv2.COLOR_BGR2GRAY)
            dark_percentage = np.sum(gray_frame < brightness_threshold) / np.prod(gray_frame.shape)
            if dark_percentage > dark_threshold:
                suitable_frames.append(frame_obj)
        except Exception as e:
            Logger.log(
                "ERROR",
                f"Error converting image to grayscale or processing error: {e}",
            )

    Logger.log("INFO", f"{len(suitable_frames)} frame(s) selected for Star Tracker.")
    return suitable_frames
