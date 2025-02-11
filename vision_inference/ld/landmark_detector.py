"""
Landmark Detection Module

This module defines the LandmarkDetector class, which utilizes a pretrained YOLO (You Only Look Once) model
to detect and process landmarks within images. The detector extracts landmarks as bounding boxes along with
their associated class IDs and confidence scores. The main functionality revolves around the detection of
landmarks in given images and the extraction of useful information such as centroids and class/confidence data.

Dependencies:
- numpy: Used for array manipulations and handling numerical operations.
- cv2 (OpenCV): Required for image processing tasks.
- ultralytics YOLO: The YOLO model implementation from Ultralytics, used for object detection tasks. (Large package warning)

Author: Eddie, Haochen
Date: [Creation or Last Update Date]
"""

import csv
import os
from time import perf_counter
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from vision_inference.logger import Logger
from vision_inference.frame import Frame

LD_MODEL_SUF = "_nadir.pt"

# Define error and info messages
error_messages = {
    "CONFIGURATION_ERROR": "Configuration error.",
    "LOADING_FAILED": "Failed to load necessary data.",
    "DETECTION_FAILED": "Detection process failed.",
    "INVALID_DIMENSIONS": "Invalid bounding box dimensions detected.",
    "LOW_CONFIDENCE": "Skipping low confidence landmark.",
    "EMPTY_DETECTION": "No landmark detected.",
}

info_messages = {
    "INITIALIZATION_START": "Initializing LandmarkDetector.",
    "DETECTION_START": "Starting the landmark detection process.",
    "DETECTION_COMPLETE": "Landmark detection completed successfully.",
}


class LandmarkDetector:
    MODEL_DIR = os.path.abspath(os.path.join(__file__, "../../models/ld"))

    def __init__(self, region_id: str):
        """
        Initialize the LandmarkDetector with a specific region ID
        The YOLO object is created with the path to a specific pretrained model
        """
        Logger.log("INFO", f"Initializing LandmarkDetector for region {region_id}.")

        self.region_id = region_id
        try:
            self.model = YOLO(
                os.path.join(LandmarkDetector.MODEL_DIR, region_id, f"{region_id}_nadir.pt")
            )
            self.ground_truth = LandmarkDetector.load_ground_truth(
                os.path.join(LandmarkDetector.MODEL_DIR, region_id, f"{region_id}_top_salient.csv")
            )
        except Exception as e:
            Logger.log("ERROR", f"{error_messages['LOADING_FAILED']}: {e}")
            raise

    @staticmethod
    def load_ground_truth(ground_truth_path: str) -> np.ndarray:
        """
        Loads ground truth bounding box coordinates from a CSV file.

        Args:
            ground_truth_path (str): Path to the ground truth CSV file.

        Returns:
            A numpy array of shape (N, 6) containing the following for each landmark:
            (centroid_lon, centroid_lat, top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat).
        """
        try:
            # TODO: change csvs to have lat, lon instead of lon, lat for consistency
            return np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)
        except Exception as e:
            Logger.log("ERROR", f"{error_messages['CONFIGURATION_ERROR']}: {e}")
            raise

    def detect_landmarks(self, frame_obj: Frame):
        """
        Detects landmarks in an input image using a pretrained YOLO model and extracts relevant information.

        Args:
            img (np.ndarray): The input image array on which to perform landmark detection.

        Returns:
            tuple: A tuple containing several numpy arrays:
                - centroid_xy (np.ndarray): Array of [x, y] coordinates for the centroids of detected landmarks.
                - centroid_latlons (np.ndarray): Array of geographical coordinates [latitude, longitude] for each detected landmark's centroid, based on class ID.
                - landmark_class (np.ndarray): Array of class IDs for each detected landmark.
                - confidence_scores

        The detection process filters out landmarks with low confidence scores (below 0.5) and invalid bounding box dimensions. It aims to provide a comprehensive set of data for each detected landmark, facilitating further analysis or processing.
        """
        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {info_messages['DETECTION_START']}",
        )

        try:
            # Detect landmarks using the YOLO model
            img = Image.fromarray(cv2.cvtColor(frame_obj.frame, cv2.COLOR_BGR2RGB))
            start_time = perf_counter()
            results = self.model.predict(img, conf=0.5, imgsz=(1088, 1920), verbose=False)
            inference_time = perf_counter() - start_time

            landmark_list = []

            for result in results:
                landmarks = result.boxes

                if landmarks:  # Sanity Check
                    # Iterate over each detected bounding box (landmark)
                    for landmark in landmarks:
                        x, y, w, h = landmark.xywh[0]
                        cls = landmark.cls[0].item()
                        conf = landmark.conf[0].item()

                        if w < 0 or h < 0:
                            Logger.log(
                                "INFO", "Skipping landmark with invalid bounding box dimensions."
                            )
                            continue

                        landmark_list.append([x, y, cls, w, h, conf])

            if not landmark_list:
                Logger.log(
                    "INFO",
                    f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] No landmarks detected in Region {self.region_id}.",
                )
                return None, None, None, None

            landmark_arr = np.array(landmark_list)

            centroid_xy = landmark_arr[:, :2]
            landmark_class = landmark_arr[:, 2].astype(int)
            confidence_scores = landmark_arr[:, 5]  # Confidence scores

            # Additional processing to calculate bounding box corners and lat/lon coordinates
            centroid_latlons = self.ground_truth[landmark_class, :2]

            Logger.log(
                "INFO",
                f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {len(landmark_list)} landmarks detected.",
            )
            Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")

            # Logging details for each detected landmark
            if landmark_arr.size > 0:
                Logger.log(
                    "INFO",
                    f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] class\tcentroid_xy\tcentroid_latlons\tconfidence",
                )
                for i in range(len(landmark_list)):
                    # Class ID, convert to int for cleaner logging
                    cls = int(landmark_arr[i, 2])
                    x, y = int(landmark_arr[i, 0]), int(
                        landmark_arr[i, 1]
                    )  # Centroid coordinates, convert to int for cleaner logging
                    lat, lon = centroid_latlons[i]
                    conf = confidence_scores[i]
                    Logger.log(
                        "INFO",
                        f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] {cls}\t({x}, {y})\t({lat:.2f}, {lon:.2f})\t{conf:.2f}",
                    )

            return centroid_xy, centroid_latlons, landmark_class, confidence_scores

        except Exception as e:
            Logger.log("ERROR", f"Detection failed: {str(e)}")
            raise
