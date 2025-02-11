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

import os
from time import perf_counter
from typing import List
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from vision_inference.logger import Logger
from vision_inference.frame import Frame


@dataclass
class LandmarkDetections:
    """
    A class to store info about landmark detections.

    Attributes:
        centroid_xys: A numpy array of shape (N, 2) containing the x and y image coordinates for each detected landmark's centroid.
        centroid_latlons: A numpy array of shape (N, 2) containing the latitudes and longitudes for each detected landmark's centroid.
        landmark_classes: A numpy array of shape (N,) containing the class IDs for each detected landmark.
        confidence_scores: A numpy array of shape (N,) containing the confidence scores for each detected landmark.
    """

    centroid_xys: np.ndarray
    centroid_latlons: np.ndarray
    landmark_classes: np.ndarray
    confidence_scores: np.ndarray

    @property
    def detection_count(self) -> int:
        """
        :return: The number of landmark detections.
        """
        return len(self.landmark_classes)

    @staticmethod
    def empty() -> "LandmarkDetections":
        """
        Creates an empty LandmarkDetections object.

        Returns:
            A LandmarkDetections object with empty arrays of the correct shape for all attributes.
        """
        return LandmarkDetections(
            centroid_xys=np.zeros((0, 2)),
            centroid_latlons=np.zeros((0, 2)),
            landmark_classes=np.zeros(0, dtype=int),
            confidence_scores=np.zeros(0),
        )

    def assert_invariants(self) -> None:
        """
        Validates the invariants of the landmark detections.

        :raises AssertionError: If any of the invariants are violated.
        """
        assert len(self.centroid_xys.shape) == 2, "centroid_xy should be a 2D array."
        assert self.centroid_xys.shape[1] == 2, "centroid_xy should have 2 columns."
        assert len(self.centroid_latlons.shape) == 2, "centroid_latlons should be a 2D array."
        assert self.centroid_latlons.shape[1] == 2, "centroid_latlons should have 2 columns."
        assert len(self.landmark_classes.shape) == 1, "landmark_classes should be a 1D array."
        assert len(self.confidence_scores.shape) == 1, "confidence_scores should be a 1D array."

        assert (
            self.centroid_xys.shape[0]
            == self.centroid_latlons.shape[0]
            == len(self.landmark_classes)
            == len(self.confidence_scores)
        ), "All arrays should have the same length."

    @staticmethod
    def stack(detections: List["LandmarkDetections"]) -> "LandmarkDetections":
        """
        Stack multiple LandmarkDetections into a single LandmarkDetections object.

        Args:
            detections: A list of LandmarkDetections objects.

        Returns:
            A LandmarkDetections object containing the stacked data.
        """
        return LandmarkDetections(
            centroid_xys=np.row_stack([det.centroid_xys for det in detections]),
            centroid_latlons=np.row_stack([det.centroid_latlons for det in detections]),
            landmark_classes=np.concatenate([det.landmark_classes for det in detections]),
            confidence_scores=np.concatenate([det.confidence_scores for det in detections]),
        )


class LandmarkDetector:
    CONFIDENCE_THRESHOLD = 0.5
    # TODO: Can we increase this to the full resolution (2592, 4608) on the Jetson?
    IMAGE_SIZE = (1088, 1920)
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
            Logger.log("ERROR", f"Failed to load necessary data: {e}")
            raise

    @staticmethod
    def load_ground_truth(ground_truth_path: str) -> np.ndarray:
        """
        Loads ground truth bounding box coordinates from a CSV file.

        Args:
            ground_truth_path (str): Path to the ground truth CSV file.

        Returns:
            A numpy array of shape (N, 6) containing the following for each landmark:
            (centroid_lat, centroid_lon, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon).
        """
        try:
            # TODO: change csvs to have lat, lon instead of lon, lat for consistency
            return np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)[:, [1, 0, 3, 2, 5, 4]]
        except Exception as e:
            Logger.log("ERROR", f"Configuration error: {e}")
            raise

    def detect_landmarks(self, frame: Frame) -> LandmarkDetections:
        """
        Detects landmarks in an input image using a pretrained YOLO model and extracts relevant information.

        The detection process filters out landmarks with low confidence scores (below 0.5) and invalid bounding box dimensions.
        It aims to provide a comprehensive set of data for each detected landmark, facilitating further analysis or processing.

        Args:
            frame: The input Frame on which to perform landmark detection.

        Returns:
            A LandmarkDetections object containing the detected landmarks and associated data.
        """
        Logger.log(
            "INFO",
            f"[Camera {frame.camera_id} frame {frame.frame_id}] Starting the landmark detection process.",
        )

        try:
            # Detect landmarks using the YOLO model
            img = Image.fromarray(cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB))
            start_time = perf_counter()
            results: Results = self.model.predict(
                img,
                conf=LandmarkDetector.CONFIDENCE_THRESHOLD,
                imgsz=LandmarkDetector.IMAGE_SIZE,
                verbose=False,
            )
            inference_time = perf_counter() - start_time

            landmark_detections = []

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

                        # TODO: can we process all landmarks at once instead of one by one?
                        landmark_detections.append(
                            LandmarkDetections(
                                centroid_xys=np.array([[x, y]]),
                                centroid_latlons=self.ground_truth[cls, :2],
                                landmark_classes=np.array([cls], dtype=int),
                                confidence_scores=np.array([conf]),
                            )
                        )

            landmark_detections = LandmarkDetections.stack(landmark_detections)

            if landmark_detections.detection_count == 0:
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_id} frame {frame.frame_id}] No landmarks detected in Region {self.region_id}.",
                )
                return LandmarkDetections.empty()

            Logger.log(
                "INFO",
                f"[Camera {frame.camera_id} frame {frame.frame_id}] {landmark_detections.detection_count} landmarks detected.",
            )
            Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")

            # Logging details for each detected landmark
            Logger.log(
                "INFO",
                f"[Camera {frame.camera_id} frame {frame.frame_id}] class\tcentroid_xy\tcentroid_latlon\tconfidence",
            )
            for cls, (x, y), (lat, lon), conf in zip(
                landmark_detections.landmark_classes,
                landmark_detections.centroid_xys,
                landmark_detections.centroid_latlons,
                landmark_detections.confidence_scores,
            ):
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_id} frame {frame.frame_id}] {cls}\t({x:.0f}, {y:.0f})\t({lat:.2f}, {lon:.2f})\t{conf:.2f}",
                )

            return landmark_detections

        except Exception as e:
            Logger.log("ERROR", f"Detection process failed: {e}")
            raise
