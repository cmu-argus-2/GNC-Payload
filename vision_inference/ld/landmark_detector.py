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
from typing import List, Sequence
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
        pixel_coordinates: A numpy array of shape (N, 2) containing the x and y pixel coordinates for each detected landmark's centroid.
        latlons: A numpy array of shape (N, 2) containing the latitudes and longitudes for each detected landmark's centroid.
        class_ids: A numpy array of shape (N,) containing the class IDs for each detected landmark.
        confidences: A numpy array of shape (N,) containing the confidence scores for each detected landmark.
    """

    pixel_coordinates: np.ndarray
    latlons: np.ndarray
    class_ids: np.ndarray
    confidences: np.ndarray

    def __len__(self) -> int:
        """
        :return: The number of landmark detections.
        """
        return len(self.class_ids)

    def __getitem__(self, index: int | slice | Sequence[int] | np.ndarray) -> "LandmarkDetections":
        """
        Get a subset of the landmark detections from this LandmarkDetections object.

        Args:
            index: The index of the landmark detections to retrieve.

        Returns:
            A LandmarkDetections object containing the specified entries.
        """
        return LandmarkDetections(
            pixel_coordinates=self.pixel_coordinates[index, :],
            latlons=self.latlons[index, :],
            class_ids=self.class_ids[index],
            confidences=self.confidences[index],
        )

    def __iter__(self):
        """
        :return: A generator that yields Tuples containing the pixel_coordinates, latlon, class_id, and confidence for each landmark.
        """
        for i in range(len(self)):
            yield (
                self.pixel_coordinates[i, :],
                self.latlons[i, :],
                self.class_ids[i],
                self.confidences[i],
            )

    @staticmethod
    def empty() -> "LandmarkDetections":
        """
        Creates an empty LandmarkDetections object.

        Returns:
            A LandmarkDetections object with empty arrays of the correct shape for all attributes.
        """
        return LandmarkDetections(
            pixel_coordinates=np.zeros((0, 2)),
            latlons=np.zeros((0, 2)),
            class_ids=np.zeros(0, dtype=int),
            confidences=np.zeros(0),
        )

    def assert_invariants(self) -> None:
        """
        Validates the invariants of the landmark detections.

        :raises AssertionError: If any of the invariants are violated.
        """
        assert len(self.pixel_coordinates.shape) == 2, "pixel_coordinates should be a 2D array."
        assert self.pixel_coordinates.shape[1] == 2, "pixel_coordinates should have 2 columns."
        assert len(self.latlons.shape) == 2, "latlons should be a 2D array."
        assert self.latlons.shape[1] == 2, "latlons should have 2 columns."
        assert len(self.class_ids.shape) == 1, "class_ids should be a 1D array."
        assert len(self.confidences.shape) == 1, "confidences should be a 1D array."

        assert (
            self.pixel_coordinates.shape[0]
            == self.latlons.shape[0]
            == len(self.class_ids)
            == len(self.confidences)
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
            pixel_coordinates=np.row_stack([det.pixel_coordinates for det in detections]),
            latlons=np.row_stack([det.latlons for det in detections]),
            class_ids=np.concatenate([det.class_ids for det in detections]),
            confidences=np.concatenate([det.confidences for det in detections]),
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
                if len(landmarks) == 0:
                    continue

                xywh = np.asarray(landmarks.xywh)
                class_ids = np.asarray(landmarks.cls, dtype=int)
                confidences = np.asarray(landmarks.conf)

                valid_indices = np.all(xywh[:, 2:] >= 0, axis=1)
                if not np.all(valid_indices):
                    Logger.log("INFO", "Skipping landmark with invalid bounding box dimensions.")
                    if not np.any(valid_indices):
                        continue
                    xywh = xywh[valid_indices]
                    class_ids = class_ids[valid_indices]
                    confidences = confidences[valid_indices]

                landmark_detections.append(
                    LandmarkDetections(
                        pixel_coordinates=xywh[:, :2],
                        latlons=self.ground_truth[class_ids, :2],
                        class_ids=class_ids,
                        confidences=confidences,
                    )
                )

            landmark_detections = LandmarkDetections.stack(landmark_detections)

            if len(landmark_detections) == 0:
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_id} frame {frame.frame_id}] No landmarks detected in Region {self.region_id}.",
                )
                return LandmarkDetections.empty()

            Logger.log(
                "INFO",
                f"[Camera {frame.camera_id} frame {frame.frame_id}] {len(landmark_detections)} landmarks detected.",
            )
            Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")

            # Logging details for each detected landmark
            Logger.log(
                "INFO",
                f"[Camera {frame.camera_id} frame {frame.frame_id}] class_id\tpixel_coordinates\tlatlon\tconfidence",
            )
            for (x, y), (lat, lon), class_id, confidence in landmark_detections:
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_id} frame {frame.frame_id}] {class_id}\t({x:.0f}, {y:.0f})\t({lat:.2f}, {lon:.2f})\t{confidence:.2f}",
                )

            return landmark_detections

        except Exception as e:
            Logger.log("ERROR", f"Detection process failed: {e}")
            raise
