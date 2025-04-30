"""
Landmark Detection Module

This module defines the LandmarkDetector class, which utilizes a pretrained YOLO (You Only Look Once) model
to detect and process landmarks within images. The detector extracts landmarks as bounding boxes along with
their associated class IDs and confidence scores. The main functionality revolves around the detection of
landmarks in given images and the extraction of useful information such as centroids and class/confidence data.

Dependencies:
- numpy: Used for array manipulations and handling numerical operations.
- cv2 (OpenCV): Required for image processing tasks.
- ultralytics YOLO: The YOLO model implementation from Ultralytics,
                    used for object detection tasks. (Large package warning)

Author: Eddie, Haochen
Date: [Creation or Last Update Date]
"""

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils.config_utils import USER_CONFIG_PATH, load_config
from sensors.camera_model import CameraModel
from vision_inference.frame import Frame
from vision_inference.logger import Logger


@dataclass
class LandmarkDetections:
    """
    A class to store info about landmark detections.

    Attributes:
        pixel_coordinates: A numpy array of shape (N, 2) containing the x and y pixel coordinates
                           for each detected landmark's centroid.
        latlons: A numpy array of shape (N, 2) containing the latitudes and longitudes
                 for each detected landmark's centroid.
        class_ids: A numpy array of shape (N,) containing the class IDs for each detected landmark.
        confidences: A numpy array of shape (N,) containing the confidence scores for each detected landmark.
    """

    pixel_coordinates: np.ndarray
    latlons: np.ndarray
    class_ids: np.ndarray
    region_ids: np.ndarray
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
            region_ids=self.region_ids[index],
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
                self.region_ids[i],
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
            region_ids=np.array([], dtype="U32"),
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
        assert len(self.region_ids.shape) == 1, "region_ids should be a 1D array."
        assert len(self.confidences.shape) == 1, "confidences should be a 1D array."

        assert (
            self.pixel_coordinates.shape[0]
            == self.latlons.shape[0]
            == len(self.class_ids)
            == len(self.confidences)
        ), "All arrays should have the same length."

        assert (
            np.all(self.pixel_coordinates >= 0)
            and np.all(self.pixel_coordinates[:, 0] <= CameraModel.IMAGE_WIDTH - 1)
            and np.all(self.pixel_coordinates[:, 1] <= CameraModel.IMAGE_HEIGHT - 1)
        ), "pixel_coordinates should be within image bounds."

    @staticmethod
    def stack(detections: List["LandmarkDetections"]) -> "LandmarkDetections":
        """
        Stack multiple LandmarkDetections into a single LandmarkDetections object.

        Args:
            detections: A list of LandmarkDetections objects.

        Returns:
            A LandmarkDetections object containing the stacked data.
        """
        if len(detections) == 0:
            return LandmarkDetections.empty()

        return LandmarkDetections(
            pixel_coordinates=np.row_stack([det.pixel_coordinates for det in detections]),
            latlons=np.row_stack([det.latlons for det in detections]),
            class_ids=np.concatenate([det.class_ids for det in detections]),
            region_ids=np.concatenate([det.region_ids for det in detections]),
            confidences=np.concatenate([det.confidences for det in detections]),
        )


class LandmarkDetector:
    """
    A class to detect landmarks in images using a pretrained YOLO model for a specific MGRS region.
    """

    CONFIDENCE_THRESHOLD = 0.5
    IMAGE_SIZE = CameraModel.RESOLUTION

    def __init__(self, region_id: str):
        """
        Initialize the LandmarkDetector with a specific region ID
        The YOLO object is created with the path to a specific pretrained model
        """
        Logger.log("INFO", f"Initializing LandmarkDetector for region {region_id}.")
        models_dir = load_config(USER_CONFIG_PATH)["models_directory"]

        self.region_id = region_id
        try:
            self.model = YOLO(
                os.path.join(models_dir, self.get_LD_model_weights_relative_path(region_id))
            )
            self.ground_truth = LandmarkDetector.load_ground_truth(
                os.path.join(models_dir, self.get_region_bounding_boxes_relative_path(region_id))
            )
        except Exception as e:
            Logger.log("ERROR", f"Failed to load necessary data: {e}")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        Logger.log("INFO", f"Model device: {self.model.device}")

    @staticmethod
    def get_LD_model_weights_relative_path(region_id: str) -> str:
        """
        Get the relative path to the model weights file for a specific MGRS region.

        Args:
            region_id: The MGRS region ID to get the LD model weights relative path for.

        Returns:
            The relative path to the LD model weights file.
        """
        return os.path.join(region_id, "yolo_model_weights.pt")

    @staticmethod
    def get_region_bounding_boxes_relative_path(region_id: str) -> str:
        """
        Get the relative path to the bounding box lat/lon coordinates file for a specific MGRS region.

        Args:
            region_id: The MGRS region ID to get the bounding box coordinates relative path for.

        Returns:
            The relative path to the bounding box coordinates file.
        """
        return os.path.join(region_id, "bounding_boxes.csv")

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
            return np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)
        except Exception as e:
            Logger.log("ERROR", f"Configuration error: {e}")
            raise

    def detect_landmarks(self, frame: Frame) -> LandmarkDetections:
        """
        Detects landmarks in an input image using a pretrained YOLO model and extracts relevant information.

        The detection process filters out landmarks with low confidence scores (below 0.5)
        and invalid bounding box dimensions.
        It aims to provide a comprehensive set of data for each detected landmark,
        facilitating further analysis or processing.

        Args:
            frame: The input Frame on which to perform landmark detection.

        Returns:
            A LandmarkDetections object containing the detected landmarks and associated data.
        """
        Logger.log(
            "INFO",
            f"[Camera {frame.camera_name} frame {frame.frame_id}] Starting the landmark detection process.",
        )

        try:
            # Detect landmarks using the YOLO model
            start_time = perf_counter()
            results: Results = self.model.predict(
                Image.fromarray(frame.image),
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

                xywh = landmarks.xywh.cpu().numpy()
                class_ids = landmarks.cls.cpu().numpy().astype(int)
                confidences = landmarks.conf.cpu().numpy()

                valid_indices = (
                    np.all(xywh >= 0, axis=1)
                    & (xywh[:, 0] <= CameraModel.IMAGE_WIDTH - 1)
                    & (xywh[:, 1] <= CameraModel.IMAGE_HEIGHT - 1)
                )
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
                        region_ids=np.array([self.region_id] * len(class_ids)),
                        confidences=confidences,
                    )
                )

            landmark_detections = LandmarkDetections.stack(landmark_detections)
            landmark_detections.assert_invariants()

            if len(landmark_detections) == 0:
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_name} frame {frame.frame_id}] No landmarks detected in Region {self.region_id}.",
                )
                return LandmarkDetections.empty()

            Logger.log(
                "INFO",
                f"[Camera {frame.camera_name} frame {frame.frame_id}] "
                f"{len(landmark_detections)} landmarks detected.",
            )
            Logger.log("INFO", f"Inference completed in {inference_time:.2f} seconds.")

            # Logging details for each detected landmark
            Logger.log(
                "INFO",
                f"[Camera {frame.camera_name} frame {frame.frame_id}] "
                f"class_id\tpixel_coordinates\tlatlon\tconfidence",
            )
            for (x, y), (lat, lon), class_id, confidence in landmark_detections:
                Logger.log(
                    "INFO",
                    f"[Camera {frame.camera_name} frame {frame.frame_id}] "
                    f"{class_id}\t({x:.0f}, {y:.0f})\t({lat:.2f}, {lon:.2f})\t{confidence:.2f}",
                )

            return landmark_detections

        except Exception as e:
            Logger.log("ERROR", f"Detection process failed: {e}")
            raise

    def batch_detect_landmarks(
        self,
        npy_paths: List[str],
        batch_size: int = 8,
    ) -> Dict[str, LandmarkDetections]:
        """
        Perform GPU-accelerated landmark detection on multiple NumPy (.npy) files using batching.

        Args:
            npy_paths: List of paths to numpy files containing images for landmark detection
            batch_size: Number of images to process in each GPU batch

        Returns:
            Dictionary mapping file paths to their corresponding LandmarkDetections
        """
        Logger.log(
            "INFO",
            f"Initialized LandmarkDetector for GPU batch processing in region {self.region_id}",
        )

        # Process in batches to utilize GPU effectively
        results = {}
        total_batches = (len(npy_paths) + batch_size - 1) // batch_size

        Logger.log("INFO", f"Processing {len(npy_paths)} NumPy files in {total_batches} batches...")
        batch_iterator = tqdm(range(0, len(npy_paths), batch_size), total=total_batches)

        for batch_start in batch_iterator:
            # Get batch paths
            batch_end = min(batch_start + batch_size, len(npy_paths))
            batch_paths = npy_paths[batch_start:batch_end]

            # Load and prepare batch images
            batch_images = []
            valid_paths = []

            for npy_path in batch_paths:
                try:
                    # Load NumPy array
                    array = np.load(npy_path)
                    batch_images.append(array)
                    valid_paths.append(os.path.basename(npy_path))

                except Exception as e:
                    Logger.log("ERROR", f"Error loading NumPy file {npy_path}: {e}")
                    results[os.path.basename(npy_path)] = LandmarkDetections.empty()

            # No valid images in this batch
            if not batch_images:
                continue

            # Process batch and get dictionary results directly
            batch_results = self._process_image_batch_direct(batch_images, valid_paths)

            # Update the results dictionary with batch results
            results.update(batch_results)

        # Log summary
        success_count = sum(1 for detections in results.values() if len(detections) > 0)
        total_landmarks = sum(len(detections) for detections in results.values())

        Logger.log(
            "INFO", f"Batch processing complete: {len(results)}/{len(npy_paths)} files processed"
        )
        Logger.log(
            "INFO",
            f"Found landmarks in {success_count} images, {total_landmarks} landmarks detected total",
        )

        return results

    def _process_image_batch_direct(
        self, images: List[np.ndarray], image_names: List[str]
    ) -> Dict[str, LandmarkDetections]:
        """
        Process a batch of image arrays through the YOLO model at once to leverage GPU parallelism.

        Args:
            images: List of numpy arrays representing images
            image_names: List of corresponding image names (not file paths)

        Returns:
            Dictionary mapping image names to their corresponding LandmarkDetections
        """
        if not images:
            return {}

        try:
            # Convert images to PIL format for YOLO
            pil_images = [Image.fromarray(img) for img in images]
            start_time = perf_counter()
            batch_results = self.model.predict(
                pil_images,
                conf=LandmarkDetector.CONFIDENCE_THRESHOLD,
                imgsz=LandmarkDetector.IMAGE_SIZE,
                verbose=False,
                batch=len(pil_images),  # Explicitly set batch size
            )
            inference_time = perf_counter() - start_time
            avg_time_per_image = inference_time / len(pil_images)
            Logger.log(
                "INFO",
                f"Batch inference completed in {inference_time:.3f}s "
                f"(avg: {avg_time_per_image:.3f}s/image) on {self.model.device}",
            )

            # Process results for each image
            detections_dict = {}
            for i, (name, result) in enumerate(zip(image_names, batch_results)):
                landmarks = result.boxes
                if len(landmarks) == 0:
                    Logger.log(
                        "INFO", f"[Image: {name}] No landmarks detected in region {self.region_id}."
                    )
                    continue

                # Extract detection data
                xywh = landmarks.xywh.cpu().numpy()
                class_ids = landmarks.cls.cpu().numpy().astype(int)
                confidences = landmarks.conf.cpu().numpy()

                # Filter valid detections
                valid_indices = np.all(xywh[:, 2:] >= 0, axis=1)
                if not np.all(valid_indices):
                    Logger.log(
                        "INFO",
                        f"[Image: {name}] Skipping landmark(s) with invalid bounding box dimensions.",
                    )
                    if not np.any(valid_indices):
                        continue

                    xywh = xywh[valid_indices]
                    class_ids = class_ids[valid_indices]
                    confidences = confidences[valid_indices]

                # Create LandmarkDetections object
                detections = LandmarkDetections(
                    pixel_coordinates=xywh[:, :2],
                    latlons=self.ground_truth[class_ids, :2],
                    class_ids=class_ids,
                    region_ids=np.array([self.region_id] * len(class_ids)),
                    confidences=confidences,
                )

                Logger.log("INFO", f"[Image: {name}] {len(detections)} landmarks detected.")
                detections_dict[name] = detections
            return detections_dict

        except Exception as e:
            Logger.log("ERROR", f"Batch detection failed: {e}")
            # Return empty detections for all images in case of failure
            return {name: LandmarkDetections.empty() for name in image_names}
