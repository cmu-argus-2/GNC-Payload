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

# pylint: disable=import-error, too-many-locals, unsubscriptable-object
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from more_itertools import batched
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from sensors.camera_model import CameraModel
from utils.config_utils import USER_CONFIG_PATH, load_config
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

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        :return: A generator that yields Tuples containing the pixel_coordinates,
        latlon, class_id, and confidence for each landmark.
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
        Logger().log("INFO", f"Initializing LandmarkDetector for region {region_id}.")
        models_dir = load_config(USER_CONFIG_PATH)["models_directory"]

        self.region_id = region_id
        try:
            self.model = YOLO(
                os.path.join(models_dir, self.get_ld_model_weights_relative_path(region_id))
            )
            self.ground_truth = LandmarkDetector.load_ground_truth(
                os.path.join(models_dir, self.get_region_bounding_boxes_relative_path(region_id))
            )
        except Exception as e:
            Logger().log("ERROR", f"Failed to load necessary data: {e}")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        Logger.log("INFO", f"Model device: {self.model.device}")

    @staticmethod
    def get_ld_model_weights_relative_path(region_id: str) -> str:
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
            Logger().log("ERROR", f"Configuration error: {e}")
            raise

    def _process_results(self, frame: Frame, result: Results) -> LandmarkDetections:
        """
        Process the results from YOLO and generate a LandmarkDetections object.

        Args:
            frame: The Frame that the results are associated with. Only used for logging.
            result: The results to process.

        Returns:
            The resulting LandmarkDetections object.
        """
        landmarks = result.boxes
        if len(landmarks) == 0:
            Logger.log(
                "INFO",
                f"{frame.debug_str} No landmarks detected in Region {self.region_id}.",
            )
            return LandmarkDetections.empty()

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
                Logger.log(
                    "INFO",
                    f"{frame.debug_str} No valid landmarks detected in Region {self.region_id}.",
                )
                return LandmarkDetections.empty()
            xywh = xywh[valid_indices]
            class_ids = class_ids[valid_indices]
            confidences = confidences[valid_indices]

        landmark_detections = LandmarkDetections(
            pixel_coordinates=xywh[:, :2],
            latlons=self.ground_truth[class_ids, :2],
            class_ids=class_ids,
            region_ids=np.array([self.region_id] * len(class_ids)),
            confidences=confidences,
        )
        # TODO: Check if we can only activate the asserts in debug mode
        landmark_detections.assert_invariants()
        return landmark_detections

    def detect_landmarks(
        self, frames: Sequence[Frame], batch_size: int = 8
    ) -> List[LandmarkDetections]:
        """
        Detects landmarks in a set of input images using a pretrained YOLO model and extracts relevant information.

        The detection process filters out landmarks with low confidence scores (below 0.5)
        and invalid bounding box dimensions.
        It aims to provide a comprehensive set of data for each detected landmark,
        facilitating further analysis or processing.

        Args:
            frames: The input Frames on which to perform landmark detection. Does not need to be a multiple of
                    batch_size.
            batch_size: The number of input Frames to process in each batch.

        Returns:
            A LandmarkDetections object containing the detected landmarks and associated data.
        """
        if len(frames) == 0:
            Logger.log("INFO", "No frames provided for landmark detection.")
            return []

        Logger.log(
            "INFO",
            f"{', '.join([frame.debug_str for frame in frames])} Starting the landmark detection process.",
        )

        try:
            landmark_detections = []
            for batch in batched(frames, batch_size):  # pylint: disable=E1133
                start_time = perf_counter()
                results_sequence: Sequence[Results] = self.model.predict(
                    # TODO: can we directly pass numpy arrays instead
                    [Image.fromarray(frame.image) for frame in batch],
                    conf=LandmarkDetector.CONFIDENCE_THRESHOLD,
                    imgsz=LandmarkDetector.IMAGE_SIZE,
                    verbose=False,
                )
                inference_time = perf_counter() - start_time
                if len(results_sequence) != len(batch):
                    Logger.log(
                        "ERROR",
                        f"Mismatch: YOLO returned {len(results_sequence)} results for a batch of {len(batch)}",
                    )
                    continue

                landmark_detections.extend(
                    [
                        self._process_results(frame, results)
                        for frame, results in zip(batch, results_sequence)
                    ]
                )
                Logger.log(
                    "INFO",
                    f"Inference completed for batch of {len(batch)} in {inference_time:.2f} seconds.",
                )

            total_detections = sum(len(det) for det in landmark_detections)
            Logger.log(
                "INFO",
                f"{total_detections} landmarks detected in total.",
            )

            # Logging details for each detected landmark
            # for frame, detections in zip(frames, landmark_detections):
            #     Logger.log(
            #         "INFO",
            #         f"{frame.debug_str} class_id\tpixel_coordinates\tlatlon\tconfidence",
            #     )
            # for (x, y), (lat, lon), class_id, region_id, confidence in detections:
            #     Logger.log(
            #         "INFO",
            #         f"{frame.debug_str} "
            #         f"{class_id}\t({x:.0f}, {y:.0f})\t({lat:.2f}, {lon:.2f})\t{confidence:.2f}",
            #     )

            return landmark_detections

        except Exception as e:
            Logger().log("ERROR", f"Detection process failed: {e}")
            raise

    def png_detect_landmarks(
        self,
        png_paths: List[str],
        batch_size: int = 1,
    ) -> Dict[str, LandmarkDetections]:
        """
        Perform GPU-accelerated landmark detection on multiple PNG image files.

        Args:
            png_paths: List of paths to PNG files for landmark detection

        Returns:
            Dictionary mapping file paths to their corresponding LandmarkDetections
        """
        Logger.log(
            "INFO",
            f"Initialized LandmarkDetector for GPU batch processing in region {self.region_id}",
        )

        results = {}

        Logger.log("INFO", f"Processing {len(png_paths)} PNG files in batches...")
        Logger.log("INFO", f"Image paths: {', '.join(png_paths)}")

        # Load and prepare images
        images = []
        valid_paths = []
        frames = []

        for png_path in png_paths:
            try:
                # Use cv2 to read PNG files
                image: MatLike = cv2.imread(png_path)

                if image is None:
                    raise ValueError(f"Failed to load image: {png_path}")
                # pylint: disable=E1101
                assert image.ndim == 3 and image.shape[2] == 3, "Image must be RGB"
                # Convert from BGR to RGB for consistency with the rest of the pipeline
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                valid_paths.append(os.path.basename(png_path))

                fname, _ = os.path.splitext(os.path.basename(png_path))
                parts = fname.split("_")
                if len(parts) < 3 or parts[0] != "img":
                    Logger.log("ERROR", f"Unexpected filename format: {fname}")
                    raise ValueError(f"Unexpected filename format: {fname}")
                camera_name = parts[-1]
                frames.append(Frame(image=image, camera_name=camera_name, timestamp=datetime.now()))

            except Exception as e:  # pylint: disable=W0718
                Logger.log("ERROR", f"Error loading PNG file {png_path}: {e}")
                results[os.path.basename(png_path)] = LandmarkDetections.empty()

        if len(images) == 0:
            Logger.log("ERROR", "No valid PNG files were loaded for landmark detection.")
            raise ValueError("None of the PNG files could be loaded!")

        landmark_detections = self.detect_landmarks(frames, batch_size=batch_size)

        results.update(dict(zip(valid_paths, landmark_detections)))

        # Log summary
        success_count = sum(1 for detections in results.values() if len(detections) > 0)
        total_landmarks = sum(len(detections) for detections in results.values())

        Logger.log(
            "INFO", f"Batch processing complete: {len(results)}/{len(png_paths)} files processed"
        )
        Logger.log(
            "INFO",
            f"Found landmarks in {success_count} images, {total_landmarks} landmarks detected total",
        )

        return results
