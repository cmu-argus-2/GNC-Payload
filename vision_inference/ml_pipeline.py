"""
Machine Learning Pipeline for Region Classification and Landmark Detection

This script defines a machine learning pipeline that processes a series of frames from camera feeds,
performs region classification to identify geographic regions within the frames, and subsequently
detects landmarks within those regions. The script is designed to handle varying lighting conditions
by discarding frames that are deemed too dark for reliable classification or detection.

Author: Eddie
Date: January 27, 2025
"""

import os
from typing import Tuple, List

import cv2
import numpy as np

from vision_inference.ld import LandmarkDetections, LandmarkDetector
from vision_inference.logger import Logger
from vision_inference.rc import RegionClassifier
from vision_inference.frame import Frame


class MLPipeline:
    """
    A class representing a machine learning pipeline for processing camera feed frames for
    region classification and landmark detection.

    Attributes:
        region_classifier (RegionClassifier): An instance of RegionClassifier for classifying geographic regions in frames.
    """

    REGION_TO_LOCATION = {
        "10S": "California",
        "10T": "Washington / Oregon",
        "11R": "Baja California, Mexico",
        "12R": "Sonora, Mexico",
        "16T": "Minnesota / Wisconsin / Iowa / Illinois",
        "17R": "Florida",
        "17T": "Toronto, Canada / Michigan / OH / PA",
        "18S": "New Jersey / Washington DC",
        "32S": "Tunisia (North Africa near Tyrrhenian Sea)",
        "32T": "Switzerland / Italy / Tyrrhenian Sea",
        "33S": "Sicilia, Italy",
        "33T": "Italy / Adriatic Sea",
        "52S": "Korea / Kumamoto, Japan",
        "53S": "Hiroshima to Nagoya, Japan",
        "54S": "Tokyo to Hachinohe, Japan",
        "54T": "Sapporo, Japan",
    }

    def __init__(self):
        """
        Initializes the MLPipeline class, setting up any necessary components for the machine learning tasks.
        """
        self.region_classifier = RegionClassifier()

    def classify_frame(self, frame: Frame) -> List[str]:
        """
        Classifies a frame to identify geographic regions using the region classifier.

        Args:
            frame: The Frame object to classify.

        Returns:
            A list of predicted region IDs classified from the frame.
        """
        return self.region_classifier.classify_region(frame)

    def run_ml_pipeline_on_single(
        self, frame_obj: Frame
    ) -> Tuple[LandmarkDetections, dict[str, slice]]:
        """
        Processes a single frame, classifying it for geographic regions and detecting landmarks,
        and returns the detection results.

        Args:
            frame_obj (Frame): The Frame object to process.

        Returns:
            A Tuple containing:
            - The LandmarkDetections object containing the landmark detection results.
            - A dictionary mapping region IDs to slices of the landmark detections.
        """
        Logger.log(
            "INFO",
            "------------------------------Inference---------------------------------",
        )
        pred_regions = self.classify_frame(frame_obj)
        if len(pred_regions) == 0:
            Logger.log(
                "INFO",
                f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] No salient regions detected. ",
            )
            return LandmarkDetections.empty(), {}

        region_slices = {}
        landmark_detections = []
        total_detections = 0
        for region_id in pred_regions:
            detector = LandmarkDetector(region_id)
            detections = detector.detect_landmarks(frame_obj)

            landmark_detections.append(detections)
            region_slices[region_id] = slice(total_detections, total_detections + len(detections))
            total_detections += len(detections)
        landmark_detections = LandmarkDetections.stack(landmark_detections)

        return landmark_detections, region_slices

    @staticmethod
    def adjust_color(color: Tuple[int, int, int], confidence):
        # Option 1: Exponential scaling
        # scale_factor = (confidence ** 2)  # Square the confidence to exaggerate differences

        # Option 2: Offset and scaling adjustment
        # This ensures that even low confidence values have a noticeable color intensity
        # min_factor = 0.5  # Ensure that even the lowest confidence gives us at least half the color intensity
        # scale_factor = min_factor + (1 - min_factor) * confidence

        # Option 3: Squared scaling (chosen for demonstration)
        # More dramatic effect as confidence increases
        scale_factor = confidence**2

        # Apply the scale factor to the color components
        adjusted_color = tuple(int(c * scale_factor) for c in color)

        return adjusted_color

    @staticmethod
    def get_region_id(
        index: int, region_slices: dict[str, slice], sequence_length: int
    ) -> str | None:
        """
        Returns the region ID for a given index within a sequence of landmark detections.

        Args:
            index: The index to find the region ID for.
            region_slices: The dictionary mapping region IDs to slices of the landmark detections.
            sequence_length: The total length of the sequence of landmark detections.

        Returns:
            The region ID for the given index, or None if the index is not within any region slice.
        """
        for region_id, slice_ in region_slices.items():
            if index in range(*slice_.indices(sequence_length)):
                return region_id
        return None

    # TODO: Improve the readability of this method.
    @staticmethod
    def visualize_landmarks(
        frame_obj: Frame,
        landmark_detections: LandmarkDetections,
        region_slices: dict[str, slice],
        save_dir: str,
    ) -> None:
        """
        Draws larger centroids of landmarks on the frame, adds a larger legend for region colors with semi-transparent boxes,
        and saves the image. Also displays camera metadata on the image.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # TODO: less hacky fix for RGB to BGR conversion
        frame_obj.frame = cv2.cvtColor(frame_obj.frame, cv2.COLOR_RGB2BGR)

        image = frame_obj.frame.copy()

        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (0, 165, 255),  # Orange
            (180, 105, 255),  # Pink
            (255, 0, 0),  # Blue
        ]

        # ============================== landmark display ==================================
        region_color_map = {}
        circle_radius = 15
        circle_thickness = -1

        for idx, (region_id, slice_) in enumerate(region_slices.items()):
            landmark_detections = landmark_detections[slice_]

            base_color = colors[idx % len(colors)]
            region_color_map[region_id] = base_color

            for (x, y), _, class_id, confidence in landmark_detections:
                adjusted_color = MLPipeline.adjust_color(base_color, confidence)
                cv2.circle(image, (int(x), int(y)), circle_radius, adjusted_color, circle_thickness)

        # Sort landmarks by confidence, descending, and keep the top few
        LANDMARK_DISPLAY_COUNT = 5
        top_landmark_indices = np.argsort(landmark_detections.confidences)[-LANDMARK_DISPLAY_COUNT:]
        top_landmark_regions = [
            MLPipeline.get_region_id(index, region_slices, len(landmark_detections))
            for index in top_landmark_indices
        ]
        top_landmarks = landmark_detections[top_landmark_indices]

        # ========================== Metadata displaying ========================================
        # Metadata drawing first to determine right edge for alignment
        metadata_info = f"Camera ID: {frame_obj.camera_id} | Time: {frame_obj.timestamp} | Frame: {frame_obj.frame_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        metadata_font_scale = 1
        text_thickness = 2
        metadata_text_size = cv2.getTextSize(
            metadata_info, font, metadata_font_scale, text_thickness
        )[0]
        metadata_text_x = image.shape[1] - metadata_text_size[0] - 10  # Right align
        metadata_text_y = 30
        metadata_box_height = metadata_text_size[1] + 20  # Some padding

        # Draw semi-transparent rectangle for metadata
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (metadata_text_x, metadata_text_y - metadata_text_size[1] - 10),
            (metadata_text_x + metadata_text_size[0] + 10, metadata_text_y + 10),
            (50, 50, 50),
            -1,
        )
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        # Place metadata text
        cv2.putText(
            image,
            metadata_info,
            (metadata_text_x, metadata_text_y),
            font,
            metadata_font_scale,
            (255, 255, 255),
            text_thickness,
        )

        # Prepare for Top Landmarks box
        top_legend_x = metadata_text_x  # Align with the left edge of metadata text
        top_legend_y = metadata_text_y + metadata_box_height + 20  # Spacing below the metadata box

        # Top landmarks settings
        top_font_scale = 1  # Three times bigger
        entry_height = int(
            cv2.getTextSize("Test", font, top_font_scale, 1)[0][1] * 1.5
        )  # Adjusted entry height
        max_width = 0
        total_height = 0

        text_entries = []
        for i, (region_id, ((x, y), (lat, lon), _, conf)) in enumerate(
            zip(top_landmark_regions, top_landmarks)
        ):
            text = f"Top {i + 1}: Region {region_id}, Conf: {conf:.2f}, XY: ({x:.0f}, {y:.0f}), LatLon: ({lat:.2f}, {lon:.2f})"
            text_size = cv2.getTextSize(text, font, top_font_scale, 1)[0]
            max_width = max(max_width, text_size[0] + 20)  # Update max width
            total_height += entry_height
            text_entries.append((text, top_legend_y + total_height))

        # Draw semi-transparent rectangle for top landmarks
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (top_legend_x, top_legend_y),
            (top_legend_x + max_width, top_legend_y + total_height + 10),
            (50, 50, 50),
            -1,
        )
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        # Place each text entry
        for text, y_position in text_entries:
            cv2.putText(
                image,
                text,
                (top_legend_x + 10, y_position),
                font,
                top_font_scale,
                (255, 255, 255),
                2,
            )
        # ==================== Region Legend =======================
        legend_x = 10
        legend_y = 30
        font_scale_legend = 1.5
        text_thickness_legend = 3
        for region, color in region_color_map.items():
            location = MLPipeline.REGION_TO_LOCATION.get(
                region, "Unknown Location"
            )  # Get the location name or default to 'Unknown Location'
            text = f"Region {region}: {location}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale_legend, text_thickness_legend
            )
            overlay = image.copy()
            # Draw a semi-transparent rectangle
            cv2.rectangle(
                overlay,
                (legend_x, legend_y),
                (legend_x + text_width, legend_y + text_height + 10),
                color,
                -1,
            )
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            # Put the text on the image
            cv2.putText(
                image,
                text,
                (legend_x, legend_y + text_height),
                font,
                font_scale_legend,
                (255, 255, 255),
                text_thickness_legend,
            )
            # Move down for the next entry
            legend_y += text_height + 10

        landmark_save_path = os.path.join(save_dir, f"frame_w_landmarks_{frame_obj.camera_id}.png")
        cv2.imwrite(landmark_save_path, image)

        img_save_path = os.path.join(save_dir, "frame.png")
        cv2.imwrite(img_save_path, frame_obj.frame)

        metadata_path = os.path.join(save_dir, "frame_metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Camera ID: {frame_obj.camera_id}\n")
            f.write(f"Timestamp: {frame_obj.timestamp}\n")
            f.write(f"Frame ID: {frame_obj.frame_id}\n")

        Logger.log(
            "INFO",
            f"[Camera {frame_obj.camera_id} frame {frame_obj.frame_id}] Landmark visualization saved to data/inference_output",
        )
