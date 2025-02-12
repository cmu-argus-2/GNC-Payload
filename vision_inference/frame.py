import hashlib
from datetime import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    frame: np.ndarray
    camera_id: int
    timestamp: datetime
    frame_id: str = field(init=False)

    def __post_init__(self):
        self.frame_id = Frame.generate_frame_id(self.timestamp)

    @staticmethod
    def generate_frame_id(timestamp: datetime) -> str:
        """
        Generates a unique frame ID using the hash of the timestamp.

        Args:
            timestamp: The timestamp associated with the frame.

        Returns:
            A hexadecimal string representing the hash of the timestamp.
        """
        # Convert the timestamp to string and encode it to bytes, then hash it
        timestamp_str = str(timestamp)
        hash_object = hashlib.sha1(timestamp_str.encode())  # Using SHA-1
        frame_id = hash_object.hexdigest()
        return frame_id[:16]  # Optionally still shorten if needed

    @staticmethod
    def resize(img, width=640, height=480):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
