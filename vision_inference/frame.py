import hashlib

import cv2

from vision_inference.logger import Logger


class Frame:
    def __init__(self, frame, camera_id, timestamp):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp
        # Generate ID by hashing the timestamp
        self.frame_id = self.generate_frame_id(timestamp)
        self.landmarks = []

    def generate_frame_id(self, timestamp):
        """
        Generates a unique frame ID using the hash of the timestamp.

        Args:
            timestamp (datetime): The timestamp associated with the frame.

        Returns:
            str: A hexadecimal string representing the hash of the timestamp.
        """
        # Convert the timestamp to string and encode it to bytes, then hash it
        timestamp_str = str(timestamp)
        hash_object = hashlib.sha1(timestamp_str.encode())  # Using SHA-1
        frame_id = hash_object.hexdigest()
        return frame_id[:16]  # Optionally still shorten if needed

    def update_landmarks(self, new_landmarks):
        """Update the frame with new landmark data."""
        self.landmarks = new_landmarks
        Logger.log("INFO", f"[Camera {self.camera_id} frame {self.frame_id}] Landmarks updated on Frame object.")

    def save(self):
        pass

    @classmethod
    def resize(cls, img, width=640, height=480):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
