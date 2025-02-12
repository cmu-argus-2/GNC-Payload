import hashlib

import cv2


class Frame:
    def __init__(self, frame, camera_id, timestamp):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp
        # Generate ID by hashing the timestamp
        self.frame_id = Frame.generate_frame_id(timestamp)

    @staticmethod
    def generate_frame_id(timestamp):
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

    @staticmethod
    def resize(img, width=640, height=480):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
