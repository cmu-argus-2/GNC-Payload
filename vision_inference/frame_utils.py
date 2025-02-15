from datetime import datetime
import os
from itertools import cycle
from typing import Generator

import cv2

from vision_inference.frame import Frame


def demo_frame_cycle_generator(image_dir: str) -> Generator[Frame, None, None]:
    """
    A generator that endlessly cycles through the images in the specified directory and yields Frame objects.

    :param image_dir: The directory containing the images to cycle through.
    :yield: A Frame object for each image in the directory.
    """
    image_files = [
        os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".png")
    ]
    image_files.sort()  # Ensure the files are in a consistent order

    for image_path in cycle(image_files):
        image = cv2.imread(image_path)
        if image is not None:
            yield Frame(frame=image, camera_id=0, timestamp=datetime.now())
        else:
            yield None


def main():
    demo_frames = demo_frame_cycle_generator("data/inference_input")
    print(next(demo_frames))


if __name__ == "__main__":
    main()
