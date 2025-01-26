from datetime import datetime
import os
from typing import Tuple

import brahe
from brahe.epoch import Epoch
import numpy as np
from scipy.spatial.transform import Rotation

from image_simulation.earth_vis import lat_lon_to_ecef, EarthImageSimulator
from vision_inference.camera import Frame
from vision_inference.ml_pipeline import MLPipeline


class LandmarkBearingSensor:
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark bearing measurements.
    """

    def __init__(self, config):
        """
        :param config: The configuration dictionary.
        """
        camera_Q_body = np.asarray(
            config["satellite"]["camera"]["orientation_in_cubesat_frame"]
        )
        self.body_R_camera = (
            Rotation.from_quat(camera_Q_body, scalar_first=True).inv().as_matrix()
        )
        self.ml_pipeline = MLPipeline()
        self.earth_image_simulator = EarthImageSimulator()

    def take_measurement(
        self, epoch: Epoch, cubesat_position: np.ndarray, eci_R_body: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        ecef_R_eci = brahe.frames.rECItoECEF(epoch)
        position_ecef = ecef_R_eci @ cubesat_position
        ecef_R_body = ecef_R_eci @ eci_R_body

        print(f"Taking measurement at {epoch=}, {cubesat_position=}, {eci_R_body=}")

        # simulate image
        # TODO: Get image from the camera rather than the simulator
        image = self.earth_image_simulator.simulate_image(position_ecef, ecef_R_body)

        if np.all(image == 0):
            print("No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        frame = Frame(image, 0, datetime.now())
        # TODO: queue requests to the model and send them in batches as the sim runs
        regions_and_landmarks = self.ml_pipeline.run_ml_pipeline_on_single(frame)

        # save the image with the detected landmarks
        epoch_str = str(epoch).replace(":", "_").replace(" ", "_").replace(".", "_")
        output_dir = os.path.abspath(
            # TODO: Determine path for actual imagery
            os.path.join(__file__, f"../log/images/seed_69420_epoch_{epoch_str}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        self.ml_pipeline.visualize_landmarks(frame, regions_and_landmarks, output_dir)

        landmark_positions_ecef = np.zeros(shape=(0, 3))
        pixel_coordinates = np.zeros(shape=(0, 2))
        confidence_scores = np.zeros(shape=(0,))

        for _, landmarks in regions_and_landmarks:
            centroids_ecef = lat_lon_to_ecef(
                landmarks.centroid_latlons[np.newaxis, ...]
            ).reshape(-1, 3)

            landmark_positions_ecef = np.concatenate(
                (landmark_positions_ecef, centroids_ecef), axis=0
            )
            pixel_coordinates = np.concatenate(
                (pixel_coordinates, landmarks.centroid_xy), axis=0
            )
            confidence_scores = np.concatenate(
                (confidence_scores, landmarks.confidence_scores), axis=0
            )

        if len(confidence_scores) == 0:
            print("No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_eci = (ecef_R_eci.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = (
            self.earth_image_simulator.camera.pixel_to_bearing_unit_vector(
                pixel_coordinates
            )
        )
        bearing_unit_vectors = (self.body_R_camera @ bearing_unit_vectors_cf.T).T

        print(f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidence_scores too
        return bearing_unit_vectors, landmark_positions_eci
