# from abc import ABC, abstractmethod
from typing import Any, Tuple
# from time import perf_counter, time
from datetime import datetime
import yaml
import os
# import pickle

import numpy as np
from scipy.spatial.transform import Rotation
# from matplotlib import pyplot as plt

import brahe
from brahe.epoch import Epoch
# from brahe.constants import R_EARTH, GM_EARTH

# from utils.time import increment_epoch
# from dynamics.orbital_dynamics import f
from image_simulation.earth_vis import EarthImageSimulator, lat_lon_to_ecef
from vision_inference.ml_pipeline import MLPipeline
from vision_inference.camera import Frame


class LandmarkBearingSensor:
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark bearing measurements.
    """

    def __init__(self, config):
        """
        :param config: The configuration dictionary.
        """
        quat_body_to_camera = np.asarray(config["satellite"]["camera"]["orientation_in_cubesat_frame"])
        self.R_camera_to_body = Rotation.from_quat(quat_body_to_camera, scalar_first=True).inv().as_matrix()
        self.ml_pipeline = MLPipeline()
        # self.earth_image_simulator = EarthImageSimulator()

    def take_measurement(self, epoch: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        R_eci_to_ecef = brahe.frames.rECItoECEF(epoch)
        position_ecef = R_eci_to_ecef @ cubesat_position
        R_body_to_ecef = R_eci_to_ecef @ R_body_to_eci

        print(f"Taking measurement at {epoch=}, {cubesat_position=}, {R_body_to_eci=}")

        # simulate image
        # TODO: Get image from the camera rather than the simulator
        image = self.earth_image_simulator.simulate_image(position_ecef, R_body_to_ecef)

        if np.all(image == 0):
            print("No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        frame = Frame(image, 0, datetime.now())
        # TODO: queue requests to the model and send them in batches as the sim runs
        regions_and_landmarks = self.ml_pipeline.run_ml_pipeline_on_single(frame)

        # save the image with the detected landmarks
        epoch_str = str(epoch) \
            .replace(':', '_') \
            .replace(' ', '_') \
            .replace('.', '_')
        output_dir = os.path.abspath(
            # TODO: Determine path for actual imagery
            os.path.join(__file__, f"../log/images/seed_69420_epoch_{epoch_str}/"))
        os.makedirs(output_dir, exist_ok=True)
        self.ml_pipeline.visualize_landmarks(frame, regions_and_landmarks, output_dir)

        landmark_positions_ecef = np.zeros(shape=(0, 3))
        pixel_coordinates = np.zeros(shape=(0, 2))
        confidence_scores = np.zeros(shape=(0,))

        for _, landmarks in regions_and_landmarks:
            centroids_ecef = lat_lon_to_ecef(landmarks.centroid_latlons[np.newaxis, ...]).reshape(-1, 3)

            landmark_positions_ecef = np.concatenate((landmark_positions_ecef, centroids_ecef), axis=0)
            pixel_coordinates = np.concatenate((pixel_coordinates, landmarks.centroid_xy), axis=0)
            confidence_scores = np.concatenate((confidence_scores, landmarks.confidence_scores), axis=0)

        if len(confidence_scores) == 0:
            print("No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_eci = (R_eci_to_ecef.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = self.earth_image_simulator.camera.pixel_to_bearing_unit_vector(pixel_coordinates)
        bearing_unit_vectors = (self.R_camera_to_body @ bearing_unit_vectors_cf.T).T

        print(f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidence_scores too
        return bearing_unit_vectors, landmark_positions_eci
