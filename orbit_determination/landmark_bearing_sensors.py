"""
Module that manages the different landmark bearing sensors.
"""

import os
import typing
from abc import ABC, abstractmethod
from typing import List, Tuple

import brahe
import cv2
import numpy as np
from brahe import R_EARTH, Epoch

# pylint: disable=import-error
from simulation.image_simulation.earth_vis import EarthImageSimulator
from simulation.image_simulation.cesium_simulator import CesiumEarthImageSimulator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from simulation.sensors.camera_model import CameraModel

from utils.config_utils import USER_CONFIG_PATH, load_config
from utils.earth_utils import lat_lon_to_ecef, noisy_bearing_measurement
from vision_inference.landmark_detector import LandmarkDetector
from vision_inference.logger import Logger
from vision_inference.ml_pipeline import MLPipeline
from datetime import timezone

# pylint: disable=too-few-public-methods
class LandmarkBearingSensor(ABC):
    """
    Abstract class for a landmark bearing sensor, which inputs the satellite pose and outputs
    landmark bearing measurements.
    """

    @abstractmethod
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a landmark bearing measurement using the sensor.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy
        array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors
        in the body frame and a numpy array of shape (N, 3) containing the landmark positions in
        ECI coordinates.
        """


class RandomLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that randomly generates landmark bearing measurements within a cone centered about the camera's boresight.
    """

    def __init__(self, max_measurements: int = 10, fov: float = np.deg2rad(120)) -> None:
        """
        :param fov: The field of view of the camera in radians.
        :param max_measurements: The number of measurements to attempt to take at once. The actual number may be less.
        """
        self.max_measurements = max_measurements
        self.fov = fov
        self.cos_fov = np.cos(fov)

        # Scaling of the noise in measurement
        self.scale = np.sqrt(0.0005)

    def sample_bearing_unit_vectors(self, camera_model: CameraModel) -> np.ndarray:
        """
        Sample self.max_measurements random bearing unit vectors in the body frame that are within the camera's field
        of view, which is a cone centered about the camera's boresight.

        :param camera_model: The camera model to use for the measurement.
        :return: A numpy array of shape (self.max_measurements, 3) containing the sampled bearing unit vectors in the
        body frame.
        """
        phi = 2 * np.pi * np.random.random(self.max_measurements)
        # uniformly sample cos(theta) instead of theta to get a uniform distribution on the unit sphere
        theta = np.arccos(np.random.uniform(self.cos_fov, 1, self.max_measurements))
        bearing_unit_vectors_cf = Rotation.from_euler("ZX", np.column_stack((phi, theta))).apply(
            np.array([0, 0, 1])
        )

        # sanity check
        assert np.all(bearing_unit_vectors_cf[:, 2] > self.cos_fov)

        bearing_unit_vectors_body = (camera_model.body_R_camera @ bearing_unit_vectors_cf.T).T
        return bearing_unit_vectors_body

    @staticmethod
    def get_ray_and_earth_intersections(
        ray_dirs: np.ndarray, ray_start: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the intersection points of rays with the Earth.
        The input number of rays, N, the output number of intersection points, M,
        and the returned boolean array, valid_intersections, are related as follows:
        M == np.sum(valid_intersections) <= N.

        :param ray_dirs: A numpy array of shape (N, 3) containing the direction vectors of the rays in ECI coordinates.
                         Note that the direction vectors must all be normalized.
        :param ray_start: A numpy array of shape (3,) containing the starting point of the rays in ECI coordinates.
        :return: A tuple containing a boolean array  of shape (N,) indicating which rays intersected the Earth,
                 and a numpy array of shape (M, 3) containing the intersection points in ECI coordinates.
        """
        assert np.allclose(np.linalg.norm(ray_dirs, axis=1), 1), "ray_dirs must be normalized"

        # As = np.sum(ray_dirs ** 2, axis=1)  # this is always 1 since the rays are normalized
        bs = 2 * ray_dirs @ ray_start
        C = np.sum(ray_start**2) - R_EARTH**2
        assert C > 0, "The ray start location is inside the Earth!"

        discriminants = bs**2 - 4 * C
        # pylint: disable=pointless-string-statement
        """
        Since C > 0 and np.all(As > 0), if the roots are real they must have the same sign.
        bs < 0 implies that the slope at x = 0 is negative, so the roots are positive.
        Intuitively, this check is equivalent to np.dot(ray_dir, ray_start) < 0 which checks if ray_dir is in
        the half-space that is pointing towards the Earth.
        """
        valid_intersections = (discriminants >= 0) & (bs < 0)

        # pick the smaller of the two positive roots from the quadratic formula, since it is closer to the camera
        ts = (-bs[valid_intersections] - np.sqrt(discriminants[valid_intersections])) / 2
        intersection_points = ray_start + ts[:, np.newaxis] * ray_dirs[valid_intersections, :]

        assert intersection_points.shape[0] == np.sum(valid_intersections)
        return valid_intersections, intersection_points

    def take_measurement(
        self,
        _: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.
        The number of measurements, N, will be some number less than or equal to self.max_measurements.

        :param _: The epoch as an instance of brahe's Epoch class. Not used.
        :param cubesat_position: The position of the satellite in ECI [m] as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        bearing_unit_vectors_body = self.sample_bearing_unit_vectors(camera_model)
        bearing_unit_vectors_eci = (eci_R_body @ bearing_unit_vectors_body.T).T
        camera_position_eci = camera_model.get_camera_position(cubesat_position, eci_R_body)

        valid_intersections, landmark_positions_eci = self.get_ray_and_earth_intersections(
            bearing_unit_vectors_eci, camera_position_eci
        )
        bearing_unit_vectors_body = bearing_unit_vectors_body[valid_intersections, :]

        # sanity check
        for bearing_unit_vector_body, landmark_position_eci in zip(
            bearing_unit_vectors_body, landmark_positions_eci
        ):
            # Convert from m to km
            # landmark_position_eci /= 1e3
            true_bearing_unit_vector_eci = landmark_position_eci - cubesat_position
            true_bearing_unit_vector_eci /= np.linalg.norm(true_bearing_unit_vector_eci)

            # Check that the angle between the two unit vectors is small
            vec_expected = true_bearing_unit_vector_eci
            vec_actual = eci_R_body @ bearing_unit_vector_body
            dot_prod = np.clip(np.dot(vec_expected, vec_actual), -1.0, 1.0)
            angle = np.rad2deg(np.arccos(dot_prod))
            # allow small angular error (degrees)
            assert angle < 1e-5, f"Angle between bearings too large: {angle} deg"


        bearing_unit_vectors_body_noisy = noisy_bearing_measurement(
            bearing_unit_vectors_body, self.scale
        )

        return bearing_unit_vectors_body_noisy, landmark_positions_eci


class GroundTruthLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that outputs the ground truth landmark bearing to all salient landmarks within a cone centered
    about the camera's boresight.
    Note that this DOES NOT (yet) accurately simulate the camera's field of view.
    """

    def __init__(self, fov: float = np.deg2rad(100)) -> None:
        self.fov = fov
        self.cos_fov_on_2 = np.cos(fov / 2)
        self.region_landmarks_ecef = GroundTruthLandmarkBearingSensor.load_region_landmark_ecef()

        # Scaling of the noise in measurement
        self.scale = 0.001

    @staticmethod
    def load_region_landmark_ecef() -> dict[str, np.ndarray]:
        """
        Load the ECEF coordinates of the landmarks from the CSV files for all salient regions.

        :return: A dictionary mapping region identifiers to numpy array of shape (N, 3) containing
                 the coordinates of the landmarks in ECEF.
        """
        salient_regions: List[str] = load_config()["vision"]["salient_mgrs_region_ids"]
        models_dir = load_config(USER_CONFIG_PATH)["models_directory"]

        region_landmarks_ecef = {}
        for region_id in salient_regions:
            region_landmarks = LandmarkDetector.load_ground_truth(
                os.path.join(
                    models_dir,
                    LandmarkDetector.get_region_bounding_boxes_relative_path(region_id),
                )
            )
            region_landmarks_ecef[region_id] = lat_lon_to_ecef(region_landmarks[:, :2])
        return region_landmarks_ecef

    # pylint: disable=too-many-locals
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
        three_d: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to ECI as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :param three_d: Whether to return 3D bearing vectors or 2D projections.
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        ecef_R_eci = brahe.frames.rECItoECEF(epoch)
        position_ecef = ecef_R_eci @ cubesat_position
        ecef_R_body = ecef_R_eci @ eci_R_body

        camera_axis_ecef = camera_model.get_camera_axis(ecef_R_body)
        camera_position_ecef = 1e-3*camera_model.get_camera_position(1e3*position_ecef, ecef_R_body)

        # TODO: optimize this by using the MGRS regions to filter out landmarks that are definitely not visible
        # Convert landmark positions from m to km
        all_landmarks_ecef = np.concatenate(list(self.region_landmarks_ecef.values()), axis=0) / 1e3

        is_same_hemisphere = all_landmarks_ecef @ camera_position_ecef > 0
        hemisphere_landmarks_ecef = all_landmarks_ecef[is_same_hemisphere, :]

        bearing_vectors_ecef = hemisphere_landmarks_ecef - camera_position_ecef
        bearing_unit_vectors_ecef = bearing_vectors_ecef / np.linalg.norm(
            bearing_vectors_ecef, axis=1, keepdims=True
        )

        is_visible = (
            (bearing_unit_vectors_ecef @ camera_axis_ecef > self.cos_fov_on_2)
            & (np.linalg.norm(camera_position_ecef) < np.linalg.norm(position_ecef))
        )
        visible_landmarks_ecef = hemisphere_landmarks_ecef[is_visible, :]
        visible_landmarks_eci = (ecef_R_eci.T @ visible_landmarks_ecef.T).T

        bearing_unit_vectors_body = (ecef_R_body.T @ bearing_unit_vectors_ecef[is_visible, :].T).T
        bearing_unit_vectors_body_noisy = noisy_bearing_measurement(
            bearing_unit_vectors_body, self.scale
        )

        return bearing_unit_vectors_body_noisy, visible_landmarks_eci


class SimulatedMLLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark
    bearing measurements.
    """

    def __init__(
        self,
        use_cesium: bool = False,
        run_inference: bool = True,
        ld_version: int = 1,
        mgrs_gzd: str = "17R",
        save_lat_lon: bool = False,
        write_labels: bool = True,
    ) -> None:
        """
        Initialize this SimulatedMLLandmarkBearingSensor.

        :param use_cesium: If True, render images via the local CesiumJS server instead of
                           GeoTIFF files. The server must be running before instantiation:
                               cd cesium_server && npm install && npm start
        :param run_inference: If False, images are simulated and saved to disk but the ML
                              pipeline is skipped and every call returns zero measurements.
                              Useful for building an image dataset before inference is ready.
        :param ld_version: Integer version of the landmark detection model (e.g. 1 for V1, 2 for V2).
                           Used to locate bounding_boxes.csv for ground-truth label generation.
        :param mgrs_gzd: MGRS Grid Zone Designator (e.g. "17R") for this simulation run.
        :param save_lat_lon: If True, save per-pixel lat/lon arrays as .npz alongside each image.
        :param write_labels: If True, write YOLO labels for each frame. Disable to speed up image generation.
        """
        self.run_inference = run_inference
        self.ml_pipeline = MLPipeline() if run_inference else None
        self.earth_image_simulator = (
            CesiumEarthImageSimulator() if use_cesium else EarthImageSimulator()
        )
        self.ld_version = ld_version
        self.mgrs_gzd = mgrs_gzd
        self.save_lat_lon = save_lat_lon
        self.write_labels = write_labels

    def _write_yolo_labels(
        self,
        lat_lon: np.ndarray,
        image_shape: tuple,
        output_path: str,
    ) -> None:
        """
        Write a YOLO-v8 label file for the simulated image using ground-truth bounding boxes.

        Landmark class IDs are the 0-based row indices in bounding_boxes.csv.
        Only landmarks whose centroid maps to a valid (non-NaN) pixel within the image are written.

        :param lat_lon: (H, W, 2) array of latitude/longitude [deg] per pixel; NaN for sky pixels.
        :param image_shape: (H, W) pixel dimensions of the image.
        :param output_path: Destination .txt file path.
        """
        _repo_root = os.path.abspath(os.path.join(__file__, "../../"))
        csv_path = os.path.join(
            _repo_root, "Vision-Models", "trained-ld",
            f"V{self.ld_version}", self.mgrs_gzd, "bounding_boxes.csv"
        )
        if not os.path.exists(csv_path):
            open(output_path, "w").close()
            return

        H, W = image_shape

        # columns: centroid_lon, centroid_lat, tl_lon, tl_lat, br_lon, br_lat
        bboxes = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if bboxes.ndim == 1:
            bboxes = bboxes[np.newaxis, :]
        n = len(bboxes)

        # Build KD-tree over valid (non-NaN) pixels using [lat, lon] coordinates
        valid_mask = ~np.isnan(lat_lon[:, :, 0])
        valid_rows, valid_cols = np.where(valid_mask)
        if len(valid_rows) == 0:
            open(output_path, "w").close()
            return

        valid_latlon = lat_lon[valid_rows, valid_cols, :]  # (N_valid, 2): [lat, lon]
        tree = cKDTree(valid_latlon)

        # Batch-query centroid, top-left, and bottom-right for every landmark
        queries = np.vstack([
            bboxes[:, [1, 0]],  # centroid [lat, lon]
            bboxes[:, [3, 2]],  # TL       [lat, lon]
            bboxes[:, [5, 4]],  # BR       [lat, lon]
        ])
        dists, idxs = tree.query(queries)

        c_rows = valid_rows[idxs[:n]]
        c_cols = valid_cols[idxs[:n]]
        tl_rows = valid_rows[idxs[n:2 * n]]
        tl_cols = valid_cols[idxs[n:2 * n]]
        br_rows = valid_rows[idxs[2 * n:]]
        br_cols = valid_cols[idxs[2 * n:]]
        c_dists = dists[:n]

        # Landmarks whose nearest pixel is >0.5 deg away are outside this image's FOV
        MAX_DIST_DEG = 0.5

        lines = []
        for i in range(n):
            if c_dists[i] > MAX_DIST_DEG:
                continue

            cx_px, cy_px = int(c_cols[i]), int(c_rows[i])
            w_px = abs(int(br_cols[i]) - int(tl_cols[i]))
            h_px = abs(int(br_rows[i]) - int(tl_rows[i]))
            if w_px == 0 or h_px == 0:
                continue

            cx_norm = cx_px / W
            cy_norm = cy_px / H
            w_norm = w_px / W
            h_norm = h_px / H
            lines.append(f"{i} {cx_norm} {cy_norm} {w_norm} {h_norm}")

        with open(output_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    # pylint: disable=too-many-locals,R0913,R0917,W0221
    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
        index: int,
        output_dir: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param eci_R_body: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :param camera_model: The camera model to use for the measurement.
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        Logger.log("INFO", f"Taking measurement at epoch={epoch}, cam={camera_model.get_camera_name}")

        ecef_R_eci = brahe.frames.rECItoECEF(epoch)
        position_ecef = ecef_R_eci @ cubesat_position
        ecef_R_body = ecef_R_eci @ eci_R_body

        # simulate image
        frame, lat_lon = self.earth_image_simulator.simulate_image_for_training(
            position_ecef, ecef_R_body, camera_model
        )
        camera_name = camera_model.get_camera_name
        cam_id = {'x+': 'xp', 'x-': 'xm', 'y+': 'yp', 'y-': 'ym'}[camera_name]
        timestamp_ms = int(epoch.to_datetime().replace(tzinfo=timezone.utc).timestamp() * 1000)

        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"raw_{timestamp_ms}_{cam_id}.jpg"),
            cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if self.save_lat_lon:
            np.savez_compressed(os.path.join(output_dir, f"lat_lon_{timestamp_ms}_{cam_id}.npz"), lat_lon)
        if self.write_labels:
            self._write_yolo_labels(
                lat_lon,
                frame.image.shape[:2],
                os.path.join(output_dir, f"{self.mgrs_gzd}_{timestamp_ms}_{cam_id}.txt"),
            )

        if np.all(frame.image == 0):
            Logger.log("INFO", "No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        if not self.run_inference:
            Logger.log("INFO", "Inference disabled — returning zero measurements")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        # TODO: queue requests to the model and send them in batches as the sim runs
        landmark_detections, region_slices = self.ml_pipeline.run_ml_pipeline_on_single(frame)

        # save the image with the detected landmarks
        epoch_str = str(epoch).replace(":", "_").replace(" ", "_").replace(".", "_")
        output_dir = os.path.abspath(
            os.path.join(__file__, f"../log/simulated_images/seed_69420_epoch_{epoch_str}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        MLPipeline.visualize_landmarks(frame, landmark_detections, region_slices, output_dir)

        if len(region_slices) is None:
            Logger.log("INFO", "No salient regions detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))
        if len(landmark_detections) == 0:
            Logger.log("INFO", "No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_ecef = lat_lon_to_ecef(landmark_detections.latlons)
        landmark_positions_eci = (ecef_R_eci.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = camera_model.pixel_to_bearing_unit_vector(
            landmark_detections.pixel_coordinates
        )
        bearing_unit_vectors_body = (camera_model.body_R_camera @ bearing_unit_vectors_cf.T).T

        Logger.log("INFO", f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidences too
        return bearing_unit_vectors_body, landmark_positions_eci


class SimulatedMLStoredLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that uses already calculated landmark bearing measurements and landmark locations
    to provide correct measurements at each timestep.
    """

    def __init__(self, output_basedir: str) -> None:
        # Set up paths to stored data
        self.vis_inf_dir = "vis_inf"
        self.base_bearing_dir = output_basedir

    def load_measurements(self, timestep, camera_name):
        """
        Load measurements.
        """
        # Load the bearing vectors and landmark positions from the stored data
        base_name = f"{timestep}_{camera_name}"
        file_path = os.path.join(
            self.base_bearing_dir,
            self.vis_inf_dir,
            str(timestep),
            "bearing_vectors",
            f"landmarks_{base_name}.npz",
        )
        Logger.log(
            "INFO",
            f"Looking for bearing vectors from {file_path} for camera {camera_name} at timestep {timestep}",
        )
        try:
            # pylint: disable=E1129
            with np.load(file_path) as data:
                bearing_vectors = data["bearing_vectors"]
                landmark_positions = data["landmark_positions"]
                msg = (
                    f"Loaded bearing vectors with shape: {bearing_vectors.shape} ",
                    f"for camera {camera_name} at timestep {timestep}",
                )
                Logger.log("INFO", msg)
            if bearing_vectors.ndim == 1:
                bearing_vectors = np.expand_dims(bearing_vectors, axis=0)
                landmark_positions = np.expand_dims(landmark_positions, axis=0)

            return bearing_vectors, landmark_positions

        except FileNotFoundError:
            Logger.log(
                "INFO", f"No measurements found for camera {camera_name} at timestep {timestep}."
            )
            return np.zeros((0, 3)), np.zeros((0, 3))

    def take_measurement(
        self,
        epoch: Epoch,
        cubesat_position: np.ndarray,
        eci_R_body: np.ndarray,
        camera_model: CameraModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "The take_measurement method is not implemented and shouldn't be used."
        )
