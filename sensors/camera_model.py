import numpy as np

from utils.config_utils import load_config


class CameraModel:
    RESOLUTION = (4608, 2592)
    HORIZONTAL_FOV = 66.1

    def __init__(self, camera_name: str, body_R_camera: np.ndarray, t_body_to_camera: np.ndarray):
        """
        Initialize the simulation camera parameters

        Parameters:
            camera_name: The name of the camera.
            body_R_camera: A numpy array of shape (3, 3) representing the rotation matrix from body to camera frame.
            t_body_to_camera: A numpy array of shape (3,) representing the translation vector from body to camera frame, in the body frame.
        """
        self.camera_name = camera_name
        self.body_R_camera = body_R_camera
        self.t_body_to_camera = t_body_to_camera

    def get_camera_position(
        self, body_position: np.ndarray, frame_R_body: np.ndarray
    ) -> np.ndarray:
        """
        Get the camera position in the frame of interest.

        Parameters:
            body_position: A numpy array of shape (3,) representing the position of the body in the frame of interest.
            frame_R_body: A numpy array of shape (3, 3) representing the rotation matrix from the body frame to the frame of interest.

        Returns:
            A numpy array of shape (3,) representing the position of the camera in the frame of interest.
        """
        return body_position + frame_R_body @ self.t_body_to_camera

    def get_camera_axis(self, frame_R_body: np.ndarray) -> np.ndarray:
        """
        Get the camera's boresight axis in the frame of interest.

        Parameters:
            frame_R_body: A numpy array of shape (3, 3) representing the rotation matrix from the body frame to the frame of interest.

        Returns:
            A numpy array of shape (3,) representing the camera's boresight axis in the frame of interest.
        """
        return frame_R_body @ self.body_R_camera @ np.array([0, 0, 1])

    def ray_directions(self):
        """
        Generate ray directions for the camera.

        Returns:
            A numpy array of shape (CameraModel.RESOLUTION) + (3,) consisting of ray directions
            in the body frame for each pixel.
        """
        width, height = self.RESOLUTION
        half_width = np.tan(self.HORIZONTAL_FOV / 2)
        half_height = half_width * (height / width)

        x = np.linspace(-half_width, half_width, width)
        y = np.linspace(-half_height, half_height, height)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)  # Assume unit depth

        # Stack and normalize ray directions
        # TODO: precompute this and store it as a class attribute
        ray_directions_cf = np.stack([xx, yy, zz], axis=-1)
        ray_directions_cf /= np.linalg.norm(ray_directions_cf, axis=-1, keepdims=True)

        ray_directions_body = ray_directions_cf @ self.body_R_camera.T
        return ray_directions_body

    def pixel_to_bearing_unit_vector(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Converts pixel coordinates to bearing unit vectors in the body frame.

        Parameters:
            pixel_coords: An array of shape (N, 2) with pixel coordinates.

        Returns:
            A numpy array of shape (N, 3) with bearing unit vectors in the body frame.
        """
        width, height = self.RESOLUTION

        half_width = np.tan(self.HORIZONTAL_FOV / 2)
        half_height = half_width * (height / width)

        u = pixel_coords[:, 0]  # Pixel x-coordinates
        v = pixel_coords[:, 1]  # Pixel y-coordinates

        # Normalize pixel coordinates to range [-half_width, half_width] and [half_height, -half_height]
        # Assuming pixel (0,0) is at the top-left corner
        x = -half_width + (2 * half_width) * (u / (width - 1))
        y = half_height - (2 * half_height) * (
            v / (height - 1)
        )  # Invert y-axis for image coordinates
        z = np.ones_like(x)

        # Stack and normalize direction vectors
        bearing_unit_vectors_cf = np.stack([x, y, z], axis=-1)
        bearing_unit_vectors_cf /= np.linalg.norm(bearing_unit_vectors_cf, axis=1, keepdims=True)

        bearing_unit_vectors_body = bearing_unit_vectors_cf @ self.body_R_camera.T
        return bearing_unit_vectors_body


class CameraManager:
    CAMERA_NAMES = ["x+", "y+", "x-", "y-"]
    CAMERA_AXES = {
        "x+": np.array([1, 0, 0]),
        "y+": np.array([0, 1, 0]),
        "x-": np.array([-1, 0, 0]),
        "y-": np.array([0, -1, 0]),
    }

    def __init__(self):
        self.camera_models = CameraManager.initialize_cameras()

    def __getitem__(self, camera_name: str) -> CameraModel:
        """
        Get the CameraModel object for the specified camera.

        Parameters:
            camera_name: The name of the camera.

        Returns:
            The CameraModel object for the specified camera.
        """
        self.validate_camera_name(camera_name)
        return self.camera_models[camera_name]

    @staticmethod
    def initialize_cameras() -> dict[str, CameraModel]:
        """
        Initialize camera models for all cameras.

        Returns:
            dict: A dictionary mapping camera names to CameraModel objects.
        """
        camera_models = {}
        for camera_info in load_config()["satellite"]["cameras"]:
            camera_models[camera_info["name"]] = CameraModel(
                camera_info["name"],
                np.array(camera_info["body_R_camera"]),
                np.array(camera_info["t_body_to_camera"]),
            )
        return camera_models

    @staticmethod
    def validate_camera_name(camera_name: str) -> None:
        if camera_name not in CameraManager.CAMERA_NAMES:
            raise ValueError(f"Invalid camera name: {camera_name}")
