from functools import cache
import numpy as np
from brahe import R_EARTH
from matplotlib import pyplot as plt


@cache
def load_equirectangular_map() -> np.ndarray:
    """
    Load an equirectangular map of the Earth.

    :return: A numpy array of shape (H, W, 3) containing the image data.
    """
    # https://en.wikipedia.org/wiki/Equirectangular_projection#/media/File:Blue_Marble_2002.png
    return plt.imread("equirectangular_map.png")


def plot_ground_track(lat_lons: np.ndarray) -> None:
    """
    Plot the ground track of a satellite's orbit on an equirectangular map.

    :param lat_lons: A numpy array of shape (N, 2) containing latitude and longitude coordinates.
    """
    fig, ax = plt.subplots()
    ax.imshow(load_equirectangular_map(), extent=(-180, 180, -90, 90))
    ax.plot(lat_lons[:, 1], lat_lons[:, 0], color="red")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Ground Track")
    plt.show()


def animate_orbits(
    positions: np.ndarray, estimated_positions: np.ndarray, landmarks: np.ndarray
) -> None:
    """
    Creates an animation where the orbital paths of the true and estimated states are plotted as evolving over time.
    The landmarks are also plotted statically.

    :param positions: A numpy array of shape (N, 3) containing the true positions.
    :param estimated_positions: A numpy array of shape (N, 3) containing the estimated positions.
    :param landmarks: A numpy array of shape (M, 3) containing the landmark positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for t in range(positions.shape[0]):
        start_idx = max(0, t - 100)
        ax.clear()
        ax.plot(
            positions[start_idx:t, 0],
            positions[start_idx:t, 1],
            positions[start_idx:t, 2],
            label="True orbit",
        )
        ax.plot(
            estimated_positions[start_idx:t, 0],
            estimated_positions[start_idx:t, 1],
            estimated_positions[start_idx:t, 2],
            label="Estimated orbit",
        )
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")

        ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.pause(0.01)
