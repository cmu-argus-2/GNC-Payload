import numpy as np
from brahe import R_EARTH
from matplotlib import pyplot as plt


def animate_orbits(positions: np.ndarray, estimated_positions: np.ndarray, landmarks: np.ndarray) -> None:
    """
    Creates an animation where the orbital paths of the true and estimated states are plotted as evolving over time.
    The landmarks are also plotted statically.

    :param positions: A numpy array of shape (N, 3) containing the true positions.
    :param estimated_positions: A numpy array of shape (N, 3) containing the estimated positions.
    :param landmarks: A numpy array of shape (M, 3) containing the landmark positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for t in range(positions.shape[0]):
        start_idx = max(0, t - 100)
        ax.clear()
        ax.plot(positions[start_idx:t, 0],
                positions[start_idx:t, 1],
                positions[start_idx:t, 2],
                label="True orbit")
        ax.plot(estimated_positions[start_idx:t, 0],
                estimated_positions[start_idx:t, 1],
                estimated_positions[start_idx:t, 2],
                label="Estimated orbit")
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")

        ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.pause(0.01)
