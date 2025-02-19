import pickle
import numpy as np

import brahe
from brahe.constants import R_EARTH

from orbit_determination.od_simulation_data_manager import ODSimulationDataManager
from orbit_determination.nonlinear_least_squares_od import OrbitDetermination
from utils.brahe_utils import increment_epoch
from utils.plotting_utils import animate_orbits


def main(od_simulation_data_path: str = "od-simulation-data-1739916755.3107357.pkl", nls_results_path: str = "nls-results-1739916755.3107357.pkl") -> None:
    """
    Compare the results of the nonlinear least squares orbit determination algorithm with the ground truth and
    generate plots to visualize the results.
    """
    with open(od_simulation_data_path, "rb") as f:
        data_manager: ODSimulationDataManager = pickle.load(f)

    with open(nls_results_path, "rb") as f:
        estimated_states: np.ndarray = pickle.load(f)

    ecef_Rs_eci = np.stack([brahe.rECItoECEF(increment_epoch(data_manager.starting_epoch, i * data_manager.dt))
                               for i in range(data_manager.state_count)], axis=0)

    # fit a circular orbit to the altitude-normalized landmarks to get an initial guess
    altitude_normalized_landmarks = data_manager.landmarks / np.linalg.norm(
        data_manager.landmarks, axis=1, keepdims=True
    )
    model = OrbitDetermination(data_manager.dt).fit_circular_orbit(
        data_manager.measurement_indices, (R_EARTH + 600e3) * altitude_normalized_landmarks
    )
    # pylint: disable=no-member
    circular_estimated_states = model(np.arange(data_manager.state_count))

    rms_error = np.sqrt(np.mean((circular_estimated_states[:, :3] - data_manager.states[:, :3])**2))
    print(f"RMS error: {rms_error:.2f} m")

    animate_orbits(
        np.einsum("ijk,ik->ij", ecef_Rs_eci, data_manager.states[:, :3]),
        np.einsum("ijk,ik->ij", ecef_Rs_eci, estimated_states[:, :3]),
        np.einsum("ijk,ik->ij", ecef_Rs_eci[data_manager.measurement_indices], data_manager.landmarks),
    )


if __name__ == "__main__":
    main()
