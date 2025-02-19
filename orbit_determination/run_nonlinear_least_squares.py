import pickle

import numpy as np

from orbit_determination.nonlinear_least_squares_od import OrbitDetermination
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager


def main(od_simulation_data_path: str = "od-simulation-data-1739916755.3107357.pkl") -> None:
    """
    Run the nonlinear least squares orbit determination algorithm.

    :param od_simulation_data_path: The file containing the simulation data.
    """
    with open(od_simulation_data_path, "rb") as f:
        od_simulation_data : ODSimulationDataManager = pickle.load(f)

    od = OrbitDetermination(od_simulation_data.dt)
    estimated_states = od.fit_orbit(od_simulation_data, semi_major_axis_guess=600e3)

    rms_error = np.sqrt(np.mean((estimated_states[:, :3] - od_simulation_data.states[:, :3])**2))
    print(f"RMS error: {rms_error:.2f} m")


if __name__ == "__main__":
    main()
