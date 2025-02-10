"""
Module to replay simulation data and solve the orbit determination problem using non-linear least squares.
"""
# pylint: disable=import-error
import pickle

from orbit_determination.nonlinear_least_squares_od import OrbitDetermination
from orbit_determination.od_simulation_data_manager import ODSimulationDataManager


def replay_simulation_data(data_file: str) -> None:
    """
    Replay simulation data from a file and solve the orbit determination problem using non-linear least squares.

    :param data_file: The path to the file containing the simulation data.
    """
    with open(data_file, "rb") as file:
        data_manager: ODSimulationDataManager = pickle.load(file)

    od = OrbitDetermination(dt=data_manager.dt)
    od.fit_orbit(data_manager)


if __name__ == "__main__":
    replay_simulation_data("od-simulation-data-*.pkl")
