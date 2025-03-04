"""
This module contains utility functions for working with the brahe library.
"""

import os

import brahe
from brahe.epoch import Epoch


def increment_epoch(epoch: Epoch, dt: float) -> Epoch:
    """
    Increments the current epoch by the specified time step.

    :param epoch: The current epoch as an instance of brahe's Epoch class.
    :param dt: The amount of time to increment the epoch by, in seconds.
    """
    return Epoch(*brahe.time.jd_to_caldate(epoch.jd() + dt / (24 * 60 * 60)))


def load_brahe_data_files() -> None:
    """
    Load up-to-date brahe files.
    """
    brahe_directory = os.path.dirname(brahe.__file__)
    try:
        print("Updating Brahe data files. Might take a minute ...")
        brahe.utils.download_all_data(brahe_directory + "/data")
    # pylint: disable=bare-except
    except Exception as e:
        print(f"One or the other files always errors out. Not a problem though. Exception: {e}")
