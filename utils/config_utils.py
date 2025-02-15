"""
This module contains utility functions for loading configuration files.
"""

from typing import Any
import os
import yaml


MAIN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.yaml"))


def load_config(config_path: str = MAIN_CONFIG_PATH) -> Any:
    """
    Loads a YAML configuration file.

    :param config_path: The path to the configuration file. If not provided, the main configuration file is loaded.
    :return: The contents of the configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
