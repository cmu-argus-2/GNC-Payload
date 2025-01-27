"""
This module contains utility functions for loading configuration files.
"""

from typing import Any

import yaml

def load_config(config_path) -> Any:
    """Loads a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
