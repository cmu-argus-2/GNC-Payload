"""
getMGRS.py
Generates a grid that transforms MGRS (Military Grid Reference System) coordinates to corresponding longitude and latitude ranges.
May return specific longitude and latitude ranges for a given MGRS coordinate if a key is provided.
Author: Kyle McCleary
"""

import argparse

from utils.earth_utils import get_MGRS_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", default=None, type=str)
    args = parser.parse_args()
    if args.key:
        grid = get_MGRS_grid()
        print(grid[args.key])
