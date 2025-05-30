#!/bin/bash
# The purpose of this script is to run the scripts that the Jetson would expect as inputs. 
# This means running the run_dynamics.py script and the run_earth_vis.py script. These are what the
# Jetson would receive as input in the form of camera imagery and ground-truth states that we use in
# the EKF.

# Process: Run dynamics, run vision 
# Source conda
# source ~/.bashrc
# conda init bash
# TODO: Set the correct conda environment for the Jetson
# conda activate sat_env_vision

# filepath: /home/argus/Arvind/GNC-Payload/scripts/run_prep.sh
bash -i -c "
    conda activate sat_env_vision
    LAT=0
    LON=-80
    NAME=testrun4
    MEAS_RATE=30
    DURATION=2700
    python scripts/run_dynamics.py --lat \$LAT --lon \$LON --name \$NAME --northwards True --duration \$DURATION
    python scripts/run_earth_vis.py --name \$NAME --meas_rate \$MEAS_RATE
"