#!/bin/bash

# Default values for configurations
BOUNDS="-84 24 -78 32"
IDATE="2020-05-01"
FDATE="2023-12-31"
LANDSAT=8
MAXIMS=500
SCALE=328
BOX_WIDTH=76
BOX_COUNT=100
VALFLAG=True

BASE_OUTPATH="Landsat_Data"

LANDMARK_BASE="/home/${USER}/VisionTrainingGround/Landsat_Data" # Path may be adjusted as needed
FINAL_OUTPUT_PATH="/home/${USER}/VisionTrainingGround/LD/datasets" # Path may be adjusted as needed

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "-b    BOUNDS        Geographic bounds (format: 'minLon minLat maxLon maxLat'). Default: '$BOUNDS'"
    echo "-i    IDATE         Initial date (format: YYYY-MM-DD). Default: '$IDATE'"
    echo "-f    FDATE         Final date (format: YYYY-MM-DD). Default: '$FDATE'"
    echo "-l    LANDSAT       Landsat version. Default: $LANDSAT"
    echo "-m    MAXIMS        Maximum number of images. Default: $MAXIMS"
    echo "-s    SCALE         Scale. Default: $SCALE"
    echo "-w    BOX_WIDTH     Width of the boxes. Default: $BOX_WIDTH"
    echo "-n    BOX_COUNT     Number of boxes. Default: $BOX_COUNT"
    echo "-o    OUTPATH       Final output path. Default: '$FINAL_OUTPUT_PATH'"
    echo "-h                  Display this help and exit"
    echo ""
}

# Parse command line arguments
while getopts ":b:i:f:l:m:s:w:n:o:h" opt; do
  case $opt in
    b) BOUNDS="$OPTARG"
    ;;
    i) IDATE="$OPTARG"
    ;;
    f) FDATE="$OPTARG"
    ;;
    l) LANDSAT="$OPTARG"
    ;;
    m) MAXIMS="$OPTARG"
    ;;
    s) SCALE="$OPTARG"
    ;;
    w) BOX_WIDTH="$OPTARG"
    ;;
    n) BOX_COUNT="$OPTARG"
    ;;
    o) FINAL_OUTPUT_PATH="$OPTARG"
    ;;
    h) show_help
       exit 0
    ;;
    \?) echo "Invalid option: -$OPTARG" >&2
       show_help
       exit 1
    ;;
  esac
done

# Array of keys to iterate through
KEYS=("17R") # Add or remove keys as needed

# Path to python bin may be adjusted as needed
# Main processing loop
for KEY in "${KEYS[@]}"; do
  # Run earthenginedl.py
  /home/${USER}/miniconda3/envs/sat_env_vision/bin/python ./DataPipeline/earthenginedl.py --bounds $BOUNDS --idate "$IDATE" --fdate "$FDATE" --landsat $LANDSAT --grid_key "$KEY" --region "$KEY" --maxims $MAXIMS --scale $SCALE --outpath "$BASE_OUTPATH/$KEY"
  
  # Run saliencymap.py
  /home/${USER}/miniconda3/envs/sat_env_vision/bin/python ./DataPipeline/saliencymap.py --dir_path "$BASE_OUTPATH/$KEY" --crs EPSG:4326 --grid_key "$KEY"
  
  # Run saliencymap2boxes.py
  /home/${USER}/miniconda3/envs/sat_env_vision/bin/python ./DataPipeline/saliencymap2boxes.py -k "$KEY" -w $BOX_WIDTH -n $BOX_COUNT -p "$BASE_OUTPATH/$KEY/landmarks"
  
  # Run prepare_yolo_data.py with configurable output path
  /home/${USER}/miniconda3/envs/sat_env_vision/bin/python ./DataPipeline/prepare_yolo_data.py --data_path "$BASE_OUTPATH/$KEY" --landmark_path $LANDMARK_BASE/$KEY/landmarks --output_path $FINAL_OUTPUT_PATH/${KEY}_dataset --r "$KEY" --val $VALFLAG
done
