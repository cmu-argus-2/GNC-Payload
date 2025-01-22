#!/bin/bash

# Default values for configurations
BOUNDS="-84 24 -78 32"
IDATE="2020-05-01"
FDATE="2023-12-31"
LANDSAT=8
MAXIMS=500
SCALE=328
BASE_OUTPATH="Landsat_Data"
KEYS=("53L") # Add or remove keys as needed

# Parse command line arguments (optional)
while getopts ":b:i:f:l:m:s:h" opt; do
  case $opt in
    b) BOUNDS="$OPTARG" ;;
    i) IDATE="$OPTARG" ;;
    f) FDATE="$OPTARG" ;;
    l) LANDSAT="$OPTARG" ;;
    m) MAXIMS="$OPTARG" ;;
    s) SCALE="$OPTARG" ;;
    h) echo "Usage: $0 [options]" ;;
    \?) echo "Invalid option: -$OPTARG" ;;
  esac
done

# Path to python bin may be adjusted as needed
for KEY in "${KEYS[@]}"; do
  # Download dataset
  python ./DataPipeline/earthenginedl.py --bounds $BOUNDS --idate "$IDATE" --fdate "$FDATE" --landsat $LANDSAT --grid_key "$KEY" --region "$KEY" --maxims $MAXIMS --scale $SCALE --outpath "$BASE_OUTPATH/$KEY"
  
  # Run saliencymap.py
  python ./DataPipeline/saliencymap.py --dir_path "$BASE_OUTPATH/$KEY" --crs EPSG:4326 --grid_key "$KEY"
  
  # Run saliencymap2boxes.py
  python ./DataPipeline/saliencymap2boxes.py -k "$KEY" -w 76 -n 100 -p "$BASE_OUTPATH/$KEY/landmarks"
done
