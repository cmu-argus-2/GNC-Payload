#!/bin/bash

# Default values for configurations
BASE_OUTPATH="Landsat_Data"
LANDMARK_BASE="$(pwd)/Landsat_Data"  # Path relative to the current directory
FINAL_OUTPUT_PATH="$(pwd)/LD/datasets"  # Path relative to the current directory
VALFLAG=False
CURR_DIR="$(pwd)"

# Parse command line arguments (optional)
while getopts ":o:h" opt; do
  case $opt in
    o) FINAL_OUTPUT_PATH="$OPTARG" ;;
    h) echo "Usage: $0 [options]" ;;
    \?) echo "Invalid option: -$OPTARG" ;;
  esac
done

# Array of keys
KEYS=('05V' '09V' '10S' '10T' '11R' '12R' '14Q' '15V' '16T' '18Q' '18S' '19J' '21H' '23L' '29Q' '30U' '32S' '32T' '33K' '33S' '33T' '35J' '36L' '37Q' '38K' '39P' '40R' '42R' '46Q' '48M' '49S' '50M' '51J' '52S' '53L' '54S' '54U' '55J' '57V' '59G') # Add or remove keys as needed

# Prepare YOLO dataset and train the model
for KEY in "${KEYS[@]}"; do
  # Prepare YOLO dataset
  python ./DataPipeline/prepare_yolo_data.py --data_path "$BASE_OUTPATH/$KEY" --landmark_path "$LANDMARK_BASE/$KEY/landmarks" --output_path "$FINAL_OUTPUT_PATH/${KEY}_dataset" --r "$KEY" --val $VALFLAG

  # Train YOLO model
  python ./LD/train_yolo.py --data "$CURR_DIR/LD/datasets/${KEY}_dataset/dataset.yaml" --save_dir "$CURR_DIR/LD/runs" --region "$KEY" --version "yolov8n" --epochs 10
done
