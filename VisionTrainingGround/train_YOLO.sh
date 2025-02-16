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
KEYS=("53L") # Add or remove keys as needed

# Prepare YOLO dataset and train the model
for KEY in "${KEYS[@]}"; do
  # Prepare YOLO dataset
  python ./DataPipeline/prepare_yolo_data.py --data_path "$BASE_OUTPATH/$KEY" --landmark_path "$LANDMARK_BASE/$KEY/landmarks" --output_path "$FINAL_OUTPUT_PATH/${KEY}_dataset" --r "$KEY" --val $VALFLAG

  # Train YOLO model
  python ./LD/train_yolo.py --data "$CURR_DIR/LD/datasets/${KEY}_dataset/dataset.yaml" --save_dir "$CURR_DIR/LD/runs" --region "$KEY" --version "yolov8n" --epochs 10
done
