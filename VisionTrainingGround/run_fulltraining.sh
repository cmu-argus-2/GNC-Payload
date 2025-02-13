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
VALFLAG=False # Set manually TODO: Could be added as arg

BASE_OUTPATH="Landsat_Data"
# cd into VisionTrainingGround directory for this to work
CURR_DIR="$(pwd)"
LANDMARK_BASE="$(pwd)/Landsat_Data"  # Path relative to the current directory
FINAL_OUTPUT_PATH="$(pwd)/LD/datasets"  # Path relative to the current directory# 

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
KEYS=("53L") # Add or remove keys as needed

# Path to python bin may be adjusted as needed
# Main processing loop
for KEY in "${KEYS[@]}"; do
  Run earthenginedl.py
  python ./DataPipeline/earthenginedl.py --bounds $BOUNDS --idate "$IDATE" --fdate "$FDATE" --landsat $LANDSAT --grid_key "$KEY" --region "$KEY" --maxims $MAXIMS --scale $SCALE --outpath "$BASE_OUTPATH/$KEY"
  
  # Run saliencymap.py
  python ./DataPipeline/saliencymap.py --dir_path "$BASE_OUTPATH/$KEY" --crs EPSG:4326 --grid_key "$KEY"
  
  # Run saliencymap2boxes.py
  python ./DataPipeline/saliencymap2boxes.py -k "$KEY" -w $BOX_WIDTH -n $BOX_COUNT -p "$BASE_OUTPATH/$KEY/landmarks"
  
  # Run prepare_yolo_data.py with configurable output path
  python ./DataPipeline/prepare_yolo_data.py --data_path "$BASE_OUTPATH/$KEY" --landmark_path $LANDMARK_BASE/$KEY/landmarks --output_path $FINAL_OUTPUT_PATH/${KEY}_dataset --r "$KEY" --val $VALFLAG

  # Train YOLO model
  python ./LD/train_yolo.py --data "$CURR_DIR/LD/datasets/${KEY}_dataset/dataset.yaml" --save_dir "$CURR_DIR/LD/runs" --region "$KEY" --version "yolov8n" --epochs 10

done

# Run prepare_RCnet_dataset.py
python ./RCnet/src/prepare_RCnet_dataset.py --root_dir "$BASE_OUTPATH" --output_dir "$CURR_DIR" --test_ratio 0.1 --val_ratio 0.1
  
# Train RCNet
python ./RCnet/src/main.py --train_flag  --save_plot_flag  --data_dir "$CURR_DIR/RCnet/datasets"  --model_save_path "$CURRDIR/RCnet/model.pth"  --model_load_path "$CURRDIR/RCnet/model.pth"  --save_plot_path "/home/argus/Arvind/GNC-Payload/VisionTrainingGround/RCnet/results/loss_vs_epoch.png"  --learning_rate 0.0001  --epochs 50

