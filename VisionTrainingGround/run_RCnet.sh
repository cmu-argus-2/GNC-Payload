#!/bin/bash

# Default values for configurations
BASE_OUTPATH="Landsat_Data"
CURR_DIR="$(pwd)" # You need to be in the Vision Training Ground folder
LANDMARK_BASE="$(pwd)/Landsat_Data"  # Path relative to the current directory
FINAL_OUTPUT_PATH="$(pwd)/LD/datasets"  # Path relative to the current directory

# Prepare RCNet dataset
python ./RCnet/src/prepare_RCnet_dataset.py --root_dir "$BASE_OUTPATH" --output_dir "$CURR_DIR/RCnet/datasets" --test_ratio 0.1 --val_ratio 0.1
  
# Train RCNet
python ./RCnet/src/main.py --train_flag --save_plot_flag --data_dir "$CURR_DIR/RCnet/datasets" --model_save_path "$CURR_DIR/RCnet/model.pth" --model_load_path "$CURR_DIR/RCnet/model.pth" --save_plot_path "$CURR_DIR/RCnet/results/loss_vs_epoch.png" --learning_rate 0.0001 --epochs 50
