#!/bin/bash

# Default values for configurations
BASE_OUTPATH="Landsat_Data"
CURR_DIR="$(pwd)/VisionTrainingGround" # You need to be in the Vision Training Ground folder
LANDMARK_BASE="$CURR_DIR/Landsat_Data"  # Path relative to the current directory
FINAL_OUTPUT_PATH="$CURR_DIR/LD/datasets"  # Path relative to the current directory

echo "Current Directory: $CURR_DIR"
echo "Base Output Path: $BASE_OUTPATH"
echo "Landmark Base Path: $LANDMARK_BASE"
echo "Final Output Path: $FINAL_OUTPUT_PATH"

# Prepare RCNet dataset
# python ./RCnet/src/prepare_RCnet_dataset.py --root_dir "$BASE_OUTPATH" --output_dir "$CURR_DIR/RCnet/datasets" --test_ratio 0.1 --val_ratio 0.1
  
# Train RCNet
# python ./RCnet/src/main.py  --train_flag --save_plot_flag --data_dir "/mnt/sdb2/training2" --non_salient_data_dir "/mnt/sda2/training_non_salient/non-salient" --model_save_path "$CURR_DIR/RCnet/model/model.pth" --model_load_path "$CURR_DIR/RCnet/model/model.pth" --save_plot_path "$CURR_DIR/RCnet/results/loss_vs_epoch.png" --learning_rate 0.0001 --epochs 10

# Training
# python $CURR_DIR/RCnet/src/main.py  --train_flag --save_plot_flag --data_dir "/mnt/sdb2/training2" --non_salient_data_dir "/mnt/sda2/training_non_salient/non-salient" --model_save_path "$CURR_DIR/RCnet/model/model.pth" --save_plot_path "$CURR_DIR/RCnet/results/loss_vs_epoch.png" --learning_rate 0.0001 --epochs 10

# Testing
python $CURR_DIR/RCnet/src/main.py --data_dir "/mnt/sdb2/training2" --model_load_path "$(pwd)/RCnet/chkpts/model1.pth" 
