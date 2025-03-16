#!/bin/bash
: '
Script to copy the models and csvs from LDnet and RCnet training 
into the respective inference folders. This script is to be run from 
GNC-Payload ie. the repo root
'
# Base directories
MODEL_SOURCE_DIR="VisionTrainingGround/LD/runs"
CSV_SOURCE_DIR="VisionTrainingGround/Landsat_Data"
DEST_DIR="vision_inference/models/ld"

# Process model directories
find "$MODEL_SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r folder; do
    folder_name=$(basename "$folder")  # Get full folder name (e.g., yolov8x_59G_n100)

    # Extract the middle part (***, which is the region name)
    if [[ "$folder_name" =~ yolov8x_(.*)_n100 ]]; then
        region_name="${BASH_REMATCH[1]}"  # Extract region name (e.g., 59G)
        model_src_file="$folder/weights/best.pt"
        csv_src_file="$CSV_SOURCE_DIR/$region_name/landmarks/${region_name}_outboxes.csv"
        dest_folder="$DEST_DIR/$region_name"
        model_dest_file="$dest_folder/${region_name}_nadir.pt"  # Use correct region name
        csv_dest_file="$dest_folder/${region_name}_top_salient.csv"

        # Create destination folder if it doesn't exist
        mkdir -p "$dest_folder"

        # Copy and rename model file
        if [[ -f "$model_src_file" ]]; then
            cp "$model_src_file" "$model_dest_file"
            echo "Copied $model_src_file to $model_dest_file"
        else
            echo "Skipping $folder_name: best.pt not found"
        fi

        # Copy and rename CSV file
        if [[ -f "$csv_src_file" ]]; then
            cp "$csv_src_file" "$csv_dest_file"
            echo "Copied $csv_src_file to $csv_dest_file"
        else
            echo "Skipping $region_name: CSV file not found at $csv_src_file"
        fi
    else
        echo "Skipping $folder_name: Name format does not match yolov8x_***_n100"
    fi
done

# Copy RCnet model
RCNET_SRC="VisionTrainingGround/RCnet/model/model.pth"
RCNET_DEST_DIR="vision_inference/models/rc"
RCNET_DEST_FILE="$RCNET_DEST_DIR/model.pth"

# Create destination folder if it doesn't exist
mkdir -p "$RCNET_DEST_DIR"

# Copy and rename RCnet model file
if [[ -f "$RCNET_SRC" ]]; then
    cp "$RCNET_SRC" "$RCNET_DEST_FILE"
    echo "Copied $RCNET_SRC to $RCNET_DEST_FILE"
else
    echo "Skipping RCnet model: model.pth not found at $RCNET_SRC"
fi
