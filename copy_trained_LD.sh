#!/bin/bash

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
        model_dest_file="$dest_folder/59G_nadir.pt"
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
