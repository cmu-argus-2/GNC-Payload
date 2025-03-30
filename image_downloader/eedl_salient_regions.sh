#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
regions=("19J" "15V" "15V" "37Q" "37Q" "42R" "42R" "18Q" "18Q" "29Q" "29Q" "30U" "30U" "32S" "32S" "32T" "32T" "33K" "33K" "33S" "33S" "33T" "33T" "35J" "35J" "36L" "36L" "38K" "38K" "39P" "39P" "40R" "40R" "46Q" "46Q" "50M" "50M" "51J" "51J" "52S" "52S" "53L" "53L" "54S" "54S" "54U" "54U" "55J" "55J" "59G" "59G" "57V" "57V" "57V" "48M" "48M" "48M" "48M" "48M" "05V" "05V" "05V" "05V" "05V" "05V" "05V" "49S" "49S" "49S" "49S" "49S" "49S" "49S" "09V" "09V" "09V" "09V" "09V" "09V" "09V" "09V" "09V")
for region in "${regions[@]}"; do
  # TODO: add cloud cover filter
  python3 "$SCRIPT_DIR/eedl.py" \
    --grid_key "$region" \
    --idate 2022 \
    --fdate 2023-06-15 \
    --scale 500 \
    --vertical_buffer 250000 \
    --horizontal_buffer 250000 \
    --outpath "${region}" \
    --sensor l9 \
    --maxims 2 \
    --crs EPSG:4326 \
    --region_mosaic True \
    --gdrive True
done
