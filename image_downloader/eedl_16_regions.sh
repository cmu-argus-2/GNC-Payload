#!/bin/bash

regions=("10S" "10T" "11R" "12R" "16T" "17R" "17T" "18S" "32S" "32T" "33S" "33T" "52S" "53S" "54S" "54T")
for region in "${regions[@]}"; do
  python3 eedl.py \
    --grid_key "$region" \
    --idate 2022 \
    --fdate 2023-06-15 \
    --scale 500 \
    --vertical_buffer 250000 \
    --horizontal_buffer 250000 \
    --outpath "${region}" \
    --sensor l9 \
    --maxims 30 \
    --crs EPSG:4326 \
    --region_mosaic True \
    --gdrive True
done
