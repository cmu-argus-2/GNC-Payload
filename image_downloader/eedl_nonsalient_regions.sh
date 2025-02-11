#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
excluded_regions=("10S" "10T" "11R" "12R" "16T" "17R" "17T" "18S" "32S" "32T" "33S" "33T" "52S" "53S" "54S" "54T")

# Iterate over all MGRS regions
for zone in {1..60}; do
  for band in {C..X}; do
    if [[ "$band" == "I" || "$band" == "O" ]]; then
      # These bands are skipped over in the MGRS system
      continue
    fi

    region="${zone}${band}"
    if [[ " ${excluded_regions[@]} " =~ " ${region} " ]]; then
      continue
    fi

    python3 "$SCRIPT_DIR/eedl.py" \
      --grid_key "$region" \
      --idate 2022 \
      --fdate 2023-06-15 \
      --scale 500 \
      --vertical_buffer 250000 \
      --horizontal_buffer 250000 \
      --outpath "${region}" \
      --sensor l9 \
      --maxims 1 \
      --crs EPSG:4326 \
      --region_mosaic True \
      --gdrive True
  done
done
