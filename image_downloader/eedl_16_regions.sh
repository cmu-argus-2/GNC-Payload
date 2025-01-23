#!/bin/bash

regions=("10S" "10T" "11R" "12R" "16T" "17R" "17T" "18S" "32S" "32T" "33S" "33T" "52S" "53S" "54S" "54T")
for region in "${regions[@]}"; do
  python3 eedl.py -g "$region" -i 2022 -f 2023-06-15 -s 500 -vb 250000 -hb 250000 -o "${region}" -se l9 -m 30 -c EPSG:4326 -rm True -gd True
done
