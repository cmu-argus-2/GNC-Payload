#!/bin/bash

regions="05V 09V 10S 10T 11R 12R 14Q 15V 16T 18Q 18S 19J 21H 23L 29Q 30U 32S 32T 33K
33S 33T 35J 36L 37Q 38K 39P 40R 42R 46Q 48M 49S 50M 51J 52S 53L 54S 54U 55J 57V 59G"
echo $regions

python run_saliency_analysis.py \
  --regions $regions \
  --overwrite \
  --gsd 50.0 \
  --bounding_box_size 7200 \
  --num_boxes 50

python prepare_yolo_data.py \
  --regions $regions \
  --overwrite \
  --test_fraction 0.2 \
  --val_fraction 0.1

python train_yolo.py \
  --regions $regions \
  --overwrite \
  --version yolov8n \
  --epochs 300
