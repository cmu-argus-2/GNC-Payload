"""
GeoTIFF Saliency Map Box Processing Tool

Description:
This script processes GeoTIFF saliency maps to identify and annotate areas of high saliency. 
It reads TIFF files representing saliency maps, scans these maps with a specified window size, 
and identifies the regions with the highest saliency. The top regions are then marked with 
bounding boxes. These boxed regions' geographical coordinates are determined and saved in 
both NumPy (.npy) and CSV formats for further analysis. Additionally, the script generates 
images with highlighted regions for visual inspection.

Features:
- Reads saliency maps from GeoTIFF files.
- Identifies and annotates areas of high saliency with bounding boxes.
- Converts pixel coordinates of the boxes to geographic coordinates.
- Saves the coordinates in NumPy and CSV formats.
- Generates images with highlighted regions.

Usage:
Run the script from the command line with options to specify the keys for processing, 
window size for scanning, the number of boxes to identify, and the path to the saliency maps.

Author(s): Kyle, Eddie
Contact: [Contact Information - Email/Phone]
"""

import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
from getMGRS import getMGRS # custom getMGRS module for MGRS grid handling
from multiprocessing import Pool, cpu_count
import csv
import rasterio
from tqdm.contrib.concurrent import process_map  # For multiprocessing with progress tracking

def sm2b(key, input_path, window, num_boxes, box_color, box_outline):
    # Read the saliency map from TIFF file
    tiff_file = input_path + '/' + key + '_saliencymap.tif'

    with rasterio.open(tiff_file) as dataset:
        saliency_map = dataset.read(1)  # Assuming saliency data is in the first band
        im_h, im_w = saliency_map.shape

        # Print window size and number of boxes
        print("Window Size:", window)
        print("Number of Boxes:", num_boxes)

        saliency_data = []

        # Scan the saliency map with a window-sized box
        for y in range(0, im_h - window + 1, window):
            for x in range(0, im_w - window + 1, window):
                # Calculate the sum of saliency in this box
                roi = saliency_map[y:y + window, x:x + window]
                sum_saliency = np.sum(roi)
                saliency_data.append((sum_saliency, (x, y, window, window)))

        # Sort the boxes by sum of saliency (descending)
        saliency_data.sort(key=lambda x: x[0], reverse=True)

        # Select the top num_boxes boxes
        boxes = [box for _, box in saliency_data[:num_boxes]]

        # Draw boxes on the saliency map image
        saliency_map_image = cv2.imread(input_path + '/' + key + '_saliencymap.jpg')
        for x, y, w, h in boxes:
            cv2.rectangle(saliency_map_image, (x, y), (x + w, y + h), box_color, box_outline)
        
        cv2.imwrite(input_path + '/' + key + '_boxes.jpg', saliency_map_image)

        # Convert box coordinates to geographic coordinates
        outboxes = []
        for x, y, w, h in boxes:
            tl_lon, tl_lat = dataset.xy(y, x)  # Top-left
            br_lon, br_lat = dataset.xy(y + h, x + w)  # Bottom-right
            # Calculate centroid coordinates
            centroid_lon = (tl_lon + br_lon) / 2
            centroid_lat = (tl_lat + br_lat) / 2
            outboxes.append([centroid_lon, centroid_lat, tl_lon, tl_lat, br_lon, br_lat])

        # Save as .npy
        outboxes = np.array(outboxes)
        np.save(input_path + '/' + key + '_outboxes.npy', outboxes)

        # Save as .csv
        csv_file_path = input_path + '/' + key + '_outboxes.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write header (optional, but recommended for clarity)
            csv_writer.writerow(['Centroid Longitude', 'Centroid Latitude', 'Top-Left Longitude', 'Top-Left Latitude', 'Bottom-Right Longitude', 'Bottom-Right Latitude'])
            # Write data
            for box in outboxes:
                csv_writer.writerow(box)

def unpack_and_call_sm2b(args):
    return sm2b(*args)
    
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys', nargs = '+', required=True)
    parser.add_argument('-w', '--window', default = 50, type=int, help = 'pixel width of box around each POI')
    # parser.add_argument('-s', '--save', default = True, type=bool, help = 'save saliency map')
    parser.add_argument('-n', '--num_boxes', default = 50, type = int)
    parser.add_argument('-p', '--path', default = '.')
    args = parser.parse_args()

    # Initialize variables and lists
    keys = zip(args.keys)
    input_path = args.path
    grid = getMGRS()
    window = args.window
    num_boxes = args.num_boxes
    box_color = (0, 0, 255)  # Color of the box in RGB (blue in this example)
    box_outline = 3  # Thickness of the box outline

    # Prepare the arguments for the function
    function_args = [(key, input_path, window, num_boxes, box_color, box_outline) for key in args.keys]

    # Use process_map
    process_map(unpack_and_call_sm2b, function_args, max_workers=cpu_count(), desc="Create Landmark Bounding Boxes:")
