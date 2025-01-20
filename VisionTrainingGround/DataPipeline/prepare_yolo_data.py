import numpy as np
import glob
import csv
import os
import warnings
import pyproj
import rasterio as rs
from rasterio.windows import Window
from PIL import Image
import imageio
import argparse
import cv2
import yaml
from tqdm import tqdm  # Import tqdm at the beginning of your script
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # Import process_map

warnings.filterwarnings('ignore')

# check if box UL and BR corners are in bounds of image
# Note: this is in whole image, including blacked out regions
def check_bounds(long_lat_coords, src):
    pp = pyproj.Proj(init=src.crs)

    xmin, ymax = pp(long_lat_coords[0][0], long_lat_coords[0][1])
    xmax, ymin = pp(long_lat_coords[1][0], long_lat_coords[1][1])

    bounds = src.bounds
    if (bounds.left < xmin < bounds.right and bounds.bottom < ymax < bounds.top
        and bounds.left < xmax < bounds.right and bounds.bottom < ymin < bounds.top):
        if check_box_vals(long_lat_coords, src):
            return True
    return False

# convert long lat coords to pixel xy
def longlat_to_xy(long_lat_coords, src):
    meta = src.meta

    xy_coords = []
    for coord in long_lat_coords:
        pp = pyproj.Proj(init=src.crs)

        lon, lat = pp(coord[0], coord[1])

        py, px = src.index(lon, lat)
        xy_coords.append([px, py])

    return xy_coords

# check if box lies in no-data region of image (invalid)
def check_box_vals(long_lat_coords, src):
    xy_coords = longlat_to_xy(long_lat_coords, src)

    xsize = xy_coords[1][0] - xy_coords[0][0]
    ysize = xy_coords[1][1] - xy_coords[0][1]
    window = Window(xy_coords[0][0], xy_coords[0][1], xsize, ysize)
    arr = src.read(window=window)
    arr_sum = np.sum(arr, axis=0)
    zero = np.count_nonzero(arr_sum == 0) / (xsize*ysize)

    if zero >= 0.5:
      return False
    else:
      return True

# convert tif images to png and save
def convert_tif_to_png(tif_path, png_path):
    with rs.open(tif_path) as src:
        img = src.read()  # Read the image (bands, rows, columns)
        
        # Normalize and scale floating-point images to 8-bit
        if np.issubdtype(img.dtype, np.floating):
            # Normalize the image to 0-1
            img -= img.min()
            if img.max() != 0:
                img /= img.max()
            # Scale to 0-255 and convert to uint8
            img = (255*img).astype(np.uint8)
        
        # Handle single-band (grayscale) images
        if img.shape[0] == 1:
            img_squeezed = np.squeeze(img, axis=0)
        else:
            img_squeezed = img.transpose((1, 2, 0))  # Reorder dimensions for multi-band images
            
        imageio.imwrite(png_path, img_squeezed)

# get dict of jpg img dimensions
def get_img_sizes(save_path):
    min_width = 100000
    min_height = 100000
    im_sizes = {}

    # List of patterns to match both jpg and png files
    patterns = ['*.jpg', '*.png']

    for pattern in patterns:
        for file in glob.glob(os.path.join(save_path, pattern)):
            img_name = os.path.basename(file).split('.')[0]
            im = Image.open(file)
            size = im.size

            if size[0] < min_width:
                min_width = size[0]
            if size[1] < min_height:
                min_height = size[1]

            im_sizes.update({img_name: size})

    #print("min dims: ", min_width, min_height)
    #print(im_sizes)
    return im_sizes

def rotate(raster, bboxes):
    im = raster.read().transpose(1, 2, 0)
    width = raster.width  # Image width
    height = raster.height  # Image height

    im = im[:, :, :3]  # Consider only RGB channels

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    edged = cv2.Canny(gray, 30, 150)  # Edge detection

    # Find the minimum area rectangle enclosing the edges
    coords = np.column_stack(np.where(edged > 0))
    center, wh, angle = cv2.minAreaRect(coords)

    # Rotate the image to align the detected rectangle
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rot_im = cv2.warpAffine(im, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=0)

    rot_boxes = {}
    for idx, box in bboxes.items():
        box_points = []
        xmin, ymin = box[0]
        xmax, ymax = box[1]

        # get box centerpoint
        cx = (xmax - xmin) / 2 + xmin
        cy = (ymax - ymin) / 2 + ymin
        center = list([int(cx), int(cy)])
        box_points.append(center)
        # get coordinates of other box corners
        other_corners = [[xmin, ymax], [xmax, ymin]]

        box_points += box
        box_points += other_corners
        # form array of box points (4 corners + center point)
        box_points = np.array(box_points).reshape((5, 2))

        # apply rotation on box points
        box_points = np.vstack([box_points.T, np.ones(5)])
        rot_box = np.dot(M, box_points).T

        # recalculate UL and BR corners using rotated centerpoint and largest bounding box
        cx, cy = rot_box[0]    
        corners = rot_box[1:, :]
        xmin = int(np.min(corners[:, 0]))
        xmax = int(np.max(corners[:, 0]))
        ymin = int(np.min(corners[:, 1]))
        ymax = int(np.max(corners[:, 1]))
        w = abs(xmax-xmin)
        h = abs(ymax-ymin)
        ul_corner = [int(cx-w/2), int(cy-h/2)]
        br_corner = [int(cx+w/2), int(cy+h/2)]

        rot_boxes.update({idx:[ul_corner, br_corner]})

    rot_im = cv2.cvtColor(rot_im, cv2.COLOR_BGR2RGB)
  
    return rot_im, rot_boxes


# Function to process a single image
def process_single_image(file, data_path, all_landmarks, region, val_set):
    
    detected_boxes = {}
    img_name = os.path.basename(file).split('.')[0]
    detected_boxes[img_name] = {}

    src = rs.open(file)
    for idx, box in all_landmarks.items():
        if check_bounds(box, src):
            xy_coords = longlat_to_xy(box, src)
            detected_boxes[img_name][idx] = xy_coords

    # IF VALIDATION DATA (rotate image and boxes)
    if val_set is True:
        rot_im, rot_boxes = rotate(src, detected_boxes[img_name])
        detected_boxes[img_name] = rot_boxes
        out_path = os.path.join(data_path, img_name + '.jpg')
        cv2.imwrite(out_path, rot_im)
    else:
        out_path = os.path.join(data_path, img_name + '.png')
        convert_tif_to_png(file, out_path)

    return detected_boxes

# Function to generate a single label file
# Adjustments to the generate_single_label function to match the argument signature
def generate_single_label(img_name, detected_boxes, label_path, im_size):
    label_file = os.path.join(label_path, img_name + '.txt')
    #if img_name in detected_boxes and len(detected_boxes[img_name]) > 0:
    with open(label_file, 'w') as f:
        for cls, b in detected_boxes[img_name].items():
            xmin, ymin = b[0]
            xmax, ymax = b[1]
            cx = (xmax + xmin) / 2
            cy = (ymax + ymin) / 2
            width = xmax - xmin
            height = ymax - ymin
            cx_norm = cx / im_size[0]
            cy_norm = cy / im_size[1]
            width_norm = width / im_size[0]
            height_norm = height / im_size[1]
            label = f"{cls} {cx_norm} {cy_norm} {width_norm} {height_norm}\n"
            f.write(label)

def generate_dataset_yaml(output_path, nc):
    # Extract the last part of the output_path to use as 'path' in dataset.yaml
    dataset_dir_name = os.path.basename(output_path)
    dataset_yaml = {
        'path': dataset_dir_name,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': nc,
        'names': [str(i) for i in range(nc)]
    }

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(dataset_yaml, yaml_file, default_flow_style=False)
    print(f"Dataset configuration saved to {yaml_path}")

def process_single_image_wrapper(args):
    return process_single_image(*args)

def generate_single_label_wrapper(args):
    return generate_single_label(*args)

def process_data_with_pool(args):
    val_set = args.val
    data_path = args.data_path
    landmark_path = args.landmark_path
    output_path = args.output_path
    region = args.region
    anno_file = os.path.join(landmark_path, f"{region}_outboxes.csv")
    
    # Make sure the images output directory exists
    if val_set is True:
        save_path = os.path.join(output_path, "val/images")
    else:
        save_path = os.path.join(output_path, "train/images")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Make sure the label output directory exists
    if val_set is True:
        label_path = os.path.join(args.output_path, 'val/labels')
    else:
        label_path = os.path.join(args.output_path, 'train/labels')
    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    print("save_path:", save_path)
    print("label_path:", label_path)

    # Load landmarks
    all_landmarks = {}
    with open(anno_file, 'r') as f:
        csvReader = csv.DictReader(f)
        for i, row in enumerate(csvReader):
            min_long = float(row["Top-Left Longitude"])
            max_lat = float(row["Top-Left Latitude"])
            max_long = float(row["Bottom-Right Longitude"])
            min_lat = float(row["Bottom-Right Latitude"])
            all_landmarks[i] = [[min_long, max_lat], [max_long, min_lat]]

    files = glob.glob(os.path.join(data_path, '*.tif'))

    # Use process_map to process images with progress tracking
    process_args = [(file, save_path, all_landmarks, region, val_set) for file in files]
    detected_boxes_list = process_map(process_single_image_wrapper, process_args, chunksize=1, max_workers=None, desc="Processing Images")

    # Combine detected_boxes from all images
    detected_boxes = {}
    for boxes in detected_boxes_list:
        detected_boxes.update(boxes)

    # Get dimensions of images in pixels
    im_sizes = get_img_sizes(save_path)

    # Use process_map to generate label files with progress tracking
    generate_args = [(img_name, detected_boxes, label_path, im_sizes[img_name]) for img_name in detected_boxes]
    process_map(generate_single_label_wrapper, generate_args, chunksize=1, max_workers=None, desc="Generating Label Files")

    # Calculate nc as the total number of unique classes
    nc = len(all_landmarks)

    # Generate dataset.yaml after processing tarin images and labels
    if val_set is not True:
        generate_dataset_yaml(args.output_path, nc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates jpgs and labels in yolo format")
    parser.add_argument("--data_path", required=True, help="path to original tiff images")
    parser.add_argument("--landmark_path", required=True, help="path to landmark annotation file")
    parser.add_argument("--output_path", required=True, help="output folder path for images and labels")
    parser.add_argument("--val", type=bool, help="Flag for creating rotated Landsat validation dataset")
    parser.add_argument('-r', '--region', type=str, default='17R')
    args = parser.parse_args()

    process_data_with_pool(args)
    