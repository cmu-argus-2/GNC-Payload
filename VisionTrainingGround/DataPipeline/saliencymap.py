"""
GeoTIFF Saliency Map Processing and Merging Tool

Description:
This program processes a set of GeoTIFF images to generate individual saliency maps,
which are then merged into a single comprehensive map. The program first applies image
processing techniques to each GeoTIFF file to compute a saliency map, highlighting 
areas of interest. These saliency maps are then merged, considering their geographical
boundaries, into a single saliency map for analysis. Additionally, the program can 
convert the merged map into a standard image format for easier visualization.

Features:
- Processes GeoTIFF images to compute saliency maps.
- Merges individual saliency maps into a comprehensive map.
- Converts and saves the final merged map in both TIFF and JPG formats.

Usage:
The program is executed from the command line with various options to specify the
directory of GeoTIFF images, whether to skip previously computed saliency maps,
the number of images to process, the coordinate reference system, and the MGRS grid key.

Author(s): Kyle, Eddie
Contact: [Contact Information - Email/Phone]
"""

import os
import rasterio, rasterio.merge
from rasterio.warp import reproject, calculate_default_transform, Resampling
import cv2
import argparse
import numpy as np
from getMGRS import getMGRS
from multiprocessing import Pool,cpu_count
from tqdm.contrib.concurrent import process_map  # For multiprocessing with progress tracking

def save_maps(folder, file, crs):
    """
    Processes a single GeoTIFF file to compute and save its saliency map.
    :param folder: The directory containing the GeoTIFF image.
    :param file: The filename of the GeoTIFF image.
    :param crs: The Coordinate Reference System to use.
    """
    # Initialize the fine-grained saliency detection model
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Open the GeoTIFF file with rasterio
    with rasterio.open(os.path.join(folder, file)) as raster:
        # Read and store raster meta information
        src_crs = raster.crs  # Source Coordinate Reference System
        width = raster.width  # Image width
        height = raster.height  # Image height

        # Read the image data and transpose the dimensions for processing
        im = raster.read().transpose(1, 2, 0)

        # Copy the metadata from the raster for future use
        kwargs = raster.meta.copy()

        # Calculate the transformation to the specified CRS
        transform, width, height = calculate_default_transform(raster.crs, crs, raster.width, raster.height, *raster.bounds)
        
        # Update metadata with new transformation information
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'float32'
        })

        # Extract pixel information and prepare the image for saliency detection
        im = im[:, :, :3]  # Consider only RGB channels
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        edged = cv2.Canny(gray, 30, 150)  # Edge detection

        # Find the minimum area rectangle enclosing the edges
        coords = np.column_stack(np.where(edged > 0))
        center, wh, angle = cv2.minAreaRect(coords)

        # Rotate the image to align the detected rectangle
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(im, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=0)

        # Detect edges in the rotated image
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        rotated_edged = cv2.Canny(rotated_gray, 30, 150)

        # Find the minimum area rectangle in the rotated image
        coords = np.column_stack(np.where(rotated_edged > 0))  
        rect = cv2.minAreaRect(coords)
        cx, cy = rect[0]
        w, h = rect[1]

        # Crop the region of interest from the rotated image
        t = int(cy - h/2 + .05*h)
        b = int(cy + h/2 - .05*h)
        l = int(cx - w/2 + .05*w)
        r = int(cx + w/2 - .05*w)
        cropped = rotated[t:b, l:r]

        # Compute the saliency map of the cropped region
        (success, saliency_map) = saliency.computeSaliency(cropped)

        # Prepare for unrotating the saliency map
        rotated_new = np.zeros(np.shape(rotated)[:2])
        rotated_new[t:b, l:r] = saliency_map

        # Unrotate the saliency map to original orientation
        M2 = cv2.getRotationMatrix2D(center, angle, 1.0)
        unrotated = cv2.warpAffine(rotated_new, M2, (width, height), flags=cv2.INTER_CUBIC, borderMode=0)

        # Create a directory for saliency maps if it does not exist
        if not os.path.exists(os.path.join(folder, 'maps')):
            os.mkdir(os.path.join(folder, 'maps'))

        # Save the saliency map as a GeoTIFF file
        with rasterio.open(os.path.join(folder, 'maps', file), 'w', **kwargs) as dst:
            reproject(
                source=unrotated,
                destination=rasterio.band(dst, 1),
                src_transform=raster.transform,
                src_crs=raster.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest)

def merge_saliency_maps(files, folder, tstnum, bounds):
    """
    Merges multiple saliency maps into a single image.
    :param files: list of file names to be merged
    :param folder: directory path where the files are located
    :param tstnum: number of images to process
    :param bounds: geographical bounds for cropping
    """
    print("Merging saliency maps..")

    # Define the output directory path
    output_path = os.path.join(folder, 'landmarks')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Construct full paths for each file to be merged
    filenames = [os.path.join(folder, 'maps', file) for file in files]
    # Limit the number of files to be processed, if specified
    filenames = filenames[:tstnum]

    # Perform the merging using rasterio's merge function
    out, aff = rasterio.merge.merge(filenames, method='sum')   
    out_count, count_aff = rasterio.merge.merge(filenames, method='count') 
    # Identify where there is at least one non-zero value (valid data)
    gt0 = out_count > 0

    # Normalize the merged data by the count to get the average
    new_out = out.copy()
    new_out[gt0] = out[gt0] / out_count[gt0]

    # Read metadata from the first file to use as a template
    with rasterio.open(filenames[0]) as src:
        meta = src.meta.copy()
        meta.update(height=new_out.shape[1], width=new_out.shape[2], transform=aff, affine=aff)

    # Save the merged image as a new GeoTIFF file
    full_merge_tif_path = os.path.join(output_path, 'full_merge.tif')
    with rasterio.open(full_merge_tif_path, 'w', **meta) as dst:
        dst.write(new_out)
        
    # Calculate the crop window based on provided bounds
    left, bottom, right, top = bounds
    col_start, row_start = ~aff*(left, top)
    col_stop, row_stop = ~aff*(right, bottom)
    window = ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

    # Crop the merged image using the calculated window
    with rasterio.open(full_merge_tif_path) as src:
        arr = src.read(1, window=window)
        meta.update(height=window[0][1] - window[0][0], width=window[1][1] - window[1][0], affine=src.window_transform(window), transform=src.window_transform(window))
    
    # Add a new axis to array to match the expected shape
    arr = arr[np.newaxis, :,:]

    # Save the cropped image as a new TIFF file
    saliencymap_tif_path = os.path.join(output_path, args.grid_key + '_saliencymap.tif')
    with rasterio.open(saliencymap_tif_path, 'w', **meta) as dst:
        dst.write(arr)

    # Convert the array to 8-bit format if necessary and save as JPG
    if arr.dtype != np.uint8:
        arr_normalized = cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        arr_8bit = np.uint8(arr_normalized)
    else:
        arr_8bit = arr

    # Save the processed image as a JPG file for visualization
    arr_2d = arr_8bit.squeeze()
    saliencymap_jpg_path = os.path.join(output_path, args.grid_key + '_saliencymap.jpg')
    cv2.imwrite(saliencymap_jpg_path, arr_2d)

    print("Maps merged.")

def unpack_and_call_save_maps(args):
    return save_maps(*args)

# Main function
if __name__ == '__main__':
    """
    Main function to process GeoTIFF images, compute and merge their saliency maps.
    :param args: Command-line arguments
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str, required=True)
    parser.add_argument('-s', '--skip_save', type=bool, default=False)
    parser.add_argument('-t', '--tstnum', type=int, default=None)
    parser.add_argument('-c', '--crs', type=str, default='EPSG:4326')
    parser.add_argument('-g', '--grid_key', type=str, default='17R')
    args = parser.parse_args()

    # Initialize variables and lists
    # Get file list from argument directory path
    folder = args.dir_path
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('tif')]

    # Create saliency instance
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Get bounds of argument MGRS region
    grid = getMGRS()
    bounds = grid[args.grid_key]

    # Assign remaining argument values
    crs = args.crs
    tstnum = args.tstnum
    skip_save = args.skip_save

    # Reduce file list to tstnum
    files = files[:tstnum]

    # Prepare the arguments for each file
    function_args = [(folder, file, crs) for file in files]

    # Process and save saliency maps
    if not skip_save:
        process_map(unpack_and_call_save_maps, function_args, max_workers=cpu_count(),  desc="Generate Saliency Maps")

    # Merge saliency maps
    merge_saliency_maps(files, folder, tstnum, bounds)
