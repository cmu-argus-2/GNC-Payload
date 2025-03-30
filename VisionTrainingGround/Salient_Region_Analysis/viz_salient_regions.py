import cv2
import numpy as np
import pandas as pd
from sal_region_selection import select_mgrs_labels

from utils.earth_utils import get_MGRS_grid


def lat_lon_to_pixel(lat, lon, img_width, img_height):
    """
    Converts latitude and longitude to pixel coordinates.
    Assumes the map spans from -180 to 180 degrees longitude and -90 to 90 degrees latitude.
    """
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    
    lon_per_pix = (max_lon - min_lon) / img_width
    lat_per_pix = (max_lat - min_lat) / img_height
    
    x = int((lon - min_lon) / lon_per_pix)
    y = int((max_lat - lat) / lat_per_pix)  # Flip lat since y=0 is at the top
    return x, y

def highlight_regions(image_path, output_path, regions):
    """Highlights specified MGRS regions on a world map."""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid format.")
    img_height, img_width = img.shape[:2]
    
    # Get MGRS grid information
    grid = get_MGRS_grid()
    
    for region in regions:
        if region in grid:
            left, bottom, right, top = grid[region]
            
            # Convert to pixel coordinates
            x1, y1 = lat_lon_to_pixel(top, left, img_width, img_height)
            x2, y2 = lat_lon_to_pixel(bottom, right, img_width, img_height)
            
            # Draw rectangle on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 10)  # Red box
            cv2.putText(img, region, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 1)
        else:
            print(f"Warning: Region {region} not found in MGRS grid.")
    
    # Save and show the highlighted map
    cv2.imwrite(output_path, img)
    # cv2.imshow('Highlighted MGRS Regions', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    regions_to_highlight = select_mgrs_labels('sorted_region_saliencys.csv', n=40)
    highlight_regions('world_saliency.jpg', 'highlighted_map_edited.jpg', regions_to_highlight)
