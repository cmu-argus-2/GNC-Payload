import os
import cv2
import numpy as np
from multiprocessing import Pool
from getMGRS import getMGRS

folder = 'bm1k'
outfolder = 'bm1k_regions'
grid = getMGRS()
min_lat = -90
max_lat = 90
min_lon = -180
max_lon = 180
lon_per_pix = (max_lon - min_lon) / 21600
lat_per_pix = (max_lat - min_lat) / 10800
regions_folder = 'bm1k_consolidated_maps'

SAVE_REGIONS = True
RECOMBINE = False

def save_regions(file):
    im = cv2.imread(os.path.join(folder, file))
    for key, value in grid.items():
        left, bottom, right, top = value
        left = left - min_lon
        right = right - min_lon
        bottom = 180 - (bottom - min_lat)
        top = 180 - (top - min_lat)
        left_px = left/lon_per_pix
        right_px = right/lon_per_pix
        top_px = bottom/lat_per_pix
        bottom_px = top/lat_per_pix
        print(bottom_px, top_px, left_px, right_px)
        reg_im = im[int(bottom_px):int(top_px), int(left_px):int(right_px)]
        cv2.imwrite(os.path.join(outfolder, file[:-4], key + '.jpg'), reg_im)
        print(file, key, 'done')
    print(file, 'done')

def recombine_regions():
    out_im = np.zeros((10800, 21600), dtype=np.uint8)
    for file in os.listdir(regions_folder):
        im = cv2.imread(os.path.join(regions_folder, file))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        reg = file[:-4]
        left, bottom, right, top = grid[reg]
        left = left - min_lon
        right = right - min_lon
        bottom = 180 - (bottom - min_lat)
        top = 180 - (top - min_lat)
        left_px = left/lon_per_pix
        right_px = right/lon_per_pix
        top_px = bottom/lat_per_pix
        bottom_px = top/lat_per_pix
        out_im[int(bottom_px):int(top_px), int(left_px):int(right_px)] = im
        print(file, 'done')
    cv2.imwrite('world_saliency.jpg', out_im)


if __name__ == '__main__':
    if SAVE_REGIONS:
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        months = os.listdir(folder)
        for month in months:
            if not os.path.exists(os.path.join(outfolder, month[:-4])):
                os.mkdir(os.path.join(outfolder, month[:-4 ]))
        with Pool(3) as p:
            p.map(save_regions, months)
    if RECOMBINE:
        recombine_regions()