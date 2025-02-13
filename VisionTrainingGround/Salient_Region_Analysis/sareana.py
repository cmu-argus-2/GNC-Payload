import cv2
import numpy as np
import csv
from getMGRS import getMGRS

min_lat = -90
max_lat = 90
min_lon = -180
max_lon = 180
lon_per_pix = (max_lon - min_lon) / 21600
lat_per_pix = (max_lat - min_lat) / 10800

def sareana(cloud_im_path, saliency_im_path, grid, use_saliency_only=False):
    if not use_saliency_only:
        cloud_im = cv2.imread(cloud_im_path)
        cloud_im = ~cv2.cvtColor(cloud_im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('inverse_cloud_map.jpg', cloud_im)
        cloud_im = cloud_im.astype('float32')
        cloud_im = cloud_im / cloud_im.max()
    
    saliency_im = cv2.imread(saliency_im_path)
    saliency_im = cv2.cvtColor(saliency_im, cv2.COLOR_BGR2GRAY)
    saliency_im = saliency_im.astype('float32')
    saliency_im = saliency_im / saliency_im.max()
    
    im_height, im_width = saliency_im.shape[:2]
    
    if not use_saliency_only:
        cloud_im_resized = cv2.resize(cloud_im, (im_width, im_height))
        sareana_im = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        sareana_im[:, :, 0] = (255 * cloud_im_resized.copy()).astype('uint8')
        sareana_im[:, :, 1] = (255 * saliency_im.copy()).astype('uint8')
        cv2.imwrite('sareana.jpg', sareana_im)
        sareana_mul = cloud_im_resized * saliency_im
    else:
        sareana_im = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        sareana_im[:, :, 1] = (255 * saliency_im.copy()).astype('uint8')
        cv2.imwrite('saliency_only.jpg', sareana_im)
        sareana_mul = saliency_im

    sareana_mul = sareana_mul / sareana_mul.max()
    sareana_mul_im = (sareana_mul.copy() * 255).astype('uint8')
    sareana_im[:, :, 2] = sareana_mul_im
    cv2.imwrite('sareana_mul.jpg', sareana_im)
    cv2.imwrite('sareana_only_mul.jpg', sareana_mul_im)

    reg_sareana = sareana_mul.copy()
    reg_sals = {}
    
    for key, value in grid.items():
        left, bottom, right, top = value
        left = left - min_lon
        right = right - min_lon
        bottom = 180 - (bottom - min_lat)
        top = 180 - (top - min_lat)
        left_px = int(left / lon_per_pix)
        right_px = int(right / lon_per_pix)
        top_px = int(bottom / lat_per_pix)
        bottom_px = int(top / lat_per_pix)
        region_im = sareana_mul[bottom_px:top_px, left_px:right_px]
        
        if key[-1] == 'X' or key[-1] == 'W' or key[-1] == 'C' or key[-1] == 'D':
            region_sal = 0
        else:
            region_sal = region_im.sum() / (region_im.shape[0] * region_im.shape[1])
        
        reg_sals[key] = region_sal
        reg_sareana[bottom_px:top_px, left_px:right_px] = region_sal
    
    reg_sareana = reg_sareana / reg_sareana.max() * 255
    reg_sareana = reg_sareana.astype('uint8')
    cv2.imwrite('reg_sareana.jpg', reg_sareana)
    
    sorted_reg_sals = sorted(reg_sals.items(), key=lambda x: x[1], reverse=True)
    
    with open('prioritized_regions.csv', 'w') as f:
        for key, value in sorted_reg_sals:
            if value == 0:
                break
            f.write(key + ',')
    
    return sorted_reg_sals


if __name__ == '__main__':
    grid = getMGRS()
    use_saliency_only = True
    
    # Set use_saliency_only to True if you want to use only saliency map data
    sorted_reg_sals = sareana('world_saliency.jpg', 'world_saliency.jpg', grid, use_saliency_only)
    
    print(sorted_reg_sals)
    np_file_name = 'sorted_region_saliencys_no_cloud.npy' if use_saliency_only else 'sorted_region_saliencys.npy'
    np.save(np_file_name, sorted_reg_sals)
    
    with open(np_file_name[:-4]+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Value"])  # Header
        writer.writerows(sorted_reg_sals)  # Write data
