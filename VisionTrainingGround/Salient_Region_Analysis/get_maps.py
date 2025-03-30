import os
import cv2
import numpy as np
from multiprocessing import Pool

from utils.earth_utils import get_MGRS_grid

folder = 'bm1k_regions'
outfolder = 'bm1k_maps'
consfolder = 'bm1k_consolidated_maps'
cloudfolder = 'cloud_maps'
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
SAVEMAP = True
CONSOLIDATE = True
CLOUDS = False
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

def save_map(file, subfolder):  
    im = cv2.imread(folder + '/' + subfolder + '/' + file)
    _, saliency_map = saliency.computeSaliency(im)
    saliency_map_int = (saliency_map * 255).astype("uint8")
    cv2.imwrite(os.path.join(outfolder, subfolder, file), saliency_map_int)
    return saliency_map

def consolidate_region_maps(region):
    reg_im = None
    for month in MONTHS:
        if reg_im is None:
            im = cv2.imread(os.path.join(outfolder, 'world_'+ month, region + '.jpg'))
            reg_im = np.zeros(im.shape, dtype=np.float32)
        else:
            reg_im += cv2.imread(os.path.join(outfolder, 'world_'+month, region + '.jpg'))
    reg_im = reg_im / len(MONTHS)
    reg_im = reg_im.astype('uint8')
    cv2.imwrite(os.path.join(consfolder, region + '.jpg'), reg_im)
    print(region, 'done')

def consolidate_clouds():
    cloudmaps = os.listdir(cloudfolder)
    conscloud = None
    for cloudmap in cloudmaps:
        if conscloud is None:
            im = cv2.imread(os.path.join(cloudfolder, cloudmap))
            conscloud = np.zeros(im.shape, dtype=np.float32)
        else:
            conscloud += cv2.imread(os.path.join(cloudfolder, cloudmap))
    conscloud = conscloud / len(cloudmaps)
    conscloud = conscloud.astype('uint8')
    cv2.imwrite('consolidated_cloudmap.jpg', conscloud)

def one_month(month):
    if not os.path.exists(os.path.join(outfolder, 'world_'+month)):
        os.mkdir(os.path.join(outfolder, 'world_'+month))           
    for file in os.listdir(os.path.join(folder, 'world_'+month)):
        save_map(file, 'world_'+month)
    print(month, 'done')



if __name__ == '__main__':
    grid = get_MGRS_grid()
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    if SAVEMAP:
        with Pool(12) as p:
            p.map(one_month, MONTHS)
    if CONSOLIDATE:
        if not os.path.exists(consfolder):
            os.mkdir(consfolder)
        for key, value in grid.items():
            consolidate_region_maps(key)
            print(key, 'done')
    if CLOUDS:
        consolidate_clouds()
        print('cloudmap done')
