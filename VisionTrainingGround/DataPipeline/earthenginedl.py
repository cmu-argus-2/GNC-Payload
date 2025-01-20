import argparse
import ee
import requests
from multiprocessing import Pool,cpu_count
import os
import shutil
from retry import retry
import numpy as np
from getMGRS import getMGRS
from tqdm.contrib.concurrent import process_map  # Import process_map from tqdm


def getRegionFilterFromBounds(bounds, getRect=True):
    """
    Creates a filter for a given geographical rectangle defined by bounds.

    Parameters:
    bounds (list): A list of four elements [left, top, right, bottom] defining the geographical rectangle.
    getRect (bool): A flag to determine whether to return the rectangle geometry along with the region filter.

    Returns:
    ee.Filter: A filter that selects images intersecting with the defined rectangle.
    ee.Geometry.Rectangle (optional): The rectangle geometry, returned if getRect is True.
    """
    left, top, right, bottom = bounds
    rect = ee.Geometry.Rectangle([left,top,right,bottom])
    region_filter = ee.Filter.bounds(rect)
    if getRect:
        return region_filter, rect
    else:
        return region_filter
  
def getDateFilter(i_date, f_date):
    """
    Creates a date filter for selecting images within a specified date range.

    Parameters:
    i_date (str): Initial date of the date range in a format recognizable by the Earth Engine API.
    f_date (str): Final date of the date range in a format recognizable by the Earth Engine API.

    Returns:
    ee.Filter: A date filter for the specified date range.
    """
    date_filter = ee.Filter.date(i_date,opt_end=f_date)
    return date_filter
    
def getCollection(landsat,region_filter,date_filter,bands=['B4','B3','B2'], cloud_cover_max = 50, date_sort=True):
    """
    Retrieves a filtered collection of Landsat images based on the specified parameters.

    Parameters:
    landsat (str): The Landsat satellite version (e.g., '8' for Landsat 8).
    region_filter (ee.Filter): The geographical region filter.
    date_filter (ee.Filter): The date range filter.
    bands (list): A list of band names to include in the collection (default is ['B4', 'B3', 'B2']).
    cloud_cover_max (float): The maximum cloud cover percentage for the images (default is 50).
    date_sort (bool): Flag to sort the collection by acquisition date.

    Returns:
    ee.ImageCollection: A collection of Landsat images filtered by the specified parameters.
    """
    collection_string = 'LANDSAT/LC0' + landsat + '/C02/T1_TOA'
    collection = ee.ImageCollection(collection_string)
    collection = collection.filter(region_filter).filter(date_filter)
    collection = collection.filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    collection = collection.select(bands)
    if date_sort:
        collection = collection.sort('DATE_ACQUIRED')       
    return collection


def getPointsInRegion(region, num_points, seed):
    """
    Selects random points within a specified region, focusing on land areas. Uses MODIS land/water data to filter out water bodies.

    Parameters:
    region (ee.Geometry): The region within which to select points.
    num_points (int): The number of random points to select.
    seed (int): A seed number for the random point generation to ensure reproducibility.

    Returns:
    list: A list of randomly selected geographical points (longitude and latitude) within the specified region.
    """
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    points = land.updateMask(mask).stratifiedSample(region=region, scale = scale,
                                                    classBand = 'LW', numPoints = num_points,
                                                    geometries=True,seed = seed)
    return points.aggregate_array('.geo').getInfo()

def makeRectangle(point):
    """
    Creates a rectangle geometry around a given point.

    Parameters:
    point (dict): A dictionary containing the 'coordinates' key, which holds the longitude and latitude of the point.

    Returns:
    ee.Geometry.Rectangle: A rectangle geometry centered around the given point with a fixed buffer.
    """
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(200000).bounds()
    rect = region
    return rect

@retry(tries=10, delay=1, backoff=2)
def getURL(index):
    """
    Generates a download URL for a satellite image from the Earth Engine image collection.

    Parameters:
    index (int): The index of the image in the Earth Engine image list.

    Returns:
    str: A URL string from which the image can be downloaded.
    """
    image = ee.Image(im_list.get(index)).multiply(650).toByte()
    if args.crs:
        crs = args.crs
    else:
        crs = image.select(0).projection()
    url = image.getDownloadURL({
        'scale':scale,
        'format':out_format,
        'bands':bands,
        'crs':crs})
    #print('URL',index,'done')
    #print(url)
    return url

@retry(tries=10, delay=1, backoff=2)
def downloadURL(index, url):
    """
    Downloads an image from a given URL and saves it to a specified path. This function will retry up to 10 times 
    with increasing delays if the download fails.

    Parameters:
    index (int): The index of the image, used for naming the downloaded file.
    url (str): The URL from which to download the image.

    Notes:
    The file is saved in either the GeoTIFF or PNG format, depending on the 'out_format' variable.
    The file name is constructed using the Landsat version, region name, and index.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = 'l' + args.landsat + '_' + region_name + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    #print('Download',out_name, 'done')

def unpack_and_call_downloadURL(args):
    return downloadURL(*args)

ee.Authenticate() # Force authentication by setting ee.Authenticate(force=True)
ee.Initialize(
    project='ee-vision-pipeline-training'
)

# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bounds', nargs='+', type=int, default=[-84, 24, -78, 32])
parser.add_argument('-g', '--grid_key', type=str)
parser.add_argument('-i', '--idate',type=str)
parser.add_argument('-f', '--fdate',type=str)
parser.add_argument('-s', '--scale', type = float, default = 150.0)
parser.add_argument('-m', '--maxims', type = int, default = 10)
parser.add_argument('-l', '--landsat', choices=['8','9'], type=str, default = '8')    
parser.add_argument('-o', '--outpath', type=str, default = 'landsat_images')   
parser.add_argument('-r', '--region', type=str, default=None)
parser.add_argument('-e', '--format', type=str,default = 'GEOTiff')
parser.add_argument('-mo', '--mosaic',type=bool, default=False)
parser.add_argument('-si', '--startindex',type=int,default=0)
parser.add_argument('-se', '--seed', type=int,default = None)
parser.add_argument('-gm', '--getmonthlies', type=bool, default = False)
parser.add_argument('-c', '--crs', type=str, default = None)
parser.add_argument('-cc', '--cloud_cover_max',type=float, default = 40.0)
parser.add_argument('-ccgt', '--cloud_cover_min', type=float, default = 0.0)
parser.add_argument('-ba','--bands',type=str,nargs='+',default =['B4','B3','B2'])
parser.add_argument('-ll', '--lonlat', type=bool, default=False)

args = parser.parse_args()     

# Assigning parsed arguments to variables
scale = args.scale
max_ims = args.maxims
out_path = args.outpath
out_format = args.format
region_name = args.region


# Adjusting bounds if grid key is provided
if args.grid_key is not None:
       grid = getMGRS()
       left, bottom, right, top = grid[args.grid_key]
       args.bounds = [float(left), float(bottom), float(right), float(top)]
# if args.region is None:
#     args.region = args.grid_key

# Setting seed for random number generation
if args.seed:
    seed = args.seed
else:
    seed = np.random.randint(100000)


bands = ['B4','B3','B2'] # Specifying bands for image collection
region_filter, rect = getRegionFilterFromBounds(args.bounds,getRect=True) # Getting region filter and rectangle from bounds
date_filter = getDateFilter(args.idate, args.fdate) # Getting date filter based on input dates

# Processing image collection based on selected options
# if not mosaic then download landsat images directly
if not args.mosaic:
    collection = getCollection(args.landsat, region_filter, date_filter,
                           bands=bands, cloud_cover_max = args.cloud_cover_max, date_sort=True)
    collection_size = collection.size().getInfo()
    if collection_size < max_ims:
        max_ims = collection_size
    im_list = collection.toList(max_ims)  

# Create a mosaic of images if selected
else:
    im_list = []
    # Process monthly mosaics if selected
    if args.getmonthlies:
        days = ['31','28','31','30','31','30','31','31','30','31','30','31']
        for i in range(1,13):
            if args.seed:
                seed = args.seed
            else:
                seed = np.random.randint(100000)
            idate = args.idate + '-' + str(i) + '-01'
            fdate = args.idate + '-' + str(i) + '-' + days[i-1]
            date_filter = ee.Filter.date(idate,fdate)
            collection = getCollection(args.landsat, region_filter, date_filter,
                                       bands=bands, cloud_cover_max=50, date_sort=False)
            points = getPointsInRegion(rect, max_ims,seed)
            im = collection.mosaic().multiply(255).toByte() 
            for point in points:
                rect = makeRectangle(point)
                rect_im = im.clip(rect)
                im_list.append(rect_im)
                
    # Process single mosaic
    else:
        if args.seed:
            seed = args.seed
        else:
            seed = np.random.randint(100000)
        collection = getCollection(args.landsat, region_filter, date_filter,
                                       bands=bands, cloud_cover_max=args.cloud_cover_max, date_sort=False)
        
        # Counting the number of images in the collection
        collection_size = collection.size().getInfo()
        # Print the number of images
        print("Number of images in the collection:", collection_size)
        points = getPointsInRegion(rect, max_ims,seed)
        im = collection.mosaic().multiply(255).toByte() 
        for point in points:
            rect = makeRectangle(point)
            rect_im = im.clip(rect)
            im_list.append(rect_im)
    im_list = ee.List(im_list)


def main():
    indexes = list(range(args.startindex, args.startindex + max_ims))
    # Use process_map to execute getURL function in parallel with progress bar
    urls = process_map(getURL, indexes, max_workers=cpu_count(), chunksize=1, desc="Generate download URLs")

    # Prepare arguments for downloadURL function
    download_args = [(i, url) for i, url in enumerate(urls)]
    # Use process_map to execute downloadURL function in parallel with progress bar
    process_map(unpack_and_call_downloadURL, download_args, max_workers=cpu_count(), chunksize=1, desc="Download Images")

    
if __name__ == '__main__':
    main()
