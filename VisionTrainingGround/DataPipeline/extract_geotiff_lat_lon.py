"""
Extract accurate lat/lon coordinates from GeoTIFF files using their geospatial metadata.

This script reads the geotransform from each GeoTIFF and creates corresponding
lat_lon.npz files with the correct geographic mapping for each image.

Usage:
    python extract_geotiff_lat_lon.py \
        --tif_dir /path/to/geotiffs \
        --output_dir /path/to/output
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.warp import transform_geom
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    exit(1)


def extract_lat_lon_from_geotiff(tif_path):
    """
    Extract lat/lon coordinates for each pixel in a GeoTIFF using its geotransform.
    
    Args:
        tif_path: Path to the GeoTIFF file
    
    Returns:
        lat_lon_array: Array of shape (H, W, 2) with [lat, lon] for each pixel
        or None if extraction fails
    """
    try:
        with rasterio.open(tif_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            
            # Create arrays of pixel indices
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert pixel coordinates to geographic coordinates
            # The transform maps (col, row) -> (x, y) in the CRS units
            xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
            xs = np.array(xs).reshape(height, width)
            ys = np.array(ys).reshape(height, width)
            
            # If CRS is not WGS84 (EPSG:4326), transform to lat/lon
            if crs and crs.to_epsg() != 4326:
                # Transform to WGS84
                from rasterio.warp import transform as rio_transform
                
                lons_flat, lats_flat = rio_transform(
                    crs, 'EPSG:4326',
                    xs.flatten(), ys.flatten()
                )
                lons = np.array(lons_flat).reshape(height, width)
                lats = np.array(lats_flat).reshape(height, width)
            else:
                # Already in lat/lon
                lons = xs
                lats = ys
            
            # Stack into (H, W, 2) array: [lat, lon]
            lat_lon_array = np.stack([lats, lons], axis=2).astype(np.float32)
            
            return lat_lon_array
            
    except Exception as e:
        print(f"  ERROR extracting from {os.path.basename(tif_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract lat/lon coordinates from GeoTIFF metadata"
    )
    parser.add_argument(
        "--tif_dir",
        type=str,
        required=True,
        help="Directory containing GeoTIFF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save lat_lon.npz files (default: same as tif_dir)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lat_lon.npz files"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.tif_dir
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all GeoTIFF files
    tif_files = []
    for filename in os.listdir(args.tif_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            tif_files.append(filename)
    
    if len(tif_files) == 0:
        print(f"No GeoTIFF files found in {args.tif_dir}")
        return
    
    print(f"Found {len(tif_files)} GeoTIFF files")
    
    # Process each file
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for filename in tqdm(tif_files, desc="Extracting lat/lon"):
        base_name = os.path.splitext(filename)[0]
        tif_path = os.path.join(args.tif_dir, filename)
        output_path = os.path.join(args.output_dir, f"{base_name}_lat_lon.npz")
        
        # Skip if already exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            skip_count += 1
            continue
        
        # Extract lat/lon from GeoTIFF
        lat_lon_array = extract_lat_lon_from_geotiff(tif_path)
        
        if lat_lon_array is not None:
            # Save as compressed npz
            np.savez_compressed(output_path, lat_lon=lat_lon_array)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nResults:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Failed: {fail_count}")


if __name__ == "__main__":
    main()
