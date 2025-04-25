from osgeo import gdal
import geopandas as gpd
import numpy as np
from osgeo import osr
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_raster(raster_file: str) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    """
    Process a raster file and return its data, geotransform, and projection.
    
    Args:
        raster_file: Path to the raster file
        
    Returns:
        Tuple containing:
        - raster_data: The raster data array
        - geotransform: The geotransform parameters
        - projection: The projection in WKT format
        
    Raises:
        ValueError: If the raster file cannot be opened
        RuntimeError: If there are issues reading the raster data
    """
    try:
        ds = gdal.Open(raster_file)
        if ds is None:
            raise ValueError(f"Could not open raster file: {raster_file}")
            
        # Get raster data
        band = ds.GetRasterBand(1)
        if band is None:
            raise RuntimeError("Could not get raster band")
            
        raster_data = band.ReadAsArray()
        if raster_data is None:
            raise RuntimeError("Could not read raster data")
            
        # Get geotransform and projection
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # Clean up
        band = None
        ds = None
        
        return raster_data, geotransform, projection
        
    except Exception as e:
        logger.error(f"Error processing raster file {raster_file}: {str(e)}")
        raise

# Make sure other necessary functions are also defined here

def geo_to_pixel(geo_x, geo_y, geotransform):
    """
    Convert geographic coordinates to pixel coordinates.

    Args:
    geo_x (float): Geographic X-coordinate.
    geo_y (float): Geographic Y-coordinate.
    geotransform (tuple): Geotransform information from the raster.

    Returns:
    tuple: Pixel coordinates (pixel_x, pixel_y).
    """
    pixel_x = int((geo_x - geotransform[0]) / geotransform[1])
    pixel_y = int((geo_y - geotransform[3]) / geotransform[5])
    return pixel_x, pixel_y

# Add more functions as needed

def read_centerline_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Read a centerline shapefile and return a GeoDataFrame.
    
    Args:
        shapefile_path: Path to the shapefile
        
    Returns:
        GeoDataFrame containing the centerline data
        
    Raises:
        ValueError: If the shapefile cannot be read
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            raise ValueError("Shapefile is empty")
        return gdf
    except Exception as e:
        logger.error(f"Error reading shapefile {shapefile_path}: {str(e)}")
        raise

def convert_dat_to_raster(dat_file, raster_file, nrows, ncols, geotransform, projection):
    """
    Convert a .dat file to a raster file.

    Args:
    dat_file (str): Path to the input .dat file.
    raster_file (str): Path to the output raster file.
    nrows (int): Number of rows in the raster.
    ncols (int): Number of columns in the raster.
    geotransform (tuple): Geotransform parameters (top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution).
    projection (str): Projection in WKT format.

    Returns:
    None
    """
    # Read the .dat file
    data = np.loadtxt(dat_file)  # Adjust this if your .dat file has a different format

    # Reshape the data to the specified number of rows and columns
    if data.size != nrows * ncols:
        raise ValueError("Data size does not match specified dimensions.")
    
    data = data.reshape((nrows, ncols))

    # Create a new raster dataset
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_file, ncols, nrows, 1, gdal.GDT_Float32)

    # Set the geotransform and projection
    dataset.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dataset.SetProjection(srs.ExportToWkt())

    # Write the data to the raster band
    dataset.GetRasterBand(1).WriteArray(data)

    # Flush the cache and close the dataset
    dataset.FlushCache()
    dataset = None

    print(f"Raster file created: {raster_file}")

