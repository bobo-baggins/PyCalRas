from osgeo import gdal
import geopandas as gpd


def process_raster(raster_file):
    ds = gdal.Open(raster_file)
    if ds is None:
        raise ValueError(f"Could not open raster file: {raster_file}")

    raster_data = ds.GetRasterBand(1).ReadAsArray()
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    ds = None  # Close the dataset

    return raster_data, geotransform, projection

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

def read_centerline_shapefile(shapefile_path):
    """
    Read a centerline shapefile and return a GeoDataFrame.

    Args:
    shapefile_path (str): Path to the shapefile.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing the centerline data.
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None

