from osgeo import gdal

def open_raster(raster_file):
    ds = gdal.Open(raster_file)
    if ds is None:
        raise ValueError(f"Could not open raster file: {raster_file}")
    return ds

def get_raster_data(ds):
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()

def get_geotransform(ds):
    return ds.GetGeoTransform()

def get_projection(ds):
    return ds.GetProjection()

def geo_to_pixel(geo_x, geo_y, geotransform):
    pixel_x = int((geo_x - geotransform[0]) / geotransform[1])
    pixel_y = int((geo_y - geotransform[3]) / geotransform[5])
    return pixel_x, pixel_y

# Add more functions as needed
