import pandas as pd
import numpy as np
import GeoDataPro as geo
import geopandas as gpd

def sample_point(raster_data, geotransform, points_df, x_col='X', y_col='Y', z_col='Z'):
    """
    Sample raster values at given points and calculate differences.
    
    Args:
    raster_data (numpy.ndarray): The raster data.
    geotransform (tuple): The geotransform of the raster.
    points_df (pandas.DataFrame): DataFrame containing point coordinates.
    x_col (str): Name of the column containing X coordinates.
    y_col (str): Name of the column containing Y coordinates.
    z_col (str): Name of the column containing Z values to compare against.
    
    Returns:
    pandas.DataFrame: Original DataFrame with added columns for sampled values and differences.
    """
    def sample_single_point(row):
        geo_x, geo_y = row[x_col], row[y_col]
        pixel_x, pixel_y = geo.geo_to_pixel(geo_x, geo_y, geotransform)
        
        if 0 <= pixel_x < raster_data.shape[1] and 0 <= pixel_y < raster_data.shape[0]:
            value = raster_data[pixel_y, pixel_x]
            if value < -9000:
                print('--------------------------------------------------')
                print('Value is ', value, 'for point ', pixel_x, pixel_y)
                closest_pixel_x, closest_pixel_y = find_closest_valid_pixel(pixel_x, pixel_y, raster_data)
                value = raster_data[closest_pixel_y, closest_pixel_x]  # Get value from the closest valid pixel
                print('So grabbing nearest value')
                print('Value is ', value, 'for point ', closest_pixel_x, closest_pixel_y)
                print('--------------------------------------------------')
                difference = value - row[z_col]
                if abs(difference) > 2 * row[z_col]:  # Check if difference is more than 10%
                    return pd.Series({'Sampled_Value': np.nan, 'Difference': np.nan})
                else:
                    return pd.Series({'Sampled_Value': value, 'Difference': difference})
            else:
                difference = value - row[z_col]
                if abs(difference) > 2 * row[z_col]:  # Check if difference is more than 10%
                    return pd.Series({'Sampled_Value': np.nan, 'Difference': np.nan})
                else:
                    return pd.Series({'Sampled_Value': value, 'Difference': difference})
        else:
            # Find the closest valid pixel if out of bounds
            print('Out of bounds')
            return 

    def find_closest_valid_pixel(pixel_x, pixel_y, raster_data):
        # Define the search range (you can adjust this if needed)
        search_range = 100  # Search 1 pixel in all directions
        height, width = raster_data.shape

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                new_x = pixel_x + dx
                new_y = pixel_y + dy
                
                # Check if the new coordinates are within bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    if raster_data[new_y, new_x] != -9999.0:  # Check for valid value
                        return new_x, new_y  # Return the first valid pixel found

        return pixel_x, pixel_y  # Return original if no valid pixel is found

    # Apply the sampling function to each row
    result = points_df.apply(sample_single_point, axis=1)
    
    # Add the new columns to the original DataFrame
    points_df['Sampled_Value'] = result['Sampled_Value']
    points_df['Difference'] = result['Difference']
    points_df['Difference_Squared'] = result['Difference'] ** 2
    return points_df

def sample_raster_at_points(raster_data, geotransform, points_df, x_col='X', y_col='Y', z_col='Z'):
    """
    Sample raster values at given points and calculate differences.
    
    Args:
    raster_data (numpy.ndarray): The raster data.
    geotransform (tuple): The geotransform of the raster.
    points_df (pandas.DataFrame): DataFrame containing point coordinates.
    x_col (str): Name of the column containing X coordinates.
    y_col (str): Name of the column containing Y coordinates.
    z_col (str): Name of the column containing Z values to compare against.
    
    Returns:
    pandas.DataFrame: Original DataFrame with added columns for sampled values and differences.
    """
    # Apply the sampling function to each row
    result = points_df.apply(lambda row: sample_point(raster_data, geotransform, points_df, x_col, y_col, z_col), axis=1)
    
    # Add the new columns to the original DataFrame
    points_df['Sampled_Value'] = result['Sampled_Value']
    points_df['Difference'] = result['Difference']
    
    return points_df

def calculate_stationing(sampled_points_df, river_line):
    """
    Calculate the stationing of sampled points along a river line.

    Args:
    sampled_points_df (pd.DataFrame): DataFrame containing 'Easting' and 'Northing' columns.
    river_line (gpd.GeoDataFrame): GeoDataFrame containing the river line geometry.

    Returns:
    gpd.GeoDataFrame: Updated GeoDataFrame with stationing calculated.
    """
    # Convert sampled points to a GeoDataFrame
    geometry = gpd.points_from_xy(sampled_points_df['E'], sampled_points_df['N'])
    sampled_points_gdf = gpd.GeoDataFrame(sampled_points_df, geometry=geometry)

    # Calculate the distance along the river line
    sampled_points_gdf['Stationing'] = sampled_points_gdf.geometry.apply(
        lambda point: river_line.geometry[0].project(point)  # Assuming a single line geometry
    )

    return sampled_points_gdf
