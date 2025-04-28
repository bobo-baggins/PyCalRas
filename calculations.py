import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_raster_at_points(
    raster_data: np.ndarray,
    geotransform: Tuple[float, ...],
    points_df: pd.DataFrame,
    x_col: str = 'X',
    y_col: str = 'Y',
    z_col: str = 'Z',
    invalid_threshold: float = -9000,
    max_difference_ratio: float = 2.0
) -> pd.DataFrame:
    """
    Sample raster values at given points and calculate differences.
    
    Args:
        raster_data: The raster data array
        geotransform: The geotransform of the raster
        points_df: DataFrame containing point coordinates
        x_col: Name of the column containing X coordinates
        y_col: Name of the column containing Y coordinates
        z_col: Name of the column containing Z values to compare against
        invalid_threshold: Threshold for invalid raster values
        max_difference_ratio: Maximum allowed ratio between sampled and reference values
    
    Returns:
        DataFrame with added columns for sampled values and differences
    """
    def geo_to_pixel(geo_x: float, geo_y: float) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel coordinates."""
        pixel_x = int((geo_x - geotransform[0]) / geotransform[1])
        pixel_y = int((geo_y - geotransform[3]) / geotransform[5])
        return pixel_x, pixel_y

    def find_closest_valid_pixel(pixel_x: int, pixel_y: int) -> Optional[Tuple[int, int]]:
        """Find the closest valid pixel using a more efficient search pattern."""
        height, width = raster_data.shape
        max_search = min(100, max(height, width) // 10)  # Limit search range
        
        # Create a spiral search pattern
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        new_x, new_y = pixel_x + dx, pixel_y + dy
                        if (0 <= new_x < width and 0 <= new_y < height and 
                            raster_data[new_y, new_x] > invalid_threshold):
                            return new_x, new_y
        return None

    # Initialize result arrays
    sampled_values = np.full(len(points_df), np.nan)
    differences = np.full(len(points_df), np.nan)
    
    # Process each point
    for idx, row in points_df.iterrows():
        try:
            pixel_x, pixel_y = geo_to_pixel(row[x_col], row[y_col])
            
            # Check if point is within raster bounds
            if not (0 <= pixel_x < raster_data.shape[1] and 0 <= pixel_y < raster_data.shape[0]):
                logger.warning(f"Point {idx} is outside raster bounds")
                continue
                
            value = raster_data[pixel_y, pixel_x]
            
            # Handle invalid values
            if value <= invalid_threshold:
                closest = find_closest_valid_pixel(pixel_x, pixel_y)
                if closest is None:
                    logger.warning(f"No valid pixel found near point {idx}")
                    continue
                value = raster_data[closest[1], closest[0]]
            
            # Calculate difference and check if it's within acceptable range
            difference = value - row[z_col]
            if abs(difference) <= max_difference_ratio * abs(row[z_col]):
                sampled_values[idx] = value
                differences[idx] = difference
            
        except Exception as e:
            logger.error(f"Error processing point {idx}: {str(e)}")
            continue
    
    # Add results to DataFrame
    points_df['Sampled_Value'] = sampled_values
    points_df['Difference'] = differences
    points_df['Difference_Squared'] = differences ** 2
    
    return points_df

def calculate_stationing(
    sampled_points_df: pd.DataFrame,
    river_line: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Calculate the stationing of sampled points along a river line.

    Args:
        sampled_points_df: DataFrame containing 'E' and 'N' columns
        river_line: GeoDataFrame containing the river line geometry

    Returns:
        GeoDataFrame with stationing calculated
    """
    try:
        # Convert sampled points to a GeoDataFrame
        geometry = gpd.points_from_xy(sampled_points_df['E'], sampled_points_df['N'])
        sampled_points_gdf = gpd.GeoDataFrame(sampled_points_df, geometry=geometry)

        # Calculate the distance along the river line
        sampled_points_gdf['Stationing'] = sampled_points_gdf.geometry.apply(
            lambda point: river_line.geometry[0].project(point)
        )

        return sampled_points_gdf
        
    except Exception as e:
        logger.error(f"Error calculating stationing: {str(e)}")
        raise
