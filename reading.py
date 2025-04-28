import pandas as pd
import numpy as np
import GeoDataPro as geo
import os
import logging
import geopandas as gpd
from typing import List, Tuple
import sys

# Function to get the executable directory
def get_executable_dir() -> str:
    """
    Get the directory of the executable or script.
    
    Returns:
        str: Path to the executable directory
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        return os.path.dirname(os.path.abspath(__file__))

def setup_directories(executable_dir: str) -> Tuple[str, str, str]:
    """
    Set up and validate required directories.
    
    Args:
        executable_dir: Base directory for the application
        
    Returns:
        Tuple containing paths to sample points, rasters, and alignment directories
        
    Raises:
        FileNotFoundError: If any required directory is missing
    """
    # Define directory paths
    sample_points_directory = os.path.join(executable_dir, 'Inputs', 'Sample_Points')
    rasters_directory = os.path.join(executable_dir, 'Inputs', 'Rasters')
    alignment_directory = os.path.join(executable_dir, 'Inputs', 'Alignment')
    
    # Create output directory if it doesn't exist
    output_directory = os.path.join(executable_dir, 'Outputs')
    os.makedirs(output_directory, exist_ok=True)
    
    # Validate input directories
    for directory in [sample_points_directory, rasters_directory, alignment_directory]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Required directory does not exist: {directory}")
    
    return sample_points_directory, rasters_directory, alignment_directory

def load_calibration_points(directory: str) -> pd.DataFrame:
    """
    Load calibration points from CSV file.
    
    Args:
        directory: Directory containing calibration point files
        
    Returns:
        DataFrame containing calibration points
        
    Raises:
        FileNotFoundError: If no CSV files are found
        ValueError: If CSV file is empty or malformed
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No .csv files found in the directory.")
    
    input_file = os.path.join(directory, csv_files[0])
    calibration_points = pd.read_csv(input_file)
    
    if calibration_points.empty:
        raise ValueError("Calibration points file is empty.")
    
    # Ensure columns are set correctly
    required_columns = ['P', 'N', 'E', 'Z', 'D']
    if not all(col in calibration_points.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    calibration_points.columns = required_columns
    return calibration_points

def load_centerline(directory: str) -> gpd.GeoDataFrame:
    """
    Load centerline shapefile.
    
    Args:
        directory: Directory containing shapefile
        
    Returns:
        GeoDataFrame containing centerline data
        
    Raises:
        FileNotFoundError: If no shapefile is found
    """
    shape_files = [f for f in os.listdir(directory) if f.endswith('.shp')]
    if not shape_files:
        raise FileNotFoundError("No .shp files found in the directory.")
    
    return geo.read_centerline_shapefile(os.path.join(directory, shape_files[0]))

def process_raster_files(directory: str) -> List[Tuple[np.ndarray, Tuple[float, ...], str, str]]:
    """
    Process all raster files in the directory.
    
    Args:
        directory: Directory containing raster files
        
    Returns:
        List of tuples containing (raster_data, geotransform, projection, filename)
        
    Raises:
        FileNotFoundError: If no raster files are found
    """
    raster_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    if not raster_files:
        raise FileNotFoundError("No .tif files found in the directory.")
    
    raster_data_list = []
    for raster_file in raster_files:
        try:
            raster_data, geotransform, projection = geo.process_raster(
                os.path.join(directory, raster_file)
            )
            raster_data_list.append((raster_data, geotransform, projection, raster_file))
        except Exception as e:
            logger.error(f"Error processing raster file {raster_file}: {str(e)}")
            continue
    
    return raster_data_list 