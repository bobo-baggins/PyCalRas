import pandas as pd
import numpy as np
import GeoDataPro as geo
import os
import logging
import geopandas as gpd
from typing import List, Tuple
import sys

# Configure logging
logger = logging.getLogger(__name__)

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
        Tuple containing paths to sample points, rasters, and alignment files
    """
    # Define directory paths
    sample_points_dir = os.path.join(executable_dir, 'Inputs', 'Sample_Points')
    raster_dir = os.path.join(executable_dir, 'Inputs', 'Rasters')
    alignment_dir = os.path.join(executable_dir, 'Inputs', 'Alignment')
    
    # Create output directory if it doesn't exist
    output_directory = os.path.join(executable_dir, 'Outputs')
    os.makedirs(output_directory, exist_ok=True)
    
    # Find the first CSV file in the Sample_Points directory
    csv_files = [f for f in os.listdir(sample_points_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {sample_points_dir}")
    sample_points_file = os.path.join(sample_points_dir, csv_files[0])
    
    # Find the first TIF file in the Rasters directory
    tif_files = [f for f in os.listdir(raster_dir) if f.endswith('.tif')]
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {raster_dir}")
    raster_file = os.path.join(raster_dir, tif_files[0])
    
    # Find the first SHP file in the Alignment directory
    shp_files = [f for f in os.listdir(alignment_dir) if f.endswith('.shp')]
    if not shp_files:
        raise FileNotFoundError(f"No SHP files found in {alignment_dir}")
    alignment_file = os.path.join(alignment_dir, shp_files[0])
    
    # Validate that files exist
    for filepath in [sample_points_file, raster_file, alignment_file]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file does not exist: {filepath}")
    
    logger.info(f"Using sample points file: {os.path.basename(sample_points_file)}")
    logger.info(f"Using raster file: {os.path.basename(raster_file)}")
    logger.info(f"Using alignment file: {os.path.basename(alignment_file)}")
    
    return sample_points_file, raster_file, alignment_file

def load_calibration_points(filepath: str) -> pd.DataFrame:
    """
    Load calibration points from CSV file.
    
    Args:
        filepath: Path to the calibration points CSV file
        
    Returns:
        DataFrame containing calibration points
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Calibration points file must be a CSV file")
    
    calibration_points = pd.read_csv(filepath)
    
    if calibration_points.empty:
        raise ValueError("Calibration points file is empty")
    
    # Ensure columns are set correctly
    required_columns = ['P', 'N', 'E', 'Z', 'D']
    if not all(col in calibration_points.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    calibration_points.columns = required_columns
    return calibration_points

def load_centerline(filepath: str) -> gpd.GeoDataFrame:
    """
    Load centerline shapefile.
    
    Args:
        filepath: Path to the centerline shapefile
        
    Returns:
        GeoDataFrame containing centerline data
    """
    if not filepath.endswith('.shp'):
        raise ValueError("Centerline file must be a shapefile")
    
    return geo.read_centerline_shapefile(filepath)

def process_raster(filepath: str) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    """
    Process a single raster file.
    
    Args:
        filepath: Path to the raster file
        
    Returns:
        Tuple containing (raster_data, geotransform, projection)
    """
    if not filepath.endswith('.tif'):
        raise ValueError("Raster file must be a TIF file")
    
    try:
        raster_data, geotransform, projection = geo.process_raster(filepath)
        return raster_data, geotransform, projection
    except Exception as e:
        logger.error(f"Error processing raster file {filepath}: {str(e)}")
        raise