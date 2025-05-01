# Standard library imports
import os
import sys
from typing import Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import gdal

# Local imports
import GeoDataPro as geo
import calculations as calc
import output as out
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

def get_executable_dir() -> str:
    """
    Get the directory where the executable or script is located.
    
    Returns:
        str: Path to the executable directory
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_run_info() -> Tuple[str, str]:
    """
    Prompt user for run name and description.
    
    Returns:
        Tuple[str, str]: (run_name, description)
        
    Note:
        Run name must be valid for directory names and cannot be empty.
        Description can be multi-line and will be saved with the run.
    """
    executable_dir = get_executable_dir()
    
    # Get run name
    while True:
        run_name = input("\nEnter a name for this calibration run: ").strip()
        if not run_name:
            print("Run name cannot be empty. Please try again.")
            continue
            
        # Validate run name for directory use
        if any(c in run_name for c in r'<>:"/\|?*'):
            print("Run name contains invalid characters. Please use only letters, numbers, spaces, and underscores.")
            continue
            
        # Check for existing run
        run_dir = os.path.join(executable_dir, 'Outputs', run_name)
        if os.path.exists(run_dir):
            overwrite = input(f"Run directory '{run_name}' already exists. Overwrite? (y/n): ").lower()
            if overwrite != 'y':
                continue
        break
    
    # Get run description
    print("\nEnter a description of this calibration run (press Enter twice to finish):")
    description_lines = []
    while True:
        line = input()
        if not line and description_lines and not description_lines[-1]:
            break
        description_lines.append(line)
    
    description = '\n'.join(description_lines).strip()
    return run_name, description or "No description provided."

def _find_first_file(directory: str, extension: str, dir_name: str) -> str:
    """
    Find the first file with given extension in a directory.
    
    Args:
        directory: Directory to search in
        extension: File extension to look for
        dir_name: Name of directory for error messages
        
    Returns:
        str: Path to the first file found
        
    Raises:
        FileNotFoundError: If no matching files are found
    """
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"No {extension} files found in {dir_name} directory")
    
    file_path = os.path.join(directory, files[0])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file does not exist: {file_path}")
        
    return file_path

def setup_directories(executable_dir: str) -> Tuple[str, str, str]:
    """
    Set up and validate required directories and input files.
    
    Args:
        executable_dir: Base directory for the application
        
    Returns:
        Tuple[str, str, str]: Paths to (sample_points_file, raster_file, alignment_file)
        
    Raises:
        FileNotFoundError: If required directories or files are not found
    """
    # Define directory paths
    sample_points_dir = os.path.join(executable_dir, 'Inputs', 'Sample_Points')
    raster_dir = os.path.join(executable_dir, 'Inputs', 'Rasters')
    alignment_dir = os.path.join(executable_dir, 'Inputs', 'Alignment')
    
    # Create output directory
    os.makedirs(os.path.join(executable_dir, 'Outputs'), exist_ok=True)
    
    # Find input files
    sample_points_file = _find_first_file(sample_points_dir, '.csv', 'Sample_Points')
    raster_file = _find_first_file(raster_dir, '.tif', 'Rasters')
    alignment_file = _find_first_file(alignment_dir, '.shp', 'Alignment')
    
    # Log file selections
    logger.info(f"Using sample points file: {os.path.basename(sample_points_file)}")
    logger.info(f"Using raster file: {os.path.basename(raster_file)}")
    logger.info(f"Using alignment file: {os.path.basename(alignment_file)}")
    
    return sample_points_file, raster_file, alignment_file

def load_calibration_points(filepath: str) -> pd.DataFrame:
    """
    Load calibration points from CSV file.
    Matches columns based on first letter:
    P - Point Name
    N - Northing
    E - Easting
    W - Water Surface Elevation (WSE)
    D - Depth
    V - Velocity
    
    Args:
        filepath: Path to the calibration points CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing calibration points with standardized column names
        
    Raises:
        ValueError: If file is not CSV or missing required columns
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Calibration points file must be a CSV file")
    
    calibration_points = pd.read_csv(filepath)
    
    if calibration_points.empty:
        raise ValueError("Calibration points file is empty")
    
    # Define column mapping based on first letter
    column_mapping = {}
    for col in calibration_points.columns:
        first_letter = col[0].upper()
        if first_letter in ['P', 'N', 'E']:
            column_mapping[col] = first_letter
        elif first_letter == 'W':
            column_mapping[col] = 'WSE'
        elif first_letter == 'D':
            column_mapping[col] = 'Depth'
        elif first_letter == 'V':
            column_mapping[col] = 'Velocity'
    
    # Verify we found the required columns
    required_columns = ['N', 'E']  # These are always required
    missing_columns = [col for col in required_columns if col not in column_mapping.values()]
    if missing_columns:
        raise ValueError(f"Could not find required columns: {missing_columns}")
    
    # Verify we have at least one data type column
    data_type_columns = ['WSE', 'Depth', 'Velocity']
    if not any(col in column_mapping.values() for col in data_type_columns):
        raise ValueError("Could not find any data type columns (WSE, Depth, or Velocity)")
    
    # Rename columns
    calibration_points = calibration_points.rename(columns=column_mapping)
    
    return calibration_points

def load_centerline(filepath: str) -> gpd.GeoDataFrame:
    """
    Load centerline shapefile.
    
    Args:
        filepath: Path to the centerline shapefile
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing centerline data
        
    Raises:
        ValueError: If file is not a shapefile
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
        Tuple[np.ndarray, Tuple[float, ...], str]: (raster_data, geotransform, projection)
        
    Raises:
        ValueError: If file is not a TIF file
        Exception: If raster processing fails
    """
    if not filepath.endswith('.tif'):
        raise ValueError("Raster file must be a TIF file")
    
    try:
        return geo.process_raster(filepath)
    except Exception as e:
        logger.error(f"Error processing raster file {filepath}: {str(e)}")
        raise

def process_calibration_run(
    executable_dir: str,
    run_name: str,
    description: str,
    test_mode: bool = False
) -> None:
    """
    Process a single calibration run.
    
    Args:
        executable_dir: Base directory for the application
        run_name: Name of the calibration run
        description: Description of the calibration run
        test_mode: Whether to run in test mode
        
    Raises:
        Exception: If any step in the calibration process fails
    """
    # Create run directory and setup output paths
    run_dir = out.create_run_directory(executable_dir, run_name, description)
    
    # Setup input file paths
    points_file, raster_file, centerline_file = setup_directories(executable_dir)
    
    # Load and process input data
    calibration_points = load_calibration_points(points_file)
    logger.info('Calibration points loaded successfully')
    
    centerline_gdf = load_centerline(centerline_file)
    logger.info('Centerline loaded successfully')
    
    raster_data, geotransform, projection = process_raster(raster_file)
    logger.info(f'Raster file processed successfully')
    
    # Generate output file paths
    raster_name = os.path.splitext(os.path.basename(raster_file))[0]
    plot_file, wse_plot_file, csv_file = out.get_output_paths(run_dir, raster_name)
    
    # Process the raster
    logger.info(f"Processing raster: {raster_file}")
    
    # Sample raster values and calculate stationing
    calibration_points = calc.sample_raster_at_points(
        raster_data, geotransform, calibration_points,
        x_col='E', y_col='N', wse='WSE'
    )
    calibration_points = calc.calculate_stationing(calibration_points, centerline_gdf)
    
    # Create visualizations and get statistics
    rmse, avg_diff = out.create_calibration_plot(calibration_points, centerline_gdf, output_file=plot_file)
    out.plot_wse_comparison(calibration_points, output_file=wse_plot_file)
    
    # Save results
    if 'geometry' in calibration_points.columns:
        calibration_points = calibration_points.drop(columns=['geometry'])
    
    calibration_points.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")
    
    # Update run log with statistics
    out.update_run_log(executable_dir, run_name, description, rmse, avg_diff)
    
    # Validate output if in test mode
    if test_mode:
        if calc.test_output(csv_file):
            logger.info("Output validation passed [OK]")
        else:
            logger.warning("Output validation failed [FAILED]")
    else:
        logger.info("Output validation skipped (use --test to enable)")