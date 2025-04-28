# Standard library imports
import os
import sys
import logging
from datetime import datetime

# Local application imports
import reading
import Calculations as calc
import Output as out

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'pycalras_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to process calibration points and raster data."""
    try:
        # Get executable directory and setup paths
        executable_dir = reading.get_executable_dir()
        sample_points_directory, rasters_directory, alignment_directory = reading.setup_directories(executable_dir)
        
        # Load calibration points
        calibration_points = reading.load_calibration_points(sample_points_directory)
        logger.info('Calibration points loaded successfully')
        
        # Load centerline
        centerline_gdf = reading.load_centerline(alignment_directory)
        logger.info('Centerline loaded successfully')
        
        # Process raster files
        raster_data_list = reading.process_raster_files(rasters_directory)
        logger.info(f'Found {len(raster_data_list)} raster files to process')
        
        # Process each raster
        for raster_data, geotransform, projection, raster_file in raster_data_list:
            try:
                # Generate output file path
                raster_name = os.path.splitext(raster_file)[0]
                output_file = os.path.join(executable_dir, 'Outputs', f"{raster_name}.png")
                
                # Skip if output already exists
                if os.path.exists(output_file):
                    logger.info(f"Output file {output_file} already exists, skipping.")
                    continue
                
                # Process the raster
                logger.info(f"Processing raster: {raster_file}")
                
                # Sample raster values
                calibration_points = calc.sample_raster_at_points(
                    raster_data, geotransform, calibration_points,
                    x_col='E', y_col='N', z_col='Z'
                )
                
                # Calculate stationing
                calibration_points = calc.calculate_stationing(calibration_points, centerline_gdf)
                
                # Create plots
                out.create_calibration_plot(calibration_points, output_file=output_file)
                out.plot_wse_comparison(calibration_points, output_file=output_file.replace('.png', '_wse.png'))
                
                # Drop geometry column before saving to CSV
                if 'geometry' in calibration_points.columns:
                    calibration_points = calibration_points.drop(columns=['geometry'])
                
                # Save results
                calibration_points.to_csv(output_file.replace('.png', '.csv'), index=False)
                logger.info(f"Results saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing raster {raster_file}: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 