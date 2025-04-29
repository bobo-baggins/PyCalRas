# Standard library imports
import os
import sys
from datetime import datetime

# Local application imports
import reading
import calculations as calc
import output as out
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

def main():
    """Main function to process calibration points and raster data."""
    try:
        # Get executable directory and setup paths
        executable_dir = reading.get_executable_dir()
        points_file, raster_file, centerline_file = reading.setup_directories(executable_dir)
        
        # Load calibration points
        calibration_points = reading.load_calibration_points(points_file)
        logger.info('Calibration points loaded successfully')
        
        # Load centerline
        centerline_gdf = reading.load_centerline(centerline_file)
        logger.info('Centerline loaded successfully')
        
        # Process raster file
        raster_data, geotransform, projection = reading.process_raster(raster_file)
        logger.info(f'Raster file processed successfully')
        
        # Generate output file paths
        raster_name = os.path.splitext(os.path.basename(raster_file))[0]
        output_file = os.path.join(executable_dir, 'Outputs', f"{raster_name}.png")
        
        # Skip if output already exists
        if os.path.exists(output_file):
            logger.info(f"Output file {output_file} already exists, skipping.")
            return
        
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
        output_csv = output_file.replace('.png', '.csv')
        calibration_points.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
        
        # Test output against reference
        if calc.test_output(output_csv):
            logger.info("Output validation passed [OK]")
        else:
            logger.warning("Output validation failed [FAILED]")
            
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 