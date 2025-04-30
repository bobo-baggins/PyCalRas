# Standard library imports
import os
import sys
import argparse
from datetime import datetime

# Local application imports
import reading
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

def main():
    """Main function to process calibration points and raster data."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Process calibration points and raster data.')
        parser.add_argument('--test', action='store_true', help='Run output validation against test files')
        args = parser.parse_args()

        # Get executable directory
        executable_dir = reading.get_executable_dir()
        
        # Get run information from user
        run_name, description = reading.get_run_info()
        
        # Process the calibration run
        reading.process_calibration_run(
            executable_dir=executable_dir,
            run_name=run_name,
            description=description,
            test_mode=args.test
        )
            
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 