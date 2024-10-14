import pandas as pd
import numpy as np
import Calculations as calc
import GeoDataPro as geo
import Output as out
import os
import sys

def get_executable_dir():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        return os.path.dirname(os.path.abspath(__file__))

# Use this function to construct your paths
executable_dir = get_executable_dir()

# Now construct your paths relative to executable_dir
sample_points_directory = os.path.join(executable_dir, 'Inputs', 'Sample_Points')
rasters_directory = os.path.join(executable_dir, 'Inputs', 'Rasters')
alignment_directory = os.path.join(executable_dir, 'Inputs', 'Alignment')

# Print for debugging
print(f"Executable directory: {executable_dir}")
print(f"sample_points_directory: {sample_points_directory}")
print(f"rasters_directory: {rasters_directory}")
print(f"alignment_directory: {alignment_directory}")

def main():
    # Check if directories exist
    for directory in [sample_points_directory, rasters_directory, alignment_directory]:
        if not os.path.exists(directory):
            print(f"Error: Directory does not exist: {directory}")
            return  # Exit the function if any directory is missing

    # Load calibration points
    csv_files = [f for f in os.listdir(sample_points_directory) if f.endswith('.csv')]
    if not csv_files:
        print("No .csv files found in the directory.")
        return
    input_PNEZ = os.path.join(sample_points_directory, csv_files[0])
    calibration_points = pd.read_csv(input_PNEZ)
    print('--------------------')
    print('calibration_points:')
    print(calibration_points.head())
    print('--------------------')

    # Ensure columns are set correctly
    calibration_points.columns = ['P', 'N', 'E', 'Z', 'D']

    # Load centerline shapefile with error-checking
    shape_files = [f for f in os.listdir(alignment_directory) if f.endswith('.shp')]
    if not shape_files:
        print("No .shp files found in the directory.")
        return
    centerline_gdf = geo.read_centerline_shapefile(os.path.join(alignment_directory, shape_files[0]))

    # Load raster data
    raster_files = [f for f in os.listdir(rasters_directory) if f.endswith('.tif')]
    if not raster_files:
        print("No .tif files found in the directory.")
        return
    raster_data_list = []
    raster_file_dict = {}  # New dictionary to store file names
    for raster_file in raster_files:
        raster_data, geotransform, projection = geo.process_raster(os.path.join(rasters_directory, raster_file))
        raster_data_list.append((raster_data, geotransform, projection))
        raster_file_dict[id(raster_data)] = raster_file  # Store file name using id of raster_data as key

    for raster_data, geotransform, projection in raster_data_list:

        # Generate name of output file based on the current raster file name
        raster_file_name = os.path.splitext(raster_file_dict[id(raster_data)])[0]
        output_file = os.path.join(executable_dir, 'Outputs', raster_file_name)
        print(output_file)

        # Check if the output file already exists, if so, skip this raster
        if os.path.exists(output_file + '.csv'):
            print(f"Output file {output_file} already exists, skipping this raster.")
            continue

        # Sample raster values at calibration points
        calibration_points = calc.sample_point(raster_data, geotransform, calibration_points, 
                                             x_col='E', y_col='N', z_col='Z')

        # Calculate stationing
        calibration_points = calc.calculate_stationing(calibration_points, centerline_gdf)

        # Create the calibration plot
        out.plot_wse_comparison(calibration_points, output_file)

        # Output calibration points to CSV
        calibration_points.to_csv(output_file+'.csv', index=False)
        print(f"Sampled points saved to {output_file}")

if __name__ == "__main__":
    main()
