import pandas as pd
import numpy as np
import Calculations as calc
import GeoDataPro as geo
import Output as out
import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    # Load calibration points
    sample_points_directory = resource_path('Inputs/Sample_Points')
    csv_files = [f for f in os.listdir(sample_points_directory) if f.endswith('.csv')]
    if not csv_files:
        print("No .csv files found in the directory.")
        return
    input_PNEZ = os.path.join(sample_points_directory, csv_files[0])
    calibration_points = pd.read_csv(input_PNEZ)

    # Ensure columns are set correctly
    calibration_points.columns = ['P', 'N', 'E', 'Z', 'D']

    # Load centerline shapefile with error-checking
    centerline_shapefile_path = resource_path('Inputs/Alignment')
    shape_files = [f for f in os.listdir(centerline_shapefile_path) if f.endswith('.shp')]
    if not shape_files:
        print("No .shp files found in the directory.")
        return
    centerline_gdf = geo.read_centerline_shapefile(os.path.join(centerline_shapefile_path, shape_files[0]))

    # Load raster data
    raster_files = [f for f in os.listdir('Inputs/Rasters') if f.endswith('.tif')]
    if not raster_files:
        print("No .tif files found in the directory.")
        return
    raster_data_list = []
    raster_file_dict = {}  # New dictionary to store file names
    for raster_file in raster_files:
        raster_data, geotransform, projection = geo.process_raster(resource_path('Inputs/Rasters/' + raster_file))
        raster_data_list.append((raster_data, geotransform, projection))
        raster_file_dict[id(raster_data)] = raster_file  # Store file name using id of raster_data as key

    for raster_data, geotransform, projection in raster_data_list:

        # Generate name of output file based on the current raster file name
        raster_file_name = os.path.splitext(raster_file_dict[id(raster_data)])[0]
        output_file = 'Outputs/' + raster_file_name
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
