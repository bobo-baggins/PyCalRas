import pandas as pd
import numpy as np
import Calculations as calc
import GeoDataPro as geo
import Output as out    

# Load raster data
raster_file = 'Inputs/Supply_6cfs_WSE_tif.tif'
raster_data, geotransform, projection = geo.process_raster(raster_file)

# Load calibration points
input_PNEZ = 'Inputs/Supply2D_80p5cfs_Surveyed_WSE.csv'
calibration_points = pd.read_csv(input_PNEZ)

# Print column names and first few rows
print(calibration_points.columns)
print(calibration_points.head())

# Sample raster values at calibration points
calibration_points = calc.sample_point(raster_data, geotransform, calibration_points, 
                                             x_col='E', y_col='N', z_col='Z')

# Now you can easily access the sampled values and differences
print(calibration_points[['E', 'N', 'Z', 'Sampled_Value', 'Difference']])


# Create the calibration plot
out.create_calibration_plot(calibration_points, 
                               l_bounds=-0.5, 
                               u_bounds=0.5, 
                               output_file='Outputs/Supply_WSE_Check.png')
# Output calibration points to CSV
output_csv_path = 'Outputs/sampled_points.csv'
calibration_points.to_csv(output_csv_path, index=False)
print(f"Sampled points saved to {output_csv_path}")

