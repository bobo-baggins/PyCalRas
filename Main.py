import pandas as pd
import numpy as np
import Calculations as calc
import GeoDataPro as geo
import Output as out    

# Load raster data
raster_file = 'Inputs/Rasters/Design_Surface.tif'
raster_data, geotransform, projection = geo.process_raster(raster_file)

# Load calibration points
input_PNEZ = 'Inputs/Sample_Points/USOLGB.csv'
calibration_points = pd.read_csv(input_PNEZ, header=None)

# Debugging: Print the first row to check its content
print("First Row of Calibration Points:", calibration_points.iloc[0])

if not calibration_points.iloc[0].str.isalpha().all():
    calibration_points.columns = ['P', 'N', 'E', 'Z', 'D']
print(calibration_points.head())

# Generate name of output file based on input file name
output_file = 'Outputs/' + input_PNEZ.split('/')[-1].replace('.csv', '')
print(output_file)

# Load centerline shapefile
centerline_shapefile = 'Inputs/Alignment/Clackamas Calibration Long Profile.shp'
centerline_gdf = geo.read_centerline_shapefile(centerline_shapefile)


# Sample raster values at calibration points
calibration_points = calc.sample_point(raster_data, geotransform, calibration_points, 
                                             x_col='E', y_col='N', z_col='Z')

# Now you can easily access the sampled values and differences
#print(calibration_points[['E', 'N', 'Z', 'Sampled_Value', 'Difference']])

# Print column names and first few rows
#print(calibration_points.columns)
#print(calibration_points.head())

# Calculate stationing
calibration_points = calc.calculate_stationing(calibration_points, centerline_gdf)

# Create the calibration plot
out.create_calibration_plot(calibration_points, 
                               l_bounds=-0.5, 
                               u_bounds=0.5, 
                               output_file=output_file)
# Output calibration points to CSV
calibration_points.to_csv(output_file+'.csv', index=False)
print(f"Sampled points saved to {output_file}")

# Plot WSE comparison
#out.plot_wse_comparison(calibration_points,output_file+'.png')
