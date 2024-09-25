import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from osgeo import gdal, osr
from adjustText import adjust_text

# Open the raster file
raster_file = 'Inputs/Supply_6cfs_WSE_tif.tif'
ds = gdal.Open(raster_file)

# Read the raster band
band = ds.GetRasterBand(1)
raster_data = band.ReadAsArray()

# Get geotransform and projection
geotransform = ds.GetGeoTransform()
#print(geotransform)
#print()
#projection = ds.GetProjection()
#print(projection)
# Function to convert geographic coordinates to pixel coordinates

def geo_to_pixel(geo_x, geo_y):
    pixel_x = int((geo_x - geotransform[0]) / geotransform[1])
    pixel_y = int((geo_y - geotransform[3]) / geotransform[5])
    return pixel_x, pixel_y

# Read PNEZ points (assuming they are in a CSV format)
input_PNEZ = 'Inputs\Supply2D_80p5cfs_Surveyed_WSE.csv'
calibration_points = np.loadtxt(input_PNEZ, delimiter=',', skiprows=1)
cal_pts_df = pd.read_csv(input_PNEZ)

#print("----------")
#print(calibration_points)
#print("----------")

# Sample raster values at PNEZ points
sampled_values = []
calc_values = []
for point in calibration_points:
    # Point [0] is point number generally
    # Here point[2] is easting (x)
    # Here point[1] is northing (y)
    geo_x, geo_y = point[2], point[1]  

    pixel_x, pixel_y = geo_to_pixel(geo_x, geo_y)
    
    # Check if pixel coordinates are within bounds
    if 0 <= pixel_x < ds.RasterXSize and 0 <= pixel_y < ds.RasterYSize:
        value = raster_data[pixel_y, pixel_x]
        #print(value)
        if value > 0:
            sampled_values.append(value)
            diff = value - point[3]
            calc_values.append(diff)
        else:
            sampled_values.append(None)
            calc_values.append(None)
    else: # Out of bounds
        sampled_values.append(None)  
        calc_values.append(None)

#print('Sample Check')
#print('length:')
#print(len(sampled_values))
#print('Values:')
#print(sampled_values)
#print('--------------------')
cal_pts_df['Target'] = sampled_values
cal_pts_df['Diff'] = calc_values

if len(sampled_values) != len(cal_pts_df):
    print("Warning Length mismatch of sample_values")

cal_pts_df.to_csv('Outputs\Supply_WSE_Check.csv', index=False)
#print(cal_pts_df)

# Calculate RMSE
# Square the difference between point[3] and value
sq_diff = [diff**2 for diff in calc_values if diff is not None]
Sum_sq_diff = sum(sq_diff)
RMSE = np.sqrt(Sum_sq_diff / len(sq_diff))

# Pearsons
model_wse = np.array(sampled_values, dtype=float)
obs_wse = np.array(calibration_points[:, 3], dtype=float)

# Create a mask for non-None values
mask = ~np.isnan(model_wse) & ~np.isnan(obs_wse)

# Apply the mask to the arrays
model_wse_mask = model_wse[mask]
#print(model_wse_mask)
#print()
obs_wse_mask = obs_wse[mask]
#print(obs_wse_mask)
#print()

# Calculate Pearson's correlation coefficient for pair values only
if len(model_wse_mask) > 1 and len(obs_wse_mask) > 1:
    correlation_coefficient, p_value = pearsonr(model_wse_mask, obs_wse_mask)
else:
    correlation_coefficient, p_value = None, None

# Close the dataset
ds = None

# Output sampled values
print('RSME Value:')
print(RMSE)
print()
print('Pearson Value:')
print(correlation_coefficient)
print()
print(p_value)

l_bounds = -0.5
u_bounds = 0.5

cal_pts_df['ColorCat'] = pd.cut(cal_pts_df['Diff'],
                                bins=[-float('inf'), l_bounds, u_bounds, float('inf')],
                                labels=['Low', 'Ok', 'High'])
palette = {'Low': 'blue', 'Ok': 'White', 'High': 'red'}

fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(data=cal_pts_df, 
                x='E', y='N', 
                edgecolor='black',
                hue='ColorCat', 
                palette=palette,
                ax=ax)



# for i in range(cal_pts_df.shape[0]):
#     if cal_pts_df['ColorCat'][i] == 'Low' or cal_pts_df['ColorCat'][i] == 'High':
#         label_sigfig = f"{cal_pts_df['Diff'][i]:.1f}"
        
#         offset_x = 5 + (1 if i% 2 == 0 else -5)
#         offset_y = 5 + (1 if i% 2 == 0 else -5)

#         plt.annotate(label_sigfig,
#                      xy=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
#                      xytext=(cal_pts_df['E'][i]+offset_x, cal_pts_df['N'][i]+offset_y),
#                      arrowprops=dict(arrowstyle='->', color='black'),
#                      fontsize=12,
#                      color='black',
#                      fontfamily='sans-serif')

# Add text labels
texts = []
for i in range(cal_pts_df.shape[0]):
    if cal_pts_df['ColorCat'][i] == 'Low' or cal_pts_df['ColorCat'][i] == 'High':
        label_text = f"{cal_pts_df['Diff'][i]:.1f}"  # Format the label
        #text = plt.text(cal_pts_df['E'][i], cal_pts_df['N'][i], label_text, fontsize=12, color='black')
        text = ax.annotate(label_text,
                      xy=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                      xytext=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                      fontsize=12,
                      color='black',
                      arrowprops=dict(arrowstyle='->', color='black', shrinkA=1),  # Optional: add arrows
                      fontfamily='Arial')

        texts.append(text)  # Store the text object for adjustment

# Use adjustText to avoid overlaps
adjust_text(texts, 
            only_move={'points': 'xy', 'text': 'xy'},  # Only move vertically
            max_iterations=300)  # Limit the number of iterations





# Add a legend with formatting
ax.legend(title='Categories', 
          loc='upper right', 
          fontsize='medium', 
          title_fontsize='13', 
          frameon=True, 
          facecolor='white', 
          edgecolor='black')

plt.title('Supply 80p5cfs: WSE Calibration')
# Add subtitles
print(cal_pts_df['Diff'].isna().sum())
calculated_value = np.average(cal_pts_df['Diff'])
plt.text(0.5, 0.04, "Ok = + or - 0.5 ft.", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
plt.text(0.5, 1.06, f"Average Difference = {calculated_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
plt.text(0.5, 1.08, f"RMSE = {RMSE:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
plt.text(0.5, 1.10, f"Pearson = {p_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

plt.xlabel('Easting (ft)')
plt.ylabel('Northing (ft)')
#plt.show()
plt.savefig('Outputs\Supply_WSE_Check.png', dpi=300)
plt.close()

