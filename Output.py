import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def create_calibration_plot(cal_pts_df, l_bounds=-0.5, u_bounds=0.5, output_file=''):
    """
    Create a calibration plot based on the provided data.

    Args:
    cal_pts_df (pd.DataFrame): DataFrame containing calibration points data.
    l_bounds (float): Lower bound for 'Ok' category.
    u_bounds (float): Upper bound for 'Ok' category.
    output_file (str): Path to save the output plot.

    Returns:
    None
    """
    # Categorize differences
    cal_pts_df['ColorCat'] = pd.cut(cal_pts_df['Difference'],
                                    bins=[-float('inf'), l_bounds, u_bounds, float('inf')],
                                    labels=['Low', 'Ok', 'High'])
    palette = {'Low': 'blue', 'Ok': 'White', 'High': 'red'}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(data=cal_pts_df, 
                    x='E', y='N', 
                    edgecolor='black',
                    hue='ColorCat', 
                    palette=palette,
                    ax=ax)

    # Set the aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add text labels
    texts = []
    for i in range(cal_pts_df.shape[0]):
        if cal_pts_df['ColorCat'][i] in ['Low', 'High']:
            label_text = f"{cal_pts_df['Difference'][i]:.1f}"
            text = ax.annotate(label_text,
                          xy=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                          xytext=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                          fontsize=10,
                          color='black',
                          arrowprops=dict(arrowstyle='->', color='black', shrinkA=1),
                          fontfamily='Arial')
            texts.append(text)

    # Adjust text positions
    adjust_text(texts, 
                only_move={'points': 'xy', 'text': 'xy'},
                max_iterations=500)

    # Add legend
    ax.legend(title='Categories', 
              loc='upper right', 
              fontsize='medium', 
              title_fontsize='13', 
              frameon=True, 
              facecolor='white', 
              edgecolor='black')

    # Add title and subtitles
    plt.title('R_IC_2_10_1: Calibration Check')
    #calculated_value = np.average(cal_pts_df['Difference'])
    Avg_Diff = cal_pts_df['Difference'].mean()
    RMSE = np.sqrt(np.mean(cal_pts_df['Difference_Squared']))
    p_value = cal_pts_df['Z'].corr(cal_pts_df['Sampled_Value'])

    plt.text(0.5, 0.04, f"Ok = + or - {u_bounds} ft.", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    plt.text(0.5, 1.08, f"Average Difference = {Avg_Diff:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    #plt.text(0.5, 1.07, f"RMSE = {RMSE:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    #plt.text(0.5, 1.10, f"Pearson = {p_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

    # Set labels
    plt.xlabel('Easting (ft)')
    plt.ylabel('Northing (ft)')

    # Save and close
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Plot saved to {output_file}")
    print(f"Number of NaN values in Difference: {cal_pts_df['Difference'].isna().sum()}")

def plot_wse_comparison(sampled_points_gdf, output_file):
    """
    Plot a comparison of measured WSE vs. model WSE.

    Args:
    sampled_points_gdf (gpd.GeoDataFrame): GeoDataFrame containing stationing and WSE data.
    model_wse_col (str): Name of the column containing model WSE values.
    output_file (str): Path to save the output PNG file.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot measured WSE (Z values)
    ax.scatter(sampled_points_gdf['Stationing'], sampled_points_gdf['Z'], 
                color='blue', label='Surveyed Depth', marker='o', s=10)
    
    # Plot sampled values
    ax.scatter(sampled_points_gdf['Stationing'], sampled_points_gdf['Sampled_Value'], 
                color='green', label='Model Results', marker='s', s=10)

   # Calculate Stats
    Avg_Diff = sampled_points_gdf['Difference'].mean() 
    RMSE = np.sqrt(np.mean(sampled_points_gdf['Difference_Squared']))
    p_value = sampled_points_gdf['Z'].corr(sampled_points_gdf['Sampled_Value'])

    # Add Subtitles
    ax.text(0.5, 1.10, f"Average Difference = {Avg_Diff:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 1.04, f"RMSE = {RMSE:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 1.07, f"Pearson = {p_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

    ax.set_title('WSE Comparison: Measured vs. Model')
    ax.set_xlabel('Stationing (ft.)')
    ax.set_ylabel('WSE (ft.)')
    ax.legend()
    ax.grid()
    
    # Reverse the x-axis
    ax.invert_xaxis()
    
    # Save the plot as a PNG file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"WSE comparison plot saved to {output_file}")
