# Standard library imports
from typing import Optional

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Local imports
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

def create_calibration_plot(
    cal_pts_df: pd.DataFrame,
    centerline_gdf: pd.DataFrame,
    l_bounds: float = -0.5,
    u_bounds: float = 0.5,
    output_file: str = ''
) -> None:
    """
    Create a calibration plot based on the provided data.

    Args:
        cal_pts_df: DataFrame containing calibration points data
        centerline_gdf: GeoDataFrame containing the centerline data
        l_bounds: Lower bound for 'Ok' category
        u_bounds: Upper bound for 'Ok' category
        output_file: Path to save the output plot

    Returns:
        None
    """
    try:
        # Calculate figure dimensions based on data extent
        x_range = cal_pts_df['E'].max() - cal_pts_df['E'].min()
        y_range = cal_pts_df['N'].max() - cal_pts_df['N'].min()
        aspect_ratio = x_range / y_range
        
        # Set figure dimensions
        base_size = 20
        if aspect_ratio > 1:
            fig_width = base_size
            fig_height = base_size / aspect_ratio
        else:
            fig_width = base_size * aspect_ratio
            fig_height = base_size
            
        # Create figure with adjusted margins
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0, right=0.95, top=1, bottom=0)
        
        # Calculate marker size based on data density
        total_points = len(cal_pts_df)
        if total_points > 10000:
            logger.warning("Large dataset detected. Optimizing plot for large river scale.")
            marker_size = max(20, 100 * (10000 / total_points))
        else:
            marker_size = 100

        # Prepare data for plotting
        plot_df = cal_pts_df.copy()
        plot_df['ColorCat'] = pd.cut(
            plot_df['Difference'],
            bins=[-float('inf'), l_bounds, u_bounds, float('inf')],
            labels=['Low', 'Ok', 'High']
        )
        plot_df = plot_df[plot_df['ColorCat'].isin(['Low', 'High'])]
        
        # Plot centerline
        centerline_gdf.plot(
            ax=ax,
            color='black',
            linewidth=1,
            alpha=0.5,
            label='Centerline'
        )

        # Plot calibration points
        sns.scatterplot(
            data=plot_df,
            x='E',
            y='N',
            edgecolor='black',
            hue='ColorCat',
            palette={'Low': 'blue', 'High': 'red'},
            hue_order=['Low', 'High'],
            ax=ax,
            s=marker_size,
            alpha=0.7
        )

        # Configure plot appearance
        ax.set_aspect('equal', adjustable='box')
        ax.legend(
            title='Categories',
            loc='upper right',
            fontsize='large',
            title_fontsize='15',
            frameon=True,
            facecolor='white',
            edgecolor='black'
        )

        # Add threshold text
        plt.text(0.5, 0.04, f"Ok = + or - {u_bounds} ft.",
                ha='center', va='bottom', fontsize=14, transform=ax.transAxes)

        # Set axis labels
        plt.xlabel('Easting (ft)', fontsize=14)
        plt.ylabel('Northing (ft)', fontsize=14)

        # Remove margins
        plt.margins(0)
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Save outlier points as shapefile
        try:
            gdf = gpd.GeoDataFrame(
                plot_df,
                geometry=gpd.points_from_xy(plot_df['E'], plot_df['N']),
                crs=centerline_gdf.crs
            )
            shapefile_path = output_file.replace('.png', '_outliers.shp')
            gdf.to_file(shapefile_path)
            logger.info(f"Outlier points shapefile saved to {shapefile_path}")
        except Exception as e:
            logger.error(f"Error saving shapefile: {str(e)}")

        logger.info(f"Calibration plot saved to {output_file}")
        logger.info(f"Number of NaN values in Difference: {cal_pts_df['Difference'].isna().sum()}")

    except Exception as e:
        logger.error(f"Error creating calibration plot: {str(e)}")
        raise

def plot_wse_comparison(
    sampled_points_gdf: pd.DataFrame,
    output_file: str
) -> None:
    """
    Plot a comparison of measured WSE vs. model WSE.

    Args:
        sampled_points_gdf: DataFrame containing stationing and WSE data
        output_file: Path to save the output PNG file

    Returns:
        None
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot measured and model WSE
        ax.scatter(
            sampled_points_gdf['Stationing'],
            sampled_points_gdf['Z'],
            color='blue',
            label='Surveyed Depth',
            marker='o',
            s=10
        )
        
        ax.scatter(
            sampled_points_gdf['Stationing'],
            sampled_points_gdf['Sampled_Value'],
            color='green',
            label='Model Results',
            marker='s',
            s=10
        )

        # Calculate and display statistics
        avg_diff = sampled_points_gdf['Difference'].abs().mean()
        rmse = np.sqrt(np.mean(sampled_points_gdf['Difference_Squared']))
        p_value = sampled_points_gdf['Z'].corr(sampled_points_gdf['Sampled_Value'])

        ax.text(0.5, 1.10, f"Absolute Average Difference = {avg_diff:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 1.04, f"RMSE = {rmse:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 1.07, f"Pearson = {p_value:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

        # Configure plot appearance
        ax.set_title('WSE Comparison: Measured vs. Model')
        ax.set_xlabel('Stationing (ft.)')
        ax.set_ylabel('WSE (ft.)')
        ax.legend()
        ax.grid()
        
        # Format axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Reverse x-axis
        ax.invert_xaxis()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"WSE comparison plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Error creating WSE comparison plot: {str(e)}")
        raise
