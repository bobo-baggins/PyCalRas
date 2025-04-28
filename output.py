import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_calibration_plot(
    cal_pts_df: pd.DataFrame,
    l_bounds: float = -0.5,
    u_bounds: float = 0.5,
    output_file: str = ''
) -> None:
    """
    Create a calibration plot based on the provided data.

    Args:
        cal_pts_df: DataFrame containing calibration points data
        l_bounds: Lower bound for 'Ok' category
        u_bounds: Upper bound for 'Ok' category
        output_file: Path to save the output plot

    Returns:
        None
    """
    try:
        # Check if dataset is too large
        if len(cal_pts_df) > 10000:
            logger.warning("Large dataset detected. Reducing plot complexity for memory efficiency.")
            # Reduce figure size
            fig, ax = plt.subplots(figsize=(8, 8))
            # Reduce marker size
            marker_size = 50
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            marker_size = 100

        # Categorize differences
        cal_pts_df['ColorCat'] = pd.cut(
            cal_pts_df['Difference'],
            bins=[-float('inf'), l_bounds, u_bounds, float('inf')],
            labels=['Low', 'Ok', 'High']
        )
        palette = {'Low': 'blue', 'Ok': 'White', 'High': 'red'}

        # Create scatter plot with reduced complexity
        sns.scatterplot(
            data=cal_pts_df,
            x='E',
            y='N',
            edgecolor='black',
            hue='ColorCat',
            palette=palette,
            ax=ax,
            s=marker_size,
            alpha=0.7
        )

        # Set aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Only add text labels for significant outliers
        significant_outliers = cal_pts_df[
            (cal_pts_df['ColorCat'].isin(['Low', 'High'])) & 
            (abs(cal_pts_df['Difference']) > 1.0)
        ]
        
        if len(significant_outliers) > 0:
            texts = []
            for idx, row in significant_outliers.iterrows():
                text = ax.annotate(
                    f"{row['Difference']:.1f}",
                    xy=(row['E'], row['N']),
                    xytext=(row['E'], row['N']),
                    fontsize=8,
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', shrinkA=1),
                    fontfamily='Arial'
                )
                texts.append(text)

            # Adjust text positions with reduced iterations
            adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'}, max_iterations=100)

        # Add legend
        ax.legend(
            title='Categories',
            loc='upper right',
            fontsize='medium',
            title_fontsize='13',
            frameon=True,
            facecolor='white',
            edgecolor='black'
        )

        # Add title and statistics
        plt.title(output_file.replace('Outputs/', '') + ": Grade Check")
        avg_diff = cal_pts_df['Difference'].mean()
        rmse = np.sqrt(np.mean(cal_pts_df['Difference_Squared']))
        p_value = cal_pts_df['Z'].corr(cal_pts_df['Sampled_Value'])

        plt.text(0.5, 0.04, f"Ok = + or - {u_bounds} ft.",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
        plt.text(0.5, 1.08, f"Average Difference = {avg_diff:.2f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

        # Set labels
        plt.xlabel('Easting (ft)')
        plt.ylabel('Northing (ft)')

        # Save plot with reduced DPI for large datasets
        dpi = 150 if len(cal_pts_df) > 10000 else 300
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()

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
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot measured WSE
        ax.scatter(
            sampled_points_gdf['Stationing'],
            sampled_points_gdf['Z'],
            color='blue',
            label='Surveyed Depth',
            marker='o',
            s=10
        )
        
        # Plot model results
        ax.scatter(
            sampled_points_gdf['Stationing'],
            sampled_points_gdf['Sampled_Value'],
            color='green',
            label='Model Results',
            marker='s',
            s=10
        )

        # Calculate statistics
        avg_diff = sampled_points_gdf['Difference'].abs().mean()
        rmse = np.sqrt(np.mean(sampled_points_gdf['Difference_Squared']))
        p_value = sampled_points_gdf['Z'].corr(sampled_points_gdf['Sampled_Value'])

        # Add statistics to plot
        ax.text(0.5, 1.10, f"Average Residual = {avg_diff:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 1.04, f"RMSE = {rmse:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 1.07, f"Pearson = {p_value:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

        # Set plot properties
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
