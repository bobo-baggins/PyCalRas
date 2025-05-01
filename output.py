# Standard library imports
from typing import Optional, Tuple, Dict
import os
import csv
from datetime import datetime

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

def create_run_directory(executable_dir: str, run_name: str, description: str) -> str:
    """
    Create a directory for the calibration run and save its description.
    
    Args:
        executable_dir: Base directory for the application
        run_name: Name of the calibration run
        description: Description of the calibration run
        
    Returns:
        str: Path to the created run directory
        
    Raises:
        OSError: If directory creation fails
    """
    run_dir = os.path.join(executable_dir, 'Outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create outliers subdirectory
    outliers_dir = os.path.join(run_dir, 'outliers')
    os.makedirs(outliers_dir, exist_ok=True)
    
    # Save run description
    with open(os.path.join(run_dir, 'description.txt'), 'w') as f:
        f.write(description)
    
    logger.info(f"Created run directory: {run_dir}")
    return run_dir

def get_output_paths(run_dir: str, raster_name: str, data_type: str) -> Tuple[str, str, str]:
    """
    Generate output file paths for a specific data type.
    
    Args:
        run_dir: Directory for the calibration run
        raster_name: Name of the raster file
        data_type: Type of data (WSE, Depth, or Velocity)
        
    Returns:
        Tuple[str, str, str]: (plot_file, comparison_plot_file, csv_file)
    """
    # Example: if raster_name is "WSE_2023" and data_type is "WSE"
    # This will create paths like:
    # - WSE_2023_plot.png
    # - WSE_2023_comparison.png
    # - WSE_2023_results.csv
    plot_file = os.path.join(run_dir, f"{raster_name}_{data_type}_plot.png")
    comparison_plot_file = plot_file.replace('_plot.png', '_comparison.png')
    csv_file = plot_file.replace('_plot.png', f'_{data_type}_results.csv')
    
    return plot_file, comparison_plot_file, csv_file

def get_shapefile_path(run_dir: str, raster_name: str) -> str:
    """
    Generate path for the outliers shapefile.
    
    Args:
        run_dir: Directory for the calibration run
        raster_name: Name of the raster file (without extension)
        
    Returns:
        str: Path to the shapefile
    """
    outliers_dir = os.path.join(run_dir, 'outliers')
    return os.path.join(outliers_dir, f"{raster_name}_outliers.shp")

def get_run_log_path(executable_dir: str) -> str:
    """
    Get the path to the calibration run log file.
    
    Args:
        executable_dir: Base directory for the application
        
    Returns:
        str: Path to the run log file
    """
    return os.path.join(executable_dir, 'Outputs', 'calibration_runs.csv')

def load_run_log(executable_dir: str) -> pd.DataFrame:
    """
    Load the calibration run log.
    
    Args:
        executable_dir: Base directory for the application
        
    Returns:
        pd.DataFrame: DataFrame containing run information
    """
    log_path = get_run_log_path(executable_dir)
    if os.path.exists(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception as e:
            logger.warning(f"Error reading run log file: {str(e)}. Creating new log.")
            return pd.DataFrame(columns=[
                'run_name', 
                'description', 
                'timestamp',
                'WSE_rmse', 'WSE_avg_diff',
                'Depth_rmse', 'Depth_avg_diff',
                'Velocity_rmse', 'Velocity_avg_diff'
            ])
    return pd.DataFrame(columns=[
        'run_name', 
        'description', 
        'timestamp',
        'WSE_rmse', 'WSE_avg_diff',
        'Depth_rmse', 'Depth_avg_diff',
        'Velocity_rmse', 'Velocity_avg_diff'
    ])

def save_run_log(executable_dir: str, run_log: pd.DataFrame) -> None:
    """
    Save the calibration run log.
    
    Args:
        executable_dir: Base directory for the application
        run_log: DataFrame containing run information
    """
    log_path = get_run_log_path(executable_dir)
    run_log.to_csv(log_path, index=False)

def update_run_log(
    executable_dir: str,
    run_name: str,
    description: str,
    rmse: float,
    avg_diff: float,
    data_type: str
) -> None:
    """
    Update the calibration run log with new run information.
    
    Args:
        executable_dir: Base directory for the application
        run_name: Name of the calibration run
        description: Description of the calibration run
        rmse: Root Mean Square Error
        avg_diff: Absolute Average Difference
        data_type: Type of data (WSE, Depth, or Velocity)
    """
    run_log = load_run_log(executable_dir)
    
    # Format numbers to three significant figures
    rmse_formatted = float(f"{rmse:.3g}")
    avg_diff_formatted = float(f"{avg_diff:.3g}")
    
    # Create new run entry or update existing one
    if run_name in run_log['run_name'].values:
        # Update existing entry
        mask = run_log['run_name'] == run_name
        run_log.loc[mask, f'{data_type}_rmse'] = rmse_formatted
        run_log.loc[mask, f'{data_type}_avg_diff'] = avg_diff_formatted
        run_log.loc[mask, 'description'] = description
        run_log.loc[mask, 'timestamp'] = datetime.now().isoformat()
    else:
        # Create new entry
        new_entry = pd.DataFrame([{
            'run_name': run_name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            f'{data_type}_rmse': rmse_formatted,
            f'{data_type}_avg_diff': avg_diff_formatted
        }])
        run_log = pd.concat([run_log, new_entry], ignore_index=True)
    
    # Sort by timestamp (most recent first)
    run_log = run_log.sort_values('timestamp', ascending=False)
    
    save_run_log(executable_dir, run_log)
    logger.info(f"Updated run log with {data_type} statistics for run: {run_name}")

def create_spatial_plot(
    cal_pts_df: pd.DataFrame,
    centerline_gdf: pd.DataFrame,
    l_bounds: float = -0.5,
    u_bounds: float = 0.5,
    output_file: str = '',
    data_type: str = ''
) -> Tuple[float, float]:
    """
    Create a spatial plot showing the distribution of differences across the river.

    Args:
        cal_pts_df: DataFrame containing calibration points data
        centerline_gdf: GeoDataFrame containing the centerline data
        l_bounds: Lower bound for 'Ok' category
        u_bounds: Upper bound for 'Ok' category
        output_file: Path to save the output plot
        data_type: Type of data (WSE, Depth, or Velocity)

    Returns:
        Tuple[float, float]: (rmse, average_difference)
    """
    try:
        # Calculate statistics
        rmse = np.sqrt(np.mean(cal_pts_df['Difference_Squared']))
        avg_diff = cal_pts_df['Difference'].abs().mean()
        
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

        # Add threshold text with units
        units = {
            'WSE': 'ft',
            'Depth': 'ft',
            'Velocity': 'ft/s'
        }
        plt.text(0.5, 0.04, f"Ok = + or - {u_bounds} {units[data_type]}",
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
            shapefile_path = get_shapefile_path(os.path.dirname(output_file), os.path.splitext(os.path.basename(output_file))[0])
            gdf.to_file(shapefile_path)
            logger.info(f"Outlier points shapefile saved to {shapefile_path}")
        except Exception as e:
            logger.error(f"Error saving shapefile: {str(e)}")

        logger.info(f"Calibration plot saved to {output_file}")
        logger.info(f"Number of NaN values in Difference: {cal_pts_df['Difference'].isna().sum()}")
        
        return rmse, avg_diff

    except Exception as e:
        logger.error(f"Error creating calibration plot: {str(e)}")
        raise

def create_centerline_plot(
    sampled_points_gdf: pd.DataFrame,
    output_file: str,
    data_type: str
) -> None:
    """
    Create a centerline plot comparing measured vs. model values along the river.

    Args:
        sampled_points_gdf: DataFrame containing stationing and data
        output_file: Path to save the output PNG file
        data_type: Type of data (WSE, Depth, or Velocity)

    Returns:
        None
    """
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        
        # Define units for each data type
        units = {
            'WSE': 'ft',
            'Depth': 'ft',
            'Velocity': 'ft/s'
        }
        
        # Sort points by stationing
        sorted_points = sampled_points_gdf.sort_values('Stationing')
        
        # Log difference statistics
        logger.info(f"Difference statistics for {data_type}:")
        logger.info(f"Min difference: {sorted_points['Difference'].min():.3f}")
        logger.info(f"Max difference: {sorted_points['Difference'].max():.3f}")
        logger.info(f"Mean difference: {sorted_points['Difference'].mean():.3f}")
        logger.info(f"Std difference: {sorted_points['Difference'].std():.3f}")
        
        # Plot measured values
        ax1.scatter(
            sorted_points['Stationing'],
            sorted_points[data_type],
            color='blue',
            label=f'Surveyed {data_type}',
            s=20,
            alpha=1.0,
            edgecolor='none'
        )
        
        # Plot model values
        ax1.scatter(
            sorted_points['Stationing'],
            sorted_points['Sampled_Value'],
            color='green',
            label='Model Results',
            s=20,
            alpha=0.75,
            edgecolor='green',
            facecolor='none',
            linewidth=1.5
        )

        # Calculate and display statistics
        avg_diff = sorted_points['Difference'].abs().mean()
        rmse = np.sqrt(np.mean(sorted_points['Difference_Squared']))
        p_value = sorted_points[data_type].corr(sorted_points['Sampled_Value'])

        ax1.text(0.5, 1.10, f"Absolute Average Difference = {avg_diff:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax1.transAxes)
        ax1.text(0.5, 1.04, f"RMSE = {rmse:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax1.transAxes)
        ax1.text(0.5, 1.07, f"Pearson = {p_value:.3f}",
                ha='center', va='bottom', fontsize=12, transform=ax1.transAxes)

        # Configure main plot appearance
        ax1.set_title(f'{data_type} Comparison: Measured vs. Model')
        ax1.set_ylabel(f'{data_type} ({units[data_type]})')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Format axis labels
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Plot differences in second subplot
        ax2.scatter(
            sorted_points['Stationing'],
            sorted_points['Difference'],
            color='red',
            s=20,
            alpha=0.6,
            edgecolor='none'
        )
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Configure difference plot
        ax2.set_xlabel('Stationing (ft.)')
        ax2.set_ylabel(f'Difference ({units[data_type]})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits for difference plot with some padding
        max_diff = sorted_points['Difference'].abs().max()
        min_diff = sorted_points['Difference'].min()
        max_diff = sorted_points['Difference'].max()
        padding = (max_diff - min_diff) * 0.1  # Add 10% padding
        ax2.set_ylim(min_diff - padding, max_diff + padding)
        
        # Format difference plot axis labels
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Dynamic decimal places based on difference range
        def format_difference(x, p):
            if abs(x) < 1:
                return f'{x:.1f}'
            return f'{x:.0f}'
        
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_difference))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"{data_type} comparison plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Error creating {data_type} comparison plot: {str(e)}")
        raise
