import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def create_calibration_plot(cal_pts_df, l_bounds=-0.5, u_bounds=0.5, output_file='Outputs/Supply_WSE_Check.png'):
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
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(data=cal_pts_df, 
                    x='E', y='N', 
                    edgecolor='black',
                    hue='ColorCat', 
                    palette=palette,
                    ax=ax)

    # Add text labels
    texts = []
    for i in range(cal_pts_df.shape[0]):
        if cal_pts_df['ColorCat'][i] in ['Low', 'High']:
            label_text = f"{cal_pts_df['Difference'][i]:.1f}"
            text = ax.annotate(label_text,
                          xy=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                          xytext=(cal_pts_df['E'][i], cal_pts_df['N'][i]),
                          fontsize=12,
                          color='black',
                          arrowprops=dict(arrowstyle='->', color='black', shrinkA=1),
                          fontfamily='Arial')
            texts.append(text)

    # Adjust text positions
    adjust_text(texts, 
                only_move={'points': 'xy', 'text': 'xy'},
                max_iterations=300)

    # Add legend
    ax.legend(title='Categories', 
              loc='upper right', 
              fontsize='medium', 
              title_fontsize='13', 
              frameon=True, 
              facecolor='white', 
              edgecolor='black')

    # Add title and subtitles
    plt.title('Supply 80p5cfs: WSE Calibration')
    calculated_value = np.average(cal_pts_df['Difference'])
    RMSE = np.sqrt(np.mean(cal_pts_df['Difference']**2))
    p_value = cal_pts_df['Z'].corr(cal_pts_df['Sampled_Value'])

    plt.text(0.5, 0.04, f"Ok = + or - {u_bounds} ft.", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    plt.text(0.5, 1.06, f"Average Difference = {calculated_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    plt.text(0.5, 1.08, f"RMSE = {RMSE:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)
    plt.text(0.5, 1.10, f"Pearson = {p_value:.2f}", ha='center', va='bottom', fontsize=12, transform=ax.transAxes)

    # Set labels
    plt.xlabel('Easting (ft)')
    plt.ylabel('Northing (ft)')

    # Save and close
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Plot saved to {output_file}")
    print(f"Number of NaN values in Difference: {cal_pts_df['Difference'].isna().sum()}")
