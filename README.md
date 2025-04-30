# PyCalRas

A Python tool for calibrating raster data against survey points.

## Features

- Processes calibration points against raster data
- Generates calibration plots and WSE comparison plots
- Creates shapefiles of outlier points
- Tracks calibration runs with statistics
- Supports test mode for validation

## Directory Structure

```
PyCalRas/
├── Inputs/
│   ├── Sample_Points/    # CSV files with calibration points
│   ├── Rasters/         # TIF files for processing
│   └── Alignment/       # Shapefiles for centerline
├── Outputs/
│   ├── calibration_runs.csv  # Log of all calibration runs
│   └── run_name/            # Individual run outputs
│       ├── description.txt
│       ├── raster_name.png
│       ├── raster_name_wse.png
│       ├── raster_name.csv
│       └── outliers/
│           └── raster_name_outliers.shp
└── src/
    ├── main.py
    ├── reading.py
    ├── calculations.py
    ├── output.py
    └── logger_config.py
```

## Input Requirements

### Sample Points (CSV)
- Required columns: P, N, E, Z, D
- Coordinates in feet
- First CSV file in Sample_Points directory will be used

### Raster (TIF)
- GeoTIFF format
- First TIF file in Rasters directory will be used

### Centerline (SHP)
- Shapefile format
- First SHP file in Alignment directory will be used

## Usage

1. Place input files in their respective directories under `Inputs/`
2. Run the script:
   ```bash
   python main.py
   ```
3. Enter a name and description for the calibration run when prompted
4. View results in the `Outputs/run_name/` directory

### Test Mode

To run in test mode with output validation:
```bash
python main.py --test
```

## Output Files

### Calibration Plot (PNG)
- Shows high and low points relative to the centerline
- Points are colored blue (low) or red (high)
- Includes threshold information

### WSE Comparison Plot (PNG)
- Compares measured vs. model WSE
- Includes RMSE and average difference statistics

### Results CSV
- Contains all point data with calculated differences
- Includes stationing and sampled values

### Outlier Shapefile
- Contains only high and low points
- Stored in the outliers subdirectory
- Can be used for further analysis in GIS software

### Run Log (CSV)
- Tracks all calibration runs
- Includes:
  - Run name and description
  - RMSE (3 significant figures)
  - Average difference (3 significant figures)
  - Timestamp
- Sorted by most recent first
- Duplicate run names overwrite previous entries

## Notes

- The script will automatically use the first file of each type found in the input directories
- Run names must be valid for directory names (no special characters)
- Existing run directories will prompt for overwrite confirmation
- Test mode validates output against reference files in `Outputs/test/`

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended) or pip
- GDAL and other geospatial libraries (required)

### Installation Steps

#### Option 1: Using Conda (Recommended)

1. Install Miniconda or Anaconda if you haven't already:
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - [Anaconda](https://www.anaconda.com/products/distribution)

2. Clone the repository:
```bash
git clone https://github.com/yourusername/pycalras.git
cd pycalras
```

3. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate pyCal
```

#### Option 2: Using pip (Not Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pycalras.git
cd pycalras
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install the package:
```bash
pip install -e .
```

Note: Installing GDAL and other geospatial dependencies through pip can be challenging. We strongly recommend using Conda instead.

## Code Structure

### Module Overview
- `main.py`: Main execution script orchestrating the workflow
- `reading.py`: Handles file input and data loading
- `calculations.py`: Performs computational analysis
- `output.py`: Generates visualizations and results
- `GeoDataPro.py`: Provides geospatial utilities

### Key Functions
- Data Loading: CSV points, TIF rasters, SHP centerlines
- Spatial Analysis: Raster sampling, stationing calculation
- Visualization: Calibration plots, WSE comparisons

### Data Flow
1. Load input files (reading.py)
2. Process spatial data (GeoDataPro.py)
3. Perform calculations (calculations.py)
4. Generate visualizations (output.py)

## Testing

The package includes a testing framework to validate output consistency. This is particularly useful when making code changes to ensure results remain consistent.

### Test Files
- Test files are stored in the `Outputs/test/` directory
- Each test file should correspond to an output file with the same name
- Test files contain reference values for validation

### Running Tests
To run the script with output validation:
```bash
python Main.py --test
```

Without the `--test` flag, validation is skipped:
```bash
python Main.py
```

### Test Validation
- The validation compares the `Sampled_Value` column in the output CSV with the reference test file
- Results must match within 1% tolerance
- Validation results are logged to the console

### Setting Up Tests
1. Create the test directory:
```bash
mkdir -p Outputs/test
```

2. Copy a known good output CSV to use as a reference:
```bash
cp Outputs/your_output.csv Outputs/test/your_output.csv
```

## Dependencies

The package requires several geospatial libraries that are best installed through Conda:
- GDAL
- GeoPandas
- PyOGRIO
- Fiona

These dependencies are automatically installed when using the Conda environment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
