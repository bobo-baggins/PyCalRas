# PyCalRas

A Python package for calibration and raster analysis.

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

## Directory Structure

The package expects the following directory structure:
```
.
├── Main.py
├── Calculations.py
├── GeoDataPro.py
├── Output.py
├── Inputs/
│   ├── Sample_Points/  # Contains CSV files with calibration points
│   ├── Rasters/       # Contains TIF raster files
│   └── Alignment/     # Contains SHP alignment files
└── Outputs/           # Generated output files will be saved here
```

## Usage

1. Place your input files in the appropriate directories:
   - Calibration points CSV files in `Inputs/Sample_Points/`
   - Raster TIF files in `Inputs/Rasters/`
   - Alignment SHP files in `Inputs/Alignment/`

2. Run the main script:
```bash
python Main.py
```

3. Check the `Outputs/` directory for generated files.

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

## Input File Requirements

### Calibration Points CSV
- Required columns: P, N, E, Z, D
- CSV format

### Raster Files
- TIF format
- Georeferenced

### Alignment Files
- SHP format
- Contains centerline data

## Dependencies

The package requires several geospatial libraries that are best installed through Conda:
- GDAL
- GeoPandas
- PyOGRIO
- Fiona

These dependencies are automatically installed when using the Conda environment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
