import pandas as pd
import os
import pytest
from reading import load_calibration_points

def create_test_csv(filename, columns, data):
    """Helper function to create test CSV files"""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    return filename

def test_valid_calibration_points():
    # Test case 1: Valid file with WSE
    columns = ['Point', 'Northing', 'Easting', 'WSE']
    data = {
        'Point': ['P1', 'P2', 'P3'],
        'Northing': [100, 101, 102],
        'Easting': [200, 201, 202],
        'WSE': [50, 51, 52]
    }
    filename = create_test_csv('test_wse.csv', columns, data)
    
    try:
        df = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'WSE' in df.columns
    finally:
        os.remove(filename)

    # Test case 2: Valid file with Depth
    columns = ['Point', 'Northing', 'Easting', 'Depth']
    data = {
        'Point': ['P1', 'P2', 'P3'],
        'Northing': [100, 101, 102],
        'Easting': [200, 201, 202],
        'Depth': [5, 6, 7]
    }
    filename = create_test_csv('test_depth.csv', columns, data)
    
    try:
        df = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'Depth' in df.columns
    finally:
        os.remove(filename)

def test_invalid_calibration_points():
    # Test case 1: Missing required columns
    columns = ['Point', 'Northing', 'WSE']
    data = {
        'Point': ['P1', 'P2', 'P3'],
        'Northing': [100, 101, 102],
        'WSE': [50, 51, 52]
    }
    filename = create_test_csv('test_missing.csv', columns, data)
    
    try:
        with pytest.raises(ValueError, match="Could not find required columns"):
            load_calibration_points(filename)
    finally:
        os.remove(filename)

    # Test case 2: No data type column
    columns = ['Point', 'Northing', 'Easting']
    data = {
        'Point': ['P1', 'P2', 'P3'],
        'Northing': [100, 101, 102],
        'Easting': [200, 201, 202]
    }
    filename = create_test_csv('test_no_data.csv', columns, data)
    
    try:
        with pytest.raises(ValueError, match="Could not find any data type columns"):
            load_calibration_points(filename)
    finally:
        os.remove(filename)

def test_case_insensitivity():
    # Test case: Different capitalizations
    columns = ['pOiNt', 'nOrThInG', 'eAsTiNg', 'wSe']
    data = {
        'pOiNt': ['P1', 'P2', 'P3'],
        'nOrThInG': [100, 101, 102],
        'eAsTiNg': [200, 201, 202],
        'wSe': [50, 51, 52]
    }
    filename = create_test_csv('test_case.csv', columns, data)
    
    try:
        df = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'WSE' in df.columns
    finally:
        os.remove(filename)

def test_actual_calibration_points():
    """Test reading the actual calibration points file"""
    # Get the path to the sample points file
    executable_dir = os.path.dirname(os.path.abspath(__file__))
    sample_points_dir = os.path.join(executable_dir, 'Inputs', 'Sample_Points')
    
    # Find the first CSV file in the directory
    csv_files = [f for f in os.listdir(sample_points_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Sample_Points directory")
    
    filepath = os.path.join(sample_points_dir, csv_files[0])
    print(f"\nReading file: {csv_files[0]}")
    
    # Load and display the data
    df = load_calibration_points(filepath)
    print("\nFirst 5 rows of the data:")
    print(df.head())
    print("\nColumn names after processing:")
    print(df.columns.tolist())

if __name__ == '__main__':
    # Run all tests
    test_valid_calibration_points()
    test_invalid_calibration_points()
    test_case_insensitivity()
    test_actual_calibration_points()
    print("All tests passed!") 