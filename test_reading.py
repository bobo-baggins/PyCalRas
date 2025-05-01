import pandas as pd
import os
import pytest
from reading import load_calibration_points, find_matching_raster
import tempfile
import shutil

def create_test_csv(filename, columns, data):
    """Helper function to create test CSV files"""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    return filename

def create_test_raster_dir():
    """Helper function to create a temporary directory with test raster files"""
    temp_dir = tempfile.mkdtemp()
    # Create test raster files
    test_files = [
        'WSE_2023.tif',
        'wse_2024.tif',  # lowercase test
        'Depth_2023.tif',
        'Velocity_2023.tif',
        'WSE_2023_2.tif',  # multiple WSE files
        'invalid.txt'  # non-tif file
    ]
    for file in test_files:
        with open(os.path.join(temp_dir, file), 'w') as f:
            f.write('dummy content')
    return temp_dir

def test_find_matching_raster():
    """Test the find_matching_raster function"""
    # Create temporary directory with test files
    temp_dir = create_test_raster_dir()
    
    try:
        # Test case 1: Find WSE raster (case insensitive)
        wse_file = find_matching_raster(temp_dir, 'WSE')
        assert wse_file is not None
        assert os.path.basename(wse_file) == 'WSE_2023.tif'
        
        # Test case 2: Find lowercase WSE raster
        wse_file_lower = find_matching_raster(temp_dir, 'wse')
        assert wse_file_lower is not None
        assert os.path.basename(wse_file_lower) == 'WSE_2023.tif'
        
        # Test case 3: Find Depth raster
        depth_file = find_matching_raster(temp_dir, 'Depth')
        assert depth_file is not None
        assert os.path.basename(depth_file) == 'Depth_2023.tif'
        
        # Test case 4: Find Velocity raster
        velocity_file = find_matching_raster(temp_dir, 'Velocity')
        assert velocity_file is not None
        assert os.path.basename(velocity_file) == 'Velocity_2023.tif'
        
        # Test case 5: No matching raster
        no_file = find_matching_raster(temp_dir, 'Invalid')
        assert no_file is None
        
        # Test case 6: Multiple matches (should use first match)
        # Note: This test might be flaky due to filesystem ordering
        # We just verify it returns a valid file
        wse_file_multi = find_matching_raster(temp_dir, 'WSE')
        assert wse_file_multi is not None
        assert wse_file_multi.endswith('.tif')
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_find_matching_raster_empty_dir():
    """Test find_matching_raster with an empty directory"""
    temp_dir = tempfile.mkdtemp()
    try:
        result = find_matching_raster(temp_dir, 'WSE')
        assert result is None
    finally:
        shutil.rmtree(temp_dir)

def test_find_matching_raster_invalid_dir():
    """Test find_matching_raster with an invalid directory"""
    with pytest.raises(FileNotFoundError):
        find_matching_raster('nonexistent_directory', 'WSE')

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
        df, data_types = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'WSE' in df.columns
        assert data_types == ['WSE']
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
        df, data_types = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'Depth' in df.columns
        assert data_types == ['Depth']
    finally:
        os.remove(filename)

    # Test case 3: Valid file with multiple data types
    columns = ['Point', 'Northing', 'Easting', 'WSE', 'Depth', 'Velocity']
    data = {
        'Point': ['P1', 'P2', 'P3'],
        'Northing': [100, 101, 102],
        'Easting': [200, 201, 202],
        'WSE': [50, 51, 52],
        'Depth': [5, 6, 7],
        'Velocity': [2, 3, 4]
    }
    filename = create_test_csv('test_multiple.csv', columns, data)
    
    try:
        df, data_types = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'WSE' in df.columns
        assert 'Depth' in df.columns
        assert 'Velocity' in df.columns
        assert set(data_types) == {'WSE', 'Depth', 'Velocity'}
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
        df, data_types = load_calibration_points(filename)
        assert 'P' in df.columns
        assert 'N' in df.columns
        assert 'E' in df.columns
        assert 'WSE' in df.columns
        assert data_types == ['WSE']
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
    df, data_types = load_calibration_points(filepath)
    print("\nFirst 5 rows of the data:")
    print(df.head())
    print("\nColumn names after processing:")
    print(df.columns.tolist())
    print("\nAvailable data types:")
    print(data_types)

if __name__ == '__main__':
    # Run all tests
    test_valid_calibration_points()
    test_invalid_calibration_points()
    test_case_insensitivity()
    test_actual_calibration_points()
    test_find_matching_raster()
    test_find_matching_raster_empty_dir()
    test_find_matching_raster_invalid_dir()
    print("All tests passed!") 