import os
import tempfile
import shutil
import pandas as pd
import pytest
from output import create_run_comparison_plot, create_interactive_run_comparison_plot
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

def create_test_run_data(temp_dir: str, run_name: str, stationing: list, wse: list, sampled_values: list) -> None:
    """Helper function to create test run data"""
    run_dir = os.path.join(temp_dir, 'Outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create test data
    data = {
        'Stationing': stationing,
        'WSE': wse,
        'Sampled_Value': sampled_values
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = os.path.join(run_dir, f'WSE_results.csv')
    df.to_csv(output_file, index=False)

def test_create_run_comparison_plot():
    """Test the create_run_comparison_plot function using actual data"""
    # Get the project root directory (assuming test is in project root)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(project_dir, 'Outputs')
    
    # Log the directories we're using
    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Outputs directory: {outputs_dir}")
    
    # Verify Outputs directory exists
    assert os.path.exists(outputs_dir), f"Outputs directory not found at {outputs_dir}"
    
    # List the run directories
    run_dirs = [d for d in os.listdir(outputs_dir) 
                if os.path.isdir(os.path.join(outputs_dir, d)) 
                and d != 'test']
    logger.info(f"Found run directories: {run_dirs}")
    
    # Create output file in the Outputs directory
    output_file = os.path.join(outputs_dir, 'run_comparison_plot.png')
    
    # Call the function with actual data
    create_run_comparison_plot(project_dir, output_file)
    
    # Verify the output file was created
    assert os.path.exists(output_file), f"Output plot file was not created at {output_file}"
    
    # Verify the file is not empty
    assert os.path.getsize(output_file) > 0, "Output plot file is empty"

def test_create_interactive_run_comparison_plot():
    """Test the create_interactive_run_comparison_plot function using actual data"""
    # Get the project root directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(project_dir, 'Outputs')
    
    # Create output file in the Outputs directory
    output_file = os.path.join(outputs_dir, 'run_comparison_plot.html')
    
    # Call the function with actual data
    create_interactive_run_comparison_plot(project_dir, output_file)
    
    # Verify the output file was created
    assert os.path.exists(output_file), f"Interactive plot file was not created at {output_file}"
    
    # Verify the file is not empty
    assert os.path.getsize(output_file) > 0, "Interactive plot file is empty"

def test_create_run_comparison_plot_no_runs():
    """Test create_run_comparison_plot with no run data"""
    temp_dir = tempfile.mkdtemp()
    outputs_dir = os.path.join(temp_dir, 'Outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    try:
        output_file = os.path.join(temp_dir, 'comparison_plot.png')
        
        # Call the function with no run data
        create_run_comparison_plot(temp_dir, output_file)
        
        # Verify the output file was created (should create an empty plot)
        assert os.path.exists(output_file), "Output plot file was not created"
        
    finally:
        shutil.rmtree(temp_dir)

def test_create_run_comparison_plot_invalid_dir():
    """Test create_run_comparison_plot with invalid directory"""
    with pytest.raises(Exception):
        create_run_comparison_plot('nonexistent_directory', 'output.png')

if __name__ == '__main__':
    # Run both tests
    print("Running run comparison plot tests...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test static plot
    output_file = os.path.join(project_dir, 'Outputs', 'run_comparison_plot.png')
    create_run_comparison_plot(project_dir, output_file, show_plot=True)
    print("Static plot test completed successfully!")
    
    # Test interactive plot
    html_file = os.path.join(project_dir, 'Outputs', 'run_comparison_plot.html')
    create_interactive_run_comparison_plot(project_dir, html_file)
    print("Interactive plot test completed successfully!") 