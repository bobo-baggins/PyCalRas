import os
import logging
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        name: Name of the logger (usually __name__ from the calling module)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    log_file = os.path.join(log_dir, f'pycalras_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Add handlers if they haven't been added already
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger 