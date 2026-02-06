"""
Centralized logging configuration for Minigrid2Storm project.

This module provides a simple logging setup with console and file output.
Call setup_logging() once at the start of your application.
"""

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "minigrid2storm.log",
    log_level: int = logging.INFO,
    console_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 3
):
    """
    Setup logging configuration for the project.
    
    Args:
        log_dir: Directory where log files will be stored
        log_file: Name of the main log file
        log_level: Minimum level for file logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: Minimum level for console output
        max_bytes: Maximum size of log file before rotation (default 10MB)
        backup_count: Number of backup files to keep
        
    Returns:
        logging.Logger: The root logger instance
        
    Example:
        >>> from logging_config import setup_logging
        >>> setup_logging()
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - less verbose for readability
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation - detailed for debugging
    file_handler = RotatingFileHandler(
        filename=log_path / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Log the initialization
    root_logger.info(f"Logging initialized - File: {log_path / log_file}, Level: {logging.getLevelName(log_level)}")
    
    return root_logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name for the logger (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
    """
    return logging.getLogger(name)


# Optional: Pre-configured loggers for specific components
# Uncomment and customize these when you want to split logs later

# def setup_training_logger():
#     """Setup a separate logger for training with its own file."""
#     logger = logging.getLogger('training')
#     handler = RotatingFileHandler('logs/training.log', maxBytes=10*1024*1024, backupCount=3)
#     handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
#     return logger

# def setup_simulation_logger():
#     """Setup a separate logger for simulations with its own file."""
#     logger = logging.getLogger('simulation')
#     handler = RotatingFileHandler('logs/simulation.log', maxBytes=10*1024*1024, backupCount=3)
#     handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#     logger.addHandler(handler)
#     logger.setLevel(logging.DEBUG)
#     return logger
