"""Logging configuration for the application."""
import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  Defaults to INFO if not specified.
    
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = "INFO"
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger("app")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs from uvicorn
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    
    return logger


# Create module-level logger
logger = setup_logging()

