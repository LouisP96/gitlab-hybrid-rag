"""Unified logging setup for all scripts."""

import logging
import os
from pathlib import Path


def setup_script_logging(script_name: str) -> logging.Logger:
    """
    Set up logging for a script with both console and file output.

    Args:
        script_name: Name of the script (used for log file name)

    Returns:
        Logger instance configured for the script
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Set up logging with both file and console handlers
    log_file = logs_dir / f"{script_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True  # Override any existing configuration
    )

    return logging.getLogger(script_name)