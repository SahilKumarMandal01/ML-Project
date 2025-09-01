import logging
import os
from datetime import datetime


def setup_logging() -> str:
    """
    Configure logging for the application.
    
    Creates a timestamped log file inside a `logs` directory.
    
    Returns:
        str: Path to the log file.
    """
    # Generate log file name with timestamp
    log_filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.log")

    # Ensure logs directory exists
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Full path for the log file
    log_file_path = os.path.join(logs_dir, log_filename)

    # Configure logging
    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "[ %(asctime)s ] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(console_handler)

    return log_file_path


# Run setup immediately when imported
LOG_FILE_PATH = setup_logging()
