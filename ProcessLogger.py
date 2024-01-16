
# Initiated on 240115 to log runs of FileTransfer-replacements

import logging

def log_init(directory):

    print("Initiating Logging...")

    logger = logging.getLogger(__name__)  # Get a logger for the current module
    logger.setLevel(logging.INFO)  # Set the logging level to INFO (captures INFO, WARNING, ERROR, and CRITICAL messages)

    # Create a file handler
    file_handler = logging.FileHandler(directory + "\\function_log.txt")  # Specify the log file name
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # Set the log message format
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  # Add the file handler to the logger

    return logger
