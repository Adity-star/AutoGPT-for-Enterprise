 # Logging framework

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Constants
LOG_DIR = 'logs'
LOG_FILE_TIMESTAMP = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of rotated log files to keep

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_log_file_path(filename_prefix: str = "app", use_timestamp: bool = True) -> str:
    filename = f"{filename_prefix}_{LOG_FILE_TIMESTAMP}.log" if use_timestamp else f"{filename_prefix}.log"
    return os.path.join(LOG_DIR, filename)


# ANSI escape sequences for colors
class LogColors:
    RESET = "\x1b[0m"
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"

# Mapping logging levels to colors
LOG_LEVEL_COLORS = {
    logging.DEBUG: LogColors.CYAN,
    logging.INFO: LogColors.GREEN,
    logging.WARNING: LogColors.YELLOW,
    logging.ERROR: LogColors.RED,
    logging.CRITICAL: LogColors.MAGENTA,
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        record.msg = f"{color}{record.msg}{LogColors.RESET}"
        return super().format(record)


def configure_logger(
    logger_name: str = "",
    level: int = logging.DEBUG,
    log_filename: str = None
) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    colored_formatter = ColoredFormatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # Setup file handler (plain, no color)
    if log_filename is None:
        log_filename = get_log_file_path()
    file_handler = RotatingFileHandler(log_filename, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Setup console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



logger = configure_logger()
logger.info("Logger is configured and ready.")
