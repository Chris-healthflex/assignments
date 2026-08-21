import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure structured logging with a simple format."""
    logger = logging.getLogger("clinical_assessment")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    file_handler = RotatingFileHandler("app.log", maxBytes=10_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()