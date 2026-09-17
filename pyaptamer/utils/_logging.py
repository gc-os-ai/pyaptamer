"""Logging Configurtion for pyaptamer."""

import logging
import sys

__all__ = ["get_logger"]

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a Configured logger instance
    
    Parameters
    ----------
    name: str
        Logger name (typically __name__).
    level: int
        Logging level (default: INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter =logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger