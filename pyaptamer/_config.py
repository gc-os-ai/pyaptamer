"""Configuration manager for pyaptamer."""

import logging
from dataclasses import dataclass


@dataclass
class PyAptamerConfig:
    """Store library-wide configuration settings."""
    log_level: int = logging.INFO


class PyAptamerConfigManager:
    """Manages the configuration and central logger for pyaptamer."""

    def __init__(self):
        self._config = PyAptamerConfig()
        self.logger = logging.getLogger("pyaptamer")
        self._setup_logging()

    def _setup_logging(self):
        """Configure the central logger with a default stream handler."""
        self.logger.setLevel(self._config.log_level)
        
        # Prevent log propagation to the root logger to avoid duplicate logs
        # if the user has their own root logger configured.
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # The format includes %(name)s (always "pyaptamer") and %(module)s 
            # to let the user know which file the log originated from.
            formatter = logging.Formatter(
                "[%(levelname)s] [%(name)s:%(module)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_log_level(self, level: int | str):
        """Set the log level for the pyaptamer library.
        
        Parameters
        ----------
        level : int or str
            The logging level (e.g., logging.DEBUG, "INFO").
        """
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        self._config.log_level = level
        self.logger.setLevel(level)


config = PyAptamerConfigManager()
