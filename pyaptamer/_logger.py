"""Centralized logger for pyaptamer."""

import logging

# Create the one global logger
logger = logging.getLogger("pyaptamer")
logger.setLevel(logging.INFO)

# Format it to show the calling module so we know where logs come from
formatter = logging.Formatter("[%(levelname)s] [%(module)s] %(message)s")

# Prevent adding multiple handlers if the module is reloaded
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Prevent log propagation to the root logger to avoid duplicate logs
logger.propagate = False
