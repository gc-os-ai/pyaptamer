"""pyaptamer: Python library for aptamer design."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyaptamer")
except PackageNotFoundError:
    __version__ = "0+unknown"
