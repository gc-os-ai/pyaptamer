"""Tests for package import behavior."""

import subprocess
import sys


def test_import_from_source_checkout_without_installed_metadata():
    """Importing pyaptamer from a checkout should not require wheel metadata."""
    result = subprocess.run(
        [sys.executable, "-c", "import pyaptamer; print(pyaptamer.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
