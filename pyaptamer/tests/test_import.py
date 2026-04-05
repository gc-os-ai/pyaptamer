"""Tests for package import behavior."""

import subprocess
import sys


def test_import_from_source_checkout_without_installed_metadata():
    """Importing pyaptamer from a checkout should not require wheel metadata."""
    # This forces the "no installed metadata" path even in CI where the package is
    # installed. The patch happens before importing pyaptamer, so the module-level
    # `from importlib.metadata import version` picks up the patched function.
    python_code = "\n".join(
        [
            "import importlib.metadata as m",
            "orig = m.version",
            "def fake(name):",
            "    if name == 'pyaptamer':",
            "        raise m.PackageNotFoundError(name)",
            "    return orig(name)",
            "m.version = fake",
            "import pyaptamer",
            "print(pyaptamer.__version__)",
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            python_code,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0+unknown"
