__author__ = ["aditi-dsi", "siddharth7113"]
__all__ = ["load_sample_fastq"]

import os


def load_sample_fastq():
    """Load the sample FASTQ file as a MoleculeLoader.

    The loader is built with ``tiling="samples"``, so each read of the file
    becomes one row.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the sample FASTQ data.

    Raises
    ------
    FileNotFoundError
        If the sample FASTQ file does not exist.
    """
    from pyaptamer.data.loader import MoleculeLoader

    fastq_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample.fastq")
    )

    if not os.path.exists(fastq_path):
        raise FileNotFoundError(
            f"Sample FASTQ not found at {fastq_path}. Please ensure the file exists."
        )

    return MoleculeLoader(data={"sequence": [fastq_path]}, tiling="samples")
