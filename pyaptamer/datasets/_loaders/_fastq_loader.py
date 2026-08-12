__author__ = ["aditi-dsi"]
__all__ = ["load_sample_fastq"]

import os


def load_sample_fastq():
    """Load the sample FASTQ file as a MoleculeLoader.

    Returns
    -------
    loader : MoleculeLoader
        A MoleculeLoader object representing the sample FASTQ data.
    """
    from pyaptamer.data.loader import MoleculeLoader

    fastq_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.fastq")

    return MoleculeLoader(data={"sequence": [fastq_path]}, tiling="samples")
