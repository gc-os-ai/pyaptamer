__author__ = "satvshr"
__all__ = ["hf_to_dataset"]

import os

from datasets import load_dataset

# File formats not natively supported by `datasets.load_dataset`
FILE_FORMATS = ["fasta", "pdb"]


def hf_to_dataset(path, keep_in_memory=True, **kwargs):
    """
    Load any Hugging Face dataset or file into a `datasets.Dataset`.

    This function first attempts to load the dataset natively using
    `datasets.load_dataset`. If the dataset format is unsupported, it falls back
    to loading the file as plain text via the "text" dataset loader.

    Parameters
    ----------
    path : str
        Path or identifier for the dataset. Can be:

          - A Hugging Face Hub dataset name (e.g. "imdb", "username/dataset_name").
          - A local dataset path.
          - A URL to a dataset file.

    keep_in_memory : bool, default=True
        Whether to keep the dataset in memory instead of writing to disk cache.
    **kwargs : dict
        Additional keyword arguments passed to `datasets.load_dataset`.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict

        - If the source provides multiple splits (e.g. "train", "test"),
          a `datasets.DatasetDict` is returned.
        - If only a single split is present, the corresponding
          `datasets.Dataset` is returned directly (this function unwraps
          single-split DatasetDicts automatically).

    """
    ext = os.path.splitext(str(path))[-1].lstrip(".").lower()

    if ext not in FILE_FORMATS:
        ds = load_dataset(path, keep_in_memory=keep_in_memory, **kwargs)
    else:
        ds = load_dataset(
            "text",
            data_files=path,
            keep_in_memory=keep_in_memory,
            **kwargs,
        )

    # Unwrap single-split DatasetDict into Dataset
    if isinstance(ds, dict) and len(ds) == 1:
        ds = next(iter(ds.values()))

    return ds
