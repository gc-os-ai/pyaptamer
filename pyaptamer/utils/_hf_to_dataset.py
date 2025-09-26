__author__ = "satvshr"
__all__ = ["hf_to_dataset"]

from datasets import load_dataset


def hf_to_dataset(path, keep_in_memory=True, **kwargs):
    """
    Load any Hugging Face dataset or file into a `datasets.Dataset`.

    This function first attempts to load the dataset natively using
    `datasets.load_dataset`. If the dataset format is unsupported, it falls back
    to loading the file(s) as plain text via the "text" dataset loader.

    Parameters
    ----------
    path : str or dict
        Path or identifier for the dataset. Can be:

          - A Hugging Face Hub dataset name (e.g. "imdb").
          - A local dataset path.
          - A URL to a dataset file.
          - A dictionary of split-to-file mappings, e.g.
            `{"train": "train.txt", "test": "test.txt"}`.

    keep_in_memory : bool, default=True
        Whether to keep the dataset in memory instead of writing to disk cache.
    **kwargs : dict
        Additional keyword arguments passed to `datasets.load_dataset`.

    Returns
    -------
    datasets.Dataset
        A Hugging Face datasets.Dataset object.
    """
    try:
        ds = load_dataset(path, keep_in_memory=keep_in_memory, **kwargs)
    except Exception:
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
