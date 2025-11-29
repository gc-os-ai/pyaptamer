__author__ = "satvshr"
__all__ = ["load_hf_to_dataset"]

import os

import requests
from datasets import load_dataset

# File formats not natively supported by `datasets.load_dataset`
FILE_FORMATS = ["fasta", "pdb"]


def _download_to_cwd(url):
    """Download URL into ./hf_datasets/ preserving the filename."""
    os.makedirs("hf_datasets", exist_ok=True)

    filename = os.path.basename(url)
    local_path = os.path.join("hf_datasets", filename)

    # Download only if file doesn't already exist
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    return local_path


def load_hf_to_dataset(path, download_locally=False, **kwargs):
    """
    Load any Hugging Face dataset or file into a `datasets.Dataset`.

    If `download_locally=True` and `path` is a URL, the file is downloaded into
    ./hf_datasets/<filename> before loading. This is especially useful for tools
    like AnyToAASeq, which require local file paths.

    Parameters
    ----------
    path : str
        HF dataset name, local file path, or URL.

    download_locally : bool, default=False
        If True and `path` is a URL, download the file into ./hf_datasets/.

    **kwargs : dict
        Additional arguments passed to `datasets.load_dataset`.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict

        - If the source provides multiple splits (e.g. "train", "test"),
          a `datasets.DatasetDict` is returned.
        - If only a single split is present, the corresponding
          `datasets.Dataset` is returned directly (this function unwraps
          single-split DatasetDicts automatically).

    """
    original_path = path

    # Download external file when requested
    if download_locally and str(path).startswith(("http://", "https://")):
        path = _download_to_cwd(path)

    # File extension
    ext = os.path.splitext(str(path))[-1].lstrip(".").lower()

    # Load dataset depending on file type
    if ext not in FILE_FORMATS:
        ds = load_dataset(original_path if not download_locally else path, **kwargs)
    else:
        ds = load_dataset(
            "text",
            data_files=(original_path if not download_locally else path),
            **kwargs,
        )

    # Unwrap single-split DatasetDict into Dataset
    if isinstance(ds, dict) and len(ds) == 1:
        ds = next(iter(ds.values()))

    return ds
