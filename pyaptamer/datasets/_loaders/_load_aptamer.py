__author__ = "Satarupa22-SD"
__all__ = ["load_aptadb", "load_encoders"]

from pathlib import Path

import pandas as pd


def _download_dataset(
    dataset_name: str,
    target_dir: Path,
    force_download: bool = False,
) -> None:
    """Download a Kaggle dataset to the specified directory and unzip it."""
    import kaggle  # avoid import-time auth

    target_dir.mkdir(parents=True, exist_ok=True)

    # Only download if forced or no CSV files exist
    if force_download or not any(target_dir.glob("*.csv")):
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(target_dir),
            unzip=True,
        )


def load_encoders(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with multi-encoding fallback.

    This function attempts to read a CSV file using a list of common text
    encodings, ensuring robust loading even when the file has ambiguous or
    non-UTF-8 formatting. When ``encoding`` is explicitly provided, only that
    encoding is used. Otherwise, the function tries the following encodings
    in order:

    ``["utf-8", "utf-8-sig", "latin-1", "cp1252", "windows-1252"]``

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file to load. The file must exist locally.
    encoding : str, optional
        Specific encoding to use for reading the CSV. If ``None`` (default),
        multiple encodings are tried sequentially until one succeeds.
    **read_csv_kwargs
        Additional keyword arguments passed directly to ``pandas.read_csv``.
        Useful for specifying delimiters, NA values, column types, etc.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the parsed CSV data.

    Raises
    ------
    RuntimeError
        If the file cannot be read with any of the attempted encodings.
    FileNotFoundError
        If the given path does not point to an existing file.

    Examples
    --------
    >>> df = load_encoders("aptamer_interactions.csv")
    >>> df.head()
    """
    candidate_encodings = (
        [
            "utf-8",
            "utf-8-sig",
            "latin-1",
            "cp1252",
            "windows-1252",
        ]
        if encoding is None
        else [encoding]
    )

    last_error: Exception | None = None

    for enc in candidate_encodings:
        try:
            return pd.read_csv(path, encoding=enc, **read_csv_kwargs)
        except Exception as e:  # pragma: no cover - exercised via fallback
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to read CSV {path} with encodings {candidate_encodings}: {last_error}"
    )


def load_aptadb(
    dataset_name: str = "satarupadeb/aptamer-interactions",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    *,
    encoding: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download (if needed) and load the aptamer interaction dataset from Kaggle.

    This function retrieves the dataset from Kaggle using the Kaggle API,
    caches it locally, and loads the expected CSV file
    ``aptamer_interactions.csv`` into a pandas DataFrame.

    If the CSV already exists in the cache directory, it is used directly
    unless ``force_download=True`` is provided.

    Parameters
    ----------
    dataset_name : str, optional
        The Kaggle dataset identifier, formatted as
        ``"username/dataset-name"``.
        Default is ``"satarupadeb/aptamer-interactions"``.
    cache_dir : str or pathlib.Path, optional
        Directory where the dataset will be downloaded and cached.
        If ``None`` (default), the cache is stored under
        ``~/.pyaptamer/cache/<dataset_name>/``.
    force_download : bool, default False
        If ``True``, the dataset is downloaded even if a cached CSV already
        exists.
    encoding : str, optional
        Encoding to pass to the CSV loader. If ``None``, multiple encodings
        are attempted by ``load_encoders``.
    **kwargs
        Additional keyword arguments passed to ``load_encoders`` and
        ultimately to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The loaded aptamer interactions dataset. Typical columns include:

        - ``aptamer_id``
        - ``target_id``
        - ``aptamer_sequence``
        - ``target_name``
        - ``target_uniprot``
        - ``organism``
        - ``ligand_type``
        - ``binding_conditions``
        - ``reference_pubmed_id``
        - ``interaction_present``

    Raises
    ------
    ImportError
        If the ``kaggle`` Python package is not installed.
    RuntimeError
        If the dataset download fails.
    FileNotFoundError
        If ``aptamer_interactions.csv`` is not present after download.

    To Be Noted
    -----
    You must have Kaggle API credentials configured before using this
    function, by setting the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment
    variables.

    Examples
    --------
    Set the required environment variables in Python:
    >>> import os
    >>> os.environ["KAGGLE_USERNAME"] = (
    ...     "yourkaggleusername"  # Replace with your username
    ... )
    >>> os.environ["KAGGLE_KEY"] = "yourkaggleapi"  # Replace with your Kaggle API key


    Then load the dataset:

    >>> from pyaptamer.datasets import load_aptadb
    >>> df = load_aptadb()
    >>> print(df.head())
    >>> df.shape
    (1234, 10)

    >>> df = load_aptadb(cache_dir="data/cache", force_download=True)
    >>> df.columns
    Index([...], dtype='object')
    """
    if cache_dir is None:
        cache_dir = (
            Path.home() / ".pyaptamer" / "cache" / dataset_name.replace("/", "_")
        )
    else:
        cache_dir = Path(cache_dir)

    # The expected CSV filename in the Kaggle dataset
    csv_file = cache_dir / "aptamer_interactions.csv"

    if not csv_file.exists() or force_download:
        try:
            _download_dataset(dataset_name, cache_dir, force_download=force_download)
        except ImportError as err:
            raise ImportError(
                "The 'kaggle' package is required to download datasets. "
                "Install it with: pip install kaggle"
            ) from err
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}' from Kaggle: {e}"
            ) from e

        if not csv_file.exists():
            raise FileNotFoundError(
                f"Expected file 'aptamer_interactions.csv' not found at {cache_dir}"
            )

    return load_encoders(path=str(csv_file), encoding=encoding, **kwargs)
