__author__ = "Satarupa22-SD"
__all__ = ["load_aptadb", "load_aptamer_interactions", "load_interactions"]

from pathlib import Path

import pandas as pd


def _download_dataset(
    dataset_name: str, target_dir: Path, force_download: bool = False
) -> None:
    """Download a Kaggle dataset to the specified directory and unzip it.

    This is a private helper function used internally by the module.

    Parameters
    ----------
    dataset_name : str
        The Kaggle dataset identifier in format "username/dataset-name"
    target_dir : Path
        Directory where the dataset should be downloaded and extracted
    force_download : bool, default False
        If True, download even if CSV files already exist in target_dir

    Raises
    ------
    ImportError
        If the kaggle package is not installed
    Exception
        If the download fails for any reason

    Notes
    -----
    This function requires the kaggle package to be installed and properly
    configured with API credentials.
    """
    import kaggle  # avoid import-time auth

    target_dir.mkdir(parents=True, exist_ok=True)

    # Only download if forced or no CSV files exist
    if force_download or not any(target_dir.glob("*.csv")):
        kaggle.api.dataset_download_files(
            dataset_name, path=str(target_dir), unzip=True
        )


def _find_csv(directory: Path) -> Path | None:
    """Find the most appropriate CSV file in a directory.

    This is a private helper function that implements smart CSV file detection.

    Parameters
    ----------
    directory : Path
        Directory to search for CSV files

    Returns
    -------
    Path or None
        Path to the most appropriate CSV file, or None if no CSV files found

    Notes
    -----
    Selection priority:
    1. If only one CSV file exists, return it
    2. If multiple CSV files exist, prefer files with names containing:
       "aptamer", "interaction", "main", or "data"
    3. If no preferred names found, return the first CSV file
    """
    csv_files = list(directory.glob("*.csv"))

    if not csv_files:
        return None

    if len(csv_files) == 1:
        return csv_files[0]

    # Look for files with preferred keywords in their names
    preferred_keywords = ["aptamer", "interaction", "main", "data"]
    candidates = [
        f
        for f in csv_files
        if any(keyword in f.name.lower() for keyword in preferred_keywords)
    ]

    return candidates[0] if candidates else csv_files[0]


def _normalize_interaction_present(df: pd.DataFrame) -> None:
    """Normalize interaction present column in the dataset.

    This is a private helper function for data preprocessing.
    Currently a placeholder for future implementation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to normalize

    Notes
    -----
    This function is currently not implemented and serves as a placeholder
    for future data normalization functionality.
    """
    # TODO: Implement interaction present normalization
    return


def load_aptamer_interactions(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load an aptamer interactions CSV file into a pandas DataFrame.

    This function provides robust CSV loading with automatic encoding detection
    and error handling for various file formats commonly found in biological
    datasets.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing aptamer interaction data
    encoding : str, optional
        Specific file encoding to use. If None (default), multiple common
        encodings will be tried automatically
    **read_csv_kwargs
        Additional keyword arguments passed directly to pandas.read_csv()

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded aptamer interaction data

    Raises
    ------
    RuntimeError
        If the CSV file cannot be read with any of the attempted encodings

    Notes
    -----
    The function attempts the following encodings in order:
    - utf-8
    - utf-8-sig (for files with BOM)
    - latin-1
    - cp1252
    - windows-1252
    """
    # Define candidate encodings to try
    candidate_encodings = (
        [
            "utf-8",
            "utf-8-sig",  # For files with byte order mark
            "latin-1",
            "cp1252",  # Common Windows encoding
            "windows-1252",  # Alternative Windows encoding
        ]
        if encoding is None
        else [encoding]
    )

    last_error: Exception | None = None

    # Try each encoding until one works
    for enc in candidate_encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **read_csv_kwargs)
            return df
        except Exception as e:
            last_error = e
            continue

    # If all encodings failed, raise informative error
    raise RuntimeError(
        f"Failed to read CSV at {path} with candidate encodings "
        f"{candidate_encodings}: {last_error}"
    )


def load_interactions(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load interaction data from a CSV file.

    This is a convenience alias for load_aptamer_interactions() with identical
    functionality and parameters.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing interaction data
    encoding : str, optional
        Specific file encoding to use. If None, automatic detection is used
    **read_csv_kwargs
        Additional keyword arguments passed to pandas.read_csv()

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded interaction data

    See Also
    --------
    load_aptamer_interactions : The main function this aliases
    """
    return load_aptamer_interactions(
        path=path,
        encoding=encoding,
        **read_csv_kwargs,
    )


def load_aptadb(
    dataset_name: str = "satarupadeb/aptamer-interactions",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    *,
    encoding: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Download and load aptamer-interactions Kaggle dataset as DataFrame.

    This is the main function for accessing aptamer interaction data. It
    handles dataset downloading, caching, and loading with a single function
    call.

    Parameters
    ----------
    dataset_name : str, default "satarupadeb/aptamer-interactions"
        Kaggle dataset identifier in format "username/dataset-name"
    cache_dir : str, Path, or None, default None
        Directory for caching downloaded datasets. If None, uses
        ~/.pyaptamer/cache/dataset_name
    force_download : bool, default False
        If True, re-download the dataset even if it exists in cache
    encoding : str, optional
        Specific file encoding for CSV reading. If None, automatic detection
        is used
    **kwargs
        Additional keyword arguments passed to pandas.read_csv()

    Returns
    -------
    pd.DataFrame
        DataFrame containing the aptamer interaction dataset

    Raises
    ------
    ImportError
        If the kaggle package is not installed
    RuntimeError
        If dataset download fails
    FileNotFoundError
        If no CSV files are found in the downloaded dataset

    Notes
    -----
    - Requires kaggle package installation and API configuration
    - First run will download data depending on dataset size
    - Subsequent runs use cached data unless force_download=True
    - Cache directory structure: cache_dir/username_dataset-name/
    """
    # Set up cache directory
    if cache_dir is None:
        cache_dir = (
            Path.home() / ".pyaptamer" / "cache" / dataset_name.replace("/", "_")
        )
    else:
        cache_dir = Path(cache_dir)

    # Check if we already have a cached CSV file
    csv_file = _find_csv(cache_dir) if cache_dir.exists() else None

    # Download dataset if no cached file exists
    if csv_file is None:
        try:
            _download_dataset(dataset_name, cache_dir, force_download=force_download)
        except ImportError:
            # Re-raise ImportError for clear messaging when kaggle is missing
            raise ImportError(
                "The 'kaggle' package is required to download datasets. "
                "Install it with: pip install kaggle"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}' from Kaggle: {e}"
            ) from e

        # Find the CSV file in downloaded data
        csv_file = _find_csv(cache_dir)
        if csv_file is None:
            raise FileNotFoundError(
                f"No CSV files found in downloaded Kaggle dataset at {cache_dir}"
            )

    # Load and return the CSV data
    return load_aptamer_interactions(path=str(csv_file), encoding=encoding, **kwargs)
