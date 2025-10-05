__author__ = "Satarupa22-SD"
__all__ = ["load_aptadb", "load_aptamer_interactions", "load_interactions"]

from pathlib import Path

import pandas as pd


def _download_dataset(
    dataset_name: str, target_dir: Path, force_download: bool = False
) -> None:
    """Download a Kaggle dataset to the specified directory and unzip it.

    Parameters
    ----------
    dataset_name : str
        Kaggle dataset identifier like "username/dataset-name".
    target_dir : Path
        Directory to download and extract the dataset.
    force_download : bool, default False
        If True, download even if CSV files already exist in target_dir.

    Raises
    ------
    ImportError
        If the kaggle package is not installed.
    Exception
        If the download fails for any reason.

    Notes
    -----
    Requires kaggle package installed and configured with API credentials.
    """
    import kaggle  # avoid import-time auth

    target_dir.mkdir(parents=True, exist_ok=True)

    # Only download if forced or no CSV files exist
    if force_download or not any(target_dir.glob("*.csv")):
        kaggle.api.dataset_download_files(
            dataset_name, path=str(target_dir), unzip=True
        )


def _find_csv(directory: Path) -> Path | None:
    """Return the most appropriate CSV file path from a directory.

    Parameters
    ----------
    directory : Path
        Directory to look for CSV files.

    Returns
    -------
    Path or None
        Path to CSV file or None if none found.

    Notes
    -----
    Preference order:
    1. If only one CSV, return it.
    2. If multiple, prefer files with "aptamer", "interaction", "main", or "data"
    in name.
    3. Otherwise, return first CSV found.
    """
    csv_files = list(directory.glob("*.csv"))

    if not csv_files:
        return None

    if len(csv_files) == 1:
        return csv_files[0]

    preferred_keywords = ["aptamer", "interaction", "main", "data"]
    candidates = [
        f
        for f in csv_files
        if any(keyword in f.name.lower() for keyword in preferred_keywords)
    ]

    return candidates[0] if candidates else csv_files[0]


def _normalize_interaction_present(df: pd.DataFrame) -> None:
    """Placeholder to normalize 'interaction_present' column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.

    Notes
    -----
    Currently not implemented, kept for future data normalization.
    """
    return


def load_aptamer_interactions(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load aptamer interactions CSV into a pandas DataFrame.

    Tries common encodings automatically for robust loading.

    Parameters
    ----------
    path : str or Path
        Path to CSV file with aptamer interactions.
    encoding : str, optional
        Specific file encoding to use. If None, tries common encodings.
    **read_csv_kwargs
        Additional arguments passed to pandas.read_csv().

    Returns
    -------
    pd.DataFrame
        DataFrame with aptamer interaction data.

    Raises
    ------
    RuntimeError
        If CSV cannot be read with any attempted encodings.

    Notes
    -----
    Encodings tried (in order): utf-8, utf-8-sig, latin-1, cp1252, windows-1252.
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
            df = pd.read_csv(path, encoding=enc, **read_csv_kwargs)
            return df
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to read CSV {path} with encodings {candidate_encodings}: {last_error}"
    )


def load_interactions(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Alias for load_aptamer_interactions with same parameters and return."""
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
    """Download and load aptamer-interactions dataset from Kaggle as DataFrame.

    Parameters
    ----------
    dataset_name : str, optional
        Kaggle dataset name.
    cache_dir : str or Path, optional
        Local directory for caching dataset files.
    force_download : bool, default False
        If True, download dataset even if cached files exist.
    encoding : str, optional
        Encoding for CSV file loading.
    **kwargs
        Additional arguments passed to CSV loader.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.

    Raises
    ------
    ImportError
        If the 'kaggle' package is missing.
    RuntimeError
        If dataset download fails.
    FileNotFoundError
        If no CSV file found after download.
    """
    if cache_dir is None:
        cache_dir = (
            Path.home() / ".pyaptamer" / "cache" / dataset_name.replace("/", "_")
        )
    else:
        cache_dir = Path(cache_dir)

    csv_file = _find_csv(cache_dir) if cache_dir.exists() else None

    if csv_file is None:
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

        csv_file = _find_csv(cache_dir)
        if csv_file is None:
            raise FileNotFoundError(
                f"No CSV files found in downloaded Kaggle dataset at {cache_dir}"
            )

    return load_aptamer_interactions(path=str(csv_file), encoding=encoding, **kwargs)
