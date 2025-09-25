__author__ = "Satarupa22-SD"
__all__ = ["load_aptadb", "load_aptamer_interactions", "load_interactions"]

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def download_dataset(dataset_name: str, target_dir: Path, force_download: bool = False):
    """Download dataset_name into target_dir using Kaggle API and unzip there."""
    import kaggle # avoid import-time auth
    target_dir.mkdir(parents=True, exist_ok=True)
    if force_download or not any(target_dir.glob("*.csv")):
        kaggle.api.dataset_download_files(dataset_name, path=str(target_dir), unzip=True)


def find_csv(directory: Path):
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        return None
    if len(csv_files) == 1:
        return csv_files[0]
    candidates = [
        f for f in csv_files
        if any(t in f.name.lower() for t in ["aptamer", "interaction", "main", "data"])
    ]
    return candidates[0] if candidates else csv_files[0]





def normalize_interaction_present(df: pd.DataFrame) -> None:
    return


def load_aptamer_interactions(
    path: Union[str, Path],
    *,
    encoding: Optional[str] = None,
    **read_csv_kwargs,
):
    """
    Load AptaDB-style CSV into a pandas.DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    encoding : str | None
        Specific file encoding. If None, several encodings are tried.
    **read_csv_kwargs : Any
        Additional arguments forwarded to pandas.read_csv.
    """
    candidate_encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",         
        "windows-1252",   
    ] if encoding is None else [encoding]
    last_error: Optional[Exception] = None
    for enc in candidate_encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **read_csv_kwargs)
            return df
        except Exception as e:
            last_error = e
            continue
    # If all encodings failed, raise the last error
    raise RuntimeError(f"Failed to read CSV at {path} with candidate encodings {candidate_encodings}: {last_error}")


def load_interactions(
    path: Union[str, Path],
    *,
    encoding: Optional[str] = None,
    **read_csv_kwargs,
):
    """Simple alias for load_aptamer_interactions."""
    return load_aptamer_interactions(
        path=path,
        encoding=encoding,
        **read_csv_kwargs,
    )


def load_aptadb(
    dataset_name: str = "satarupadeb/aptamer-interactions",
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    *,
    encoding: Optional[str] = None,
    **kwargs,
):
    """
    Download (optional) and load the aptamer-interactions Kaggle dataset as pandas.DataFrame.
    """
    cache_dir = (
        Path.home() / ".pyaptamer" / "cache" / dataset_name.replace("/", "_")
        if cache_dir is None else Path(cache_dir)
    )

    csv_file = find_csv(cache_dir) if cache_dir.exists() else None
    if csv_file is None:
        try:
            download_dataset(dataset_name, cache_dir, force_download=force_download)
        except ImportError:
            # ImportError for tests and clear messaging when kaggle is missing
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset '{dataset_name}' from Kaggle: {e}") from e
        csv_file = find_csv(cache_dir)
        if csv_file is None:
            raise FileNotFoundError(f"No CSV found in downloaded Kaggle dataset at {cache_dir}")

    return load_aptamer_interactions(path=str(csv_file), encoding=encoding, **kwargs)
