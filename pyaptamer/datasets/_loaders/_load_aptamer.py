__author__ = "Satarupa22-SD"
__all__ = ["load_aptadb", "load_encoders"]

from pathlib import Path

import pandas as pd


def _download_dataset(
    dataset_name: str, target_dir: Path, force_download: bool = False
) -> None:
    """Download a Kaggle dataset to the specified directory and unzip it."""
    import kaggle  # avoid import-time auth

    target_dir.mkdir(parents=True, exist_ok=True)

    # Only download if forced or no CSV files exist
    if force_download or not any(target_dir.glob("*.csv")):
        kaggle.api.dataset_download_files(
            dataset_name, path=str(target_dir), unzip=True
        )


def load_encoders(
    path: str | Path,
    *,
    encoding: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV into a DataFrame with robust multi-encoding fallback."""
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
        except Exception as e:
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
    """Download and load aptamer-interactions dataset from Kaggle as DataFrame."""
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
