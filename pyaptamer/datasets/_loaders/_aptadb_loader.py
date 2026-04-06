__author__ = "Jayant-kernel"
__all__ = [
    "load_aptadb_interaction",
    "load_aptadb_aptamer",
    "load_aptadb_protein",
    "load_aptadb_molecule",
    "load_aptadb_cell",
    "load_aptadb_other",
]

from pathlib import Path

import pandas as pd
import requests

_APTADB_BASE_URL = "https://lmmd.ecust.edu.cn/aptadb/download/"

_TABLES = {
    "interaction": "interaction.csv",
    "aptamer": "aptamer.csv",
    "protein": "protein.csv",
    "molecule": "molecule.csv",
    "cell": "cell.csv",
    "other": "other.csv",
}

_DEFAULT_CACHE_DIR = Path.home() / ".pyaptamer" / "cache" / "aptadb"


def _download_csv(table_name, cache_dir, force_download=False):
    """Download a single AptaDB CSV file to the local cache directory.

    Parameters
    ----------
    table_name : str
        One of: ``"interaction"``, ``"aptamer"``, ``"protein"``,
        ``"molecule"``, ``"cell"``, or ``"other"``.
    cache_dir : pathlib.Path
        Directory to cache downloaded files.
    force_download : bool, default False
        If ``True``, re-download even if the file already exists.

    Returns
    -------
    pathlib.Path
        Local path of the downloaded CSV file.
    """
    filename = _TABLES[table_name]
    local_path = cache_dir / filename
    if not local_path.exists() or force_download:
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = _APTADB_BASE_URL + filename
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        local_path.write_bytes(response.content)
    return local_path


def _load_table(table_name, cache_dir=None, force_download=False, **read_csv_kwargs):
    """Download (if needed) and load an AptaDB CSV table as a pandas DataFrame.

    Parameters
    ----------
    table_name : str
        One of: ``"interaction"``, ``"aptamer"``, ``"protein"``,
        ``"molecule"``, ``"cell"``, or ``"other"``.
    cache_dir : str or pathlib.Path, optional
        Directory for caching downloaded CSV files.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        Re-download the file even when a cached copy exists.
    **read_csv_kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The loaded table as a DataFrame.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    else:
        cache_dir = Path(cache_dir)
    local_path = _download_csv(table_name, cache_dir, force_download=force_download)
    return pd.read_csv(local_path, encoding="latin-1", **read_csv_kwargs)


def load_aptadb_interaction(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB aptamer-target interaction table.

    Downloads ``interaction.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB interaction table. Each row describes one aptamer-target
        interaction. Columns include at minimum:

        - ``Aptamer_ID`` - AptaDB internal aptamer identifier.
        - ``Target_ID`` - AptaDB internal target identifier.
        - ``Aptamer_Sequence`` - Nucleotide sequence of the aptamer.
        - ``Target_Name`` - Name of the binding target.
        - ``Kd`` - Dissociation constant (affinity) value.
        - ``Kd_Unit`` - Unit for the Kd value.
        - ``Selection_Method`` - SELEX or other selection method used.
        - ``Reference`` - Literature reference (PubMed ID or DOI).

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_interaction
    >>> df = load_aptadb_interaction()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "interaction", cache_dir=cache_dir, force_download=force_download, **kwargs
    )


def load_aptadb_aptamer(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB aptamer sequence and property table.

    Downloads ``aptamer.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB aptamer table. Each row describes one aptamer.
        Columns include at minimum:

        - ``Aptamer_ID`` - AptaDB internal aptamer identifier.
        - ``Aptamer_Name`` - Common name of the aptamer.
        - ``Aptamer_Sequence`` - Nucleotide sequence.
        - ``Aptamer_Type`` - Chemistry type (DNA, RNA, or modified).
        - ``Aptamer_Length`` - Length of the aptamer sequence.
        - ``GC_Content`` - GC content percentage.
        - ``Molecular_Weight`` - Estimated molecular weight.
        - ``Structure`` - Secondary structure notation (if available).

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_aptamer
    >>> df = load_aptadb_aptamer()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "aptamer", cache_dir=cache_dir, force_download=force_download, **kwargs
    )


def load_aptadb_protein(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB protein target table.

    Downloads ``protein.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB protein table. Each row describes one protein target.
        Columns include at minimum:

        - ``Target_ID`` - AptaDB internal target identifier.
        - ``Target_Name`` - Protein name.
        - ``UniProt_ID`` - UniProt accession number.
        - ``Protein_Sequence`` - Amino acid sequence.
        - ``Organism`` - Source organism.
        - ``RCSB_PDB_ID`` - PDB structure identifier (if available).

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_protein
    >>> df = load_aptadb_protein()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "protein", cache_dir=cache_dir, force_download=force_download, **kwargs
    )


def load_aptadb_molecule(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB small molecule target table.

    Downloads ``molecule.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB small molecule table. Each row describes one small
        molecule target. Columns include at minimum:

        - ``Target_ID`` - AptaDB internal target identifier.
        - ``Target_Name`` - Molecule name.
        - ``CAS_Number`` - CAS registry number.
        - ``Molecular_Formula`` - Chemical formula.
        - ``Molecular_Weight`` - Molecular weight in g/mol.
        - ``SMILES`` - SMILES notation.
        - ``PubChem_CID`` - PubChem compound identifier.

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_molecule
    >>> df = load_aptadb_molecule()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "molecule", cache_dir=cache_dir, force_download=force_download, **kwargs
    )


def load_aptadb_cell(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB cellular target table.

    Downloads ``cell.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB cell table. Each row describes one cellular target.
        Columns include at minimum:

        - ``Target_ID`` - AptaDB internal target identifier.
        - ``Cell_Name`` - Cell line or cell type name.
        - ``Cell_Type`` - Classification (e.g. cancer, normal).
        - ``Organism`` - Source organism.
        - ``Tissue`` - Tissue of origin.
        - ``Disease`` - Associated disease (if applicable).

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_cell
    >>> df = load_aptadb_cell()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "cell", cache_dir=cache_dir, force_download=force_download, **kwargs
    )


def load_aptadb_other(cache_dir=None, force_download=False, **kwargs):
    """Load the AptaDB miscellaneous target table.

    Downloads ``other.csv`` from https://lmmd.ecust.edu.cn/aptadb
    and returns it as a ``pandas.DataFrame``. The file is cached locally
    after the first download.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory for caching the downloaded CSV.
        Defaults to ``~/.pyaptamer/cache/aptadb/``.
    force_download : bool, default False
        If ``True``, re-download even when a cached copy exists.
    **kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        The AptaDB other-targets table. Each row describes one target that
        does not fall under protein, molecule, or cell categories. Columns
        include at minimum:

        - ``Target_ID`` - AptaDB internal target identifier.
        - ``Target_Name`` - Target name.
        - ``Target_Type`` - Category (virus, bacterium, etc.).
        - ``Organism`` - Source organism (if applicable).
        - ``Description`` - Free-text description of the target.

    Examples
    --------
    >>> from pyaptamer.datasets import load_aptadb_other
    >>> df = load_aptadb_other()
    >>> type(df).__name__
    'DataFrame'
    """
    return _load_table(
        "other", cache_dir=cache_dir, force_download=force_download, **kwargs
    )
