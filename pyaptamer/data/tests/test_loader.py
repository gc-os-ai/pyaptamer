"""Tests for the MoleculeLoader lazy data container."""

__author__ = ["siddharth7113"]

from pathlib import Path

import pandas as pd
import pytest

from pyaptamer.data.loader import MoleculeLoader

DATA_DIR = Path(__file__).parent.parent.parent / "datasets" / "data"
PDB_MULTI = str(DATA_DIR / "1gnh.pdb")  # 10 chains A-J, all identical sequence
PDB_SINGLE = str(DATA_DIR / "1brq.pdb")  # 1 chain A
PDB_NO_SEQRES = str(DATA_DIR / "1gnh_no_seqres.pdb")


# --------------------------------------------------------------------------- #
# in-memory data
# --------------------------------------------------------------------------- #
def test_in_memory_data_is_noop():
    """In-memory sequences/primitives round-trip into columns unchanged."""
    proteins = ["ASCJNBDSFBWUJBCW", "SDUWEIPBNVNEWVUBW", "IOJVDPOIJWIDNVIVNV"]
    aptamers = ["AAACTAATATAAAATAAT", "CTCTAGGGGGGGGGG", "GGGGCAAAAAACCC"]
    bindings = [0.4, 0.5, 0.6]

    loader = MoleculeLoader(
        data={"target": proteins, "ligand": aptamers, "binding": bindings}
    )
    df = loader.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["target", "ligand", "binding"]
    assert df["target"].to_list() == proteins
    assert df["ligand"].to_list() == aptamers
    assert df["binding"].to_list() == bindings


# --------------------------------------------------------------------------- #
# tiling: bag
# --------------------------------------------------------------------------- #
def test_bag_tiling_str_vs_list():
    """bag: a multi-chain file -> list[str]; a single-chain file -> str."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI, PDB_SINGLE]}, tiling="bag")
    df = loader.to_dataframe()

    multi = df["target"].iloc[0]
    single = df["target"].iloc[1]

    assert isinstance(multi, list)
    assert len(multi) == 10
    assert all(isinstance(seq, str) for seq in multi)
    assert isinstance(single, str)


def test_bag_ignore_duplicates_collapses_to_str():
    """bag + ignore_duplicates: 10 identical chains dedup to 1 -> plain str."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="bag", ignore_duplicates=True
    )
    chain = loader.to_dataframe()["target"].iloc[0]

    assert isinstance(chain, str)
    assert chain.startswith("QTDMSRK")


# --------------------------------------------------------------------------- #
# tiling: concat / first
# --------------------------------------------------------------------------- #
def test_concat_joins_sequences():
    """concat: all chains joined into one string (length scales with count)."""
    plain = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="concat")
    deduped = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="concat", ignore_duplicates=True
    )

    full = plain.to_dataframe()["target"].iloc[0]
    one = deduped.to_dataframe()["target"].iloc[0]

    assert isinstance(full, str) and isinstance(one, str)
    # 10 identical chains -> concat is 10x the length of the deduped single one
    assert len(full) == 10 * len(one)


def test_first_keeps_only_first_sequence():
    """first: replace a multi-chain file with just its first sequence."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="first")
    chain = loader.to_dataframe()["target"].iloc[0]

    assert isinstance(chain, str)
    assert chain.startswith("QTDMSRK")


# --------------------------------------------------------------------------- #
# tiling: samples / samples_product
# --------------------------------------------------------------------------- #
def test_samples_explodes_to_rows():
    """samples: one multi-chain file becomes one row per chain."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="samples")
    df = loader.to_dataframe()

    assert len(df) == 10
    assert df["target"].map(type).eq(str).all()


def test_samples_multiple_file_columns_raise():
    """samples: expanding two file columns in one row is rejected as ambiguous."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI], "ligand": [PDB_SINGLE]}, tiling="samples"
    )
    with pytest.raises(ValueError, match="more than one file column"):
        loader.to_dataframe()


def test_samples_product_is_cartesian():
    """samples_product: two 10-chain files in a row -> 100 rows (10 x 10)."""
    crossed = MoleculeLoader(
        data={"a": [PDB_MULTI], "b": [PDB_MULTI]}, tiling="samples_product"
    )

    assert len(crossed.to_dataframe()) == 100  # cartesian product


# --------------------------------------------------------------------------- #
# tiling: features
# --------------------------------------------------------------------------- #
def test_features_expands_to_columns():
    """features: a 10-chain file spreads into target_0 ... target_9."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="features")
    df = loader.to_dataframe()

    assert len(df) == 1
    assert list(df.columns) == [f"target_{i}" for i in range(10)]


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def test_indexing_new_gives_rangeindex():
    """indexing='new': discard chain IDs, use a clean RangeIndex."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="new"
    )
    df = loader.to_dataframe()

    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.tolist() == list(range(10))


def test_indexing_preserve_flatten():
    """indexing='preserve' + multiindex='flatten': index is 'row__chain'."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="preserve"
    )
    df = loader.to_dataframe()

    assert df.index.tolist() == [f"0__{c}" for c in "ABCDEFGHIJ"]


def test_indexing_keep_as_column():
    """indexing='keep_as_column': chain IDs become a <col>_chain_id column."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="keep_as_column"
    )
    df = loader.to_dataframe()

    assert "target_chain_id" in df.columns
    assert df["target_chain_id"].tolist() == list("ABCDEFGHIJ")
    assert isinstance(df.index, pd.RangeIndex)


# --------------------------------------------------------------------------- #
# multiindex
# --------------------------------------------------------------------------- #
def test_multiindex_real():
    """multiindex='multiindex': a real two-level (row, sequence) index."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", multiindex="multiindex"
    )
    df = loader.to_dataframe()

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["row", "sequence"]
    assert df.index.get_level_values("sequence").tolist() == list("ABCDEFGHIJ")


def test_multiindex_auto_stays_flat_without_expansion():
    """multiindex='auto': single-sequence file does not trigger a MultiIndex."""
    loader = MoleculeLoader(
        data={"target": [PDB_SINGLE]}, tiling="samples", multiindex="auto"
    )
    df = loader.to_dataframe()

    assert not isinstance(df.index, pd.MultiIndex)


# --------------------------------------------------------------------------- #
# format coverage: FASTA via SeqIO
# --------------------------------------------------------------------------- #
def test_fasta_multi_record_bag(tmp_path):
    """A non-PDB format (FASTA) is parsed via SeqIO; 2 records -> list of 2."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(">seq1\nACGTACGT\n>seq2\nTTTTGGGG\n")

    loader = MoleculeLoader(data={"aptamer": [str(fasta)]}, tiling="bag")
    cell = loader.to_dataframe()["aptamer"].iloc[0]

    assert cell == ["ACGTACGT", "TTTTGGGG"]


def test_fastq_is_sequence_only(tmp_path):
    """FASTQ is read sequence-only (quality dropped); reads explode to rows."""
    fastq = tmp_path / "selex.fastq"
    fastq.write_text("@read1\nACGTACGT\n+\nIIIIIIII\n@read2\nTTTTGGGG\n+\nIIIIIIII\n")

    loader = MoleculeLoader(data={"selex": [str(fastq)]}, tiling="samples")
    df = loader.to_dataframe()

    assert df["selex"].tolist() == ["ACGTACGT", "TTTTGGGG"]
    # sequence-only: per-base quality is dropped, so no quality column appears
    assert list(df.columns) == ["selex"]


@pytest.mark.parametrize(
    "ext, content",
    [
        (
            "gb",
            "LOCUS       TEST                       8 bp    DNA     linear   UNK"
            " 01-JAN-1980\nDEFINITION  test.\nACCESSION   TEST\nFEATURES        "
            "     Location/Qualifiers\nORIGIN\n        1 acgtacgt\n//\n",
        ),
        (
            "embl",
            "ID   TEST; SV 1; linear; DNA; STD; UNC; 8 BP.\nSQ   Sequence 8 BP;"
            "\n     acgtacgt"
            "                                                                8\n//\n",
        ),
    ],
)
def test_genbank_and_embl_dispatch(tmp_path, ext, content):
    """GenBank/EMBL files dispatch through SeqIO by suffix and yield sequences."""
    path = tmp_path / f"record.{ext}"
    path.write_text(content)

    loader = MoleculeLoader(data={"seq": [str(path)]}, tiling="samples")
    df = loader.to_dataframe()

    assert df["seq"].tolist() == ["ACGTACGT"]


def test_fasta_path_broadcasts_against_in_memory_column(tmp_path):
    """A FASTA file column explodes per-record while an in-memory column broadcasts."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(">apt1\nACGTACGT\n>apt2\nTTTTGGGG\n>apt3\nGGGGCCCC\n")
    protein = "ACDEFGHIKLMNPQRSTVWY"

    loader = MoleculeLoader(
        data={"aptamer": [str(fasta)], "protein": [protein]}, tiling="samples"
    )
    df = loader.to_dataframe()

    assert df["aptamer"].tolist() == ["ACGTACGT", "TTTTGGGG", "GGGGCCCC"]
    assert df["protein"].tolist() == [protein, protein, protein]


# --------------------------------------------------------------------------- #
# validation / errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiling": "nonsense"},
        {"indexing": "nonsense"},
        {"multiindex": "nonsense"},
    ],
)
def test_invalid_options_raise(kwargs):
    """Unknown tiling/indexing/multiindex values are rejected at construction."""
    with pytest.raises(ValueError):
        MoleculeLoader(data={"target": [PDB_SINGLE]}, **kwargs)


def test_no_seqres_pdb_raises():
    """A PDB file without SEQRES records raises on materialization."""
    loader = MoleculeLoader(data={"target": [PDB_NO_SEQRES]})
    with pytest.raises(ValueError, match="No sequences found"):
        loader.to_dataframe()
