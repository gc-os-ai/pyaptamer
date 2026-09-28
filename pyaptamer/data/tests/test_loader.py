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


# in-memory data
def test_in_memory_data_is_noop():
    """In-memory cells are returned as given: same columns, same values."""
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


# tiling: bag
def test_bag_tiling_str_vs_list():
    """bag: a multi-chain file gives list[str]; a single-chain file gives str."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI, PDB_SINGLE]}, tiling="bag")
    df = loader.to_dataframe()

    multi = df["target"].iloc[0]
    single = df["target"].iloc[1]

    assert isinstance(multi, list)
    assert len(multi) == 10
    assert all(isinstance(seq, str) for seq in multi)
    assert isinstance(single, str)


def test_bag_ignore_duplicates_collapses_to_str():
    """bag + ignore_duplicates: 10 identical chains give one str."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="bag", ignore_duplicates=True
    )
    chain = loader.to_dataframe()["target"].iloc[0]

    assert isinstance(chain, str)
    assert chain.startswith("QTDMSRK")


# tiling: concat / first
def test_concat_joins_sequences():
    """concat: the cell is all chains joined into one string."""
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
    """first: the cell is the first sequence of the file."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="first")
    chain = loader.to_dataframe()["target"].iloc[0]

    assert isinstance(chain, str)
    assert chain.startswith("QTDMSRK")


# tiling: samples / samples_product
def test_samples_explodes_to_rows():
    """samples: a 10-chain file gives 10 rows, one per chain."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="samples")
    df = loader.to_dataframe()

    assert len(df) == 10
    assert df["target"].map(type).eq(str).all()


def test_samples_multiple_file_columns_raise():
    """samples: two file columns in one row raise ValueError."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI], "ligand": [PDB_SINGLE]}, tiling="samples"
    )
    with pytest.raises(ValueError, match="more than one file column"):
        loader.to_dataframe()


def test_samples_literal_only_row_passes_through():
    """samples: a row with no file cell gives one row; its label has no chain part."""
    loader = MoleculeLoader(data={"seq": [PDB_SINGLE, "ACGT"]}, tiling="samples")
    df = loader.to_dataframe()

    assert df["seq"].tolist()[1] == "ACGT"
    assert df.index.tolist() == ["0__A", "1"]


def test_samples_product_is_cartesian():
    """samples_product: two 10-chain files in one row give 100 rows."""
    crossed = MoleculeLoader(
        data={"a": [PDB_MULTI], "b": [PDB_MULTI]}, tiling="samples_product"
    )

    assert len(crossed.to_dataframe()) == 100  # cartesian product


# tiling: features
def test_features_expands_to_columns():
    """features: a 10-chain file gives columns target_0 to target_9."""
    loader = MoleculeLoader(data={"target": [PDB_MULTI]}, tiling="features")
    df = loader.to_dataframe()

    assert len(df) == 1
    assert list(df.columns) == [f"target_{i}" for i in range(10)]


def test_features_keeps_literal_and_single_sequence_columns():
    """features: literal and single-sequence columns keep their name and values."""
    loader = MoleculeLoader(
        data={
            "target": [PDB_MULTI, "ACGT"],
            "ligand": [PDB_SINGLE, PDB_SINGLE],
            "name": ["a", "b"],
        },
        tiling="features",
    )
    df = loader.to_dataframe()

    assert list(df.columns) == [f"target_{i}" for i in range(10)] + ["ligand", "name"]
    assert df["target_0"].iloc[1] == "ACGT"
    assert pd.isna(df["target_1"].iloc[1])
    assert df["ligand"].map(type).eq(str).all()
    assert df["name"].tolist() == ["a", "b"]


# indexing
def test_indexing_new_gives_rangeindex():
    """indexing='new': the index is a RangeIndex; chain IDs are dropped."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="new"
    )
    df = loader.to_dataframe()

    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.tolist() == list(range(10))


def test_indexing_preserve_flatten():
    """indexing='preserve' + multiindex='flatten': index labels are 'row__chain'."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="preserve"
    )
    df = loader.to_dataframe()

    assert df.index.tolist() == [f"0__{c}" for c in "ABCDEFGHIJ"]


def test_indexing_keep_as_column():
    """indexing='keep_as_column': chain IDs are in a <col>_chain_id column."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", indexing="keep_as_column"
    )
    df = loader.to_dataframe()

    assert "target_chain_id" in df.columns
    assert df["target_chain_id"].tolist() == list("ABCDEFGHIJ")
    assert isinstance(df.index, pd.RangeIndex)


# multiindex
def test_multiindex_real():
    """multiindex='multiindex': the index is a (row, sequence) MultiIndex."""
    loader = MoleculeLoader(
        data={"target": [PDB_MULTI]}, tiling="samples", multiindex="multiindex"
    )
    df = loader.to_dataframe()

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["row", "sequence"]
    assert df.index.get_level_values("sequence").tolist() == list("ABCDEFGHIJ")


def test_multiindex_auto_stays_flat_without_expansion():
    """multiindex='auto': a single-sequence file gives a flat index."""
    loader = MoleculeLoader(
        data={"target": [PDB_SINGLE]}, tiling="samples", multiindex="auto"
    )
    df = loader.to_dataframe()

    assert not isinstance(df.index, pd.MultiIndex)


# format coverage: the generic SeqIO reader
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
    """GenBank and EMBL files are read by SeqIO.parse, picked by suffix."""
    path = tmp_path / f"record.{ext}"
    path.write_text(content)

    loader = MoleculeLoader(data={"seq": [str(path)]}, tiling="samples")
    df = loader.to_dataframe()

    assert df["seq"].tolist() == ["ACGTACGT"]


def test_fasta_path_broadcasts_against_in_memory_columns(tmp_path):
    """samples: one row per record; literal columns are repeated and keep dtype."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(">apt1\nACGTACGT\n>apt2\nTTTTGGGG\n>apt3\nGGGGCCCC\n")
    protein = "ACDEFGHIKLMNPQRSTVWY"

    loader = MoleculeLoader(
        data={"aptamer": [str(fasta)], "protein": [protein], "binding": [0.5]},
        tiling="samples",
    )
    df = loader.to_dataframe()

    assert df["aptamer"].tolist() == ["ACGTACGT", "TTTTGGGG", "GGGGCCCC"]
    assert df["protein"].tolist() == [protein, protein, protein]
    assert df["binding"].tolist() == [0.5, 0.5, 0.5]
    assert df["binding"].dtype == "float64"


# path-like cells
#
# Cells may hold a str or any os.PathLike. A Path used to fall through to the
# literal branch and be returned unparsed, which was silent rather than an
# error, so both forms are checked against each other.
def test_path_object_parses_like_str():
    """A Path cell is read as a file and gives the same table as the str cell."""
    as_str = MoleculeLoader(data={"target": [PDB_SINGLE]}).to_dataframe()
    as_path = MoleculeLoader(data={"target": [Path(PDB_SINGLE)]}).to_dataframe()

    assert as_path.equals(as_str)
    assert isinstance(as_path["target"].iloc[0], str)


def test_path_object_explodes_to_rows_under_samples():
    """samples: a Path FASTQ gives the same rows as the str path."""
    fastq = DATA_DIR / "sample.fastq"

    as_str = MoleculeLoader(data={"seq": [str(fastq)]}, tiling="samples").to_dataframe()
    as_path = MoleculeLoader(data={"seq": [fastq]}, tiling="samples").to_dataframe()

    assert len(as_path) == 10
    assert as_path.equals(as_str)


def test_suffixless_str_stays_a_literal():
    """A str with no suffix is kept as a literal value."""
    df = MoleculeLoader(data={"seq": ["ACGT"]}).to_dataframe()

    assert df["seq"].iloc[0] == "ACGT"


def test_str_and_path_mix_in_one_column():
    """str and Path cells in one column are both read as files."""
    fastq = DATA_DIR / "sample.fastq"

    loader = MoleculeLoader(data={"seq": [str(fastq), fastq]}, tiling="samples")
    df = loader.to_dataframe()

    assert len(df) == 20


# validation / errors
@pytest.mark.parametrize(
    "kwargs",
    [
        {"tiling": "nonsense"},
        {"indexing": "nonsense"},
        {"multiindex": "nonsense"},
    ],
)
def test_invalid_options_raise(kwargs):
    """Unknown tiling, indexing or multiindex values raise ValueError in __init__."""
    with pytest.raises(ValueError):
        MoleculeLoader(data={"target": [PDB_SINGLE]}, **kwargs)


def test_no_seqres_pdb_raises():
    """A PDB file with no SEQRES records raises ValueError in to_dataframe."""
    loader = MoleculeLoader(data={"target": [PDB_NO_SEQRES]})
    with pytest.raises(ValueError, match="No sequences found"):
        loader.to_dataframe()


@pytest.mark.parametrize(
    "cell", [Path("reads"), "reads.gz"], ids=["bare-path", "gz-only"]
)
def test_missing_format_suffix_raises(cell):
    """A file cell with no format suffix (bare Path, or only .gz) raises ValueError."""
    with pytest.raises(ValueError, match="picks the parser from the file suffix"):
        MoleculeLoader(data={"seq": [cell]}).to_dataframe()


# file readers: suffix rule, gzip, reader dispatch
@pytest.mark.parametrize(
    "name, expected",
    [
        ("reads.fastq", "fastq"),
        ("reads.fastq.gz", "fastq"),
        ("READS.FASTA", "fasta"),
        ("sample.v2.fastq", "fastq"),
        ("reads.gz", None),
        ("reads", None),
    ],
)
def test_determine_type(name, expected):
    """_determine_type: last suffix, lowercased, with a trailing .gz skipped."""
    assert MoleculeLoader(data={})._determine_type(Path(name)) == expected


def test_fastq_does_not_use_seqio_parse(tmp_path, monkeypatch):
    """FASTQ files are read by FastqGeneralIterator, not SeqIO.parse."""
    from pyaptamer.data import loader as loader_module

    def fail(*args, **kwargs):
        raise AssertionError("SeqIO.parse was called for a FASTQ file")

    monkeypatch.setattr(loader_module.SeqIO, "parse", fail)
    fastq = tmp_path / "selex.fastq"
    fastq.write_text("@r1\nACGT\n+\nIIII\n")

    df = MoleculeLoader(data={"seq": [str(fastq)]}, tiling="samples").to_dataframe()

    assert df["seq"].tolist() == ["ACGT"]


@pytest.mark.parametrize(
    "name, content",
    [
        ("library.fasta.gz", ">apt1\nACGTACGT\n>apt2\nTTTT\nGGGG\n"),
        ("selex.fastq.gz", "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTGGGG\n+\nIIIIIIII\n"),
    ],
    ids=["fasta", "fastq"],
)
def test_gzipped_file_is_read(tmp_path, name, content):
    """A .gz file is decompressed and read by the reader for the suffix under .gz."""
    import gzip

    path = tmp_path / name
    with gzip.open(path, "wt") as handle:
        handle.write(content)

    df = MoleculeLoader(data={"seq": [str(path)]}, tiling="samples").to_dataframe()

    assert df["seq"].tolist() == ["ACGTACGT", "TTTTGGGG"]


def test_fasta_id_is_first_word_of_header(tmp_path):
    """FASTA chain_id is the first word of the header line, as in SeqIO record.id."""
    fasta = tmp_path / "library.fasta"
    fasta.write_text(
        ">apt1 first aptamer\nACGTACGT\n>apt2 second aptamer\nTTTT\nGGGG\n"
    )

    df = MoleculeLoader(
        data={"seq": [str(fasta)]}, tiling="samples", indexing="keep_as_column"
    ).to_dataframe()

    assert df["seq_chain_id"].tolist() == ["apt1", "apt2"]
    assert df["seq"].tolist() == ["ACGTACGT", "TTTTGGGG"]


def test_tricky_fastq_parses(tmp_path):
    """A '@' starting a quality line or a repeated '+' title does not split a read."""
    fastq = tmp_path / "tricky.fastq"
    fastq.write_text("@r1 desc\nACGT\n+r1 desc\n@III\n@r2\nTTTT\n+\n@@@@\n")

    df = MoleculeLoader(
        data={"seq": [str(fastq)]}, tiling="samples", indexing="keep_as_column"
    ).to_dataframe()

    assert df["seq"].tolist() == ["ACGT", "TTTT"]
    assert df["seq_chain_id"].tolist() == ["r1", "r2"]
