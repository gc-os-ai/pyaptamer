import warnings

from pyaptamer.utils._pseaac_utils import clean_protein_seq


def test_clean_protein_seq_normalizes_lowercase_without_warning():
    seq = "acdefghiklmnpqrstvwy"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = clean_protein_seq(seq)

    assert cleaned == seq.upper()
    assert caught == []


def test_clean_protein_seq_warns_only_for_truly_invalid_residues():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = clean_protein_seq("Acd?z")

    assert cleaned == "ACDNN"
    assert len(caught) == 1
    assert "Invalid amino acid(s) found in sequence" in str(caught[0].message)
