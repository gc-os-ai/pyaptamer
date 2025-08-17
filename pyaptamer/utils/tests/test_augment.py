"""Test suite for the data augmentation utilities."""

__author__ = ["nennomp"]


from pyaptamer.utils.augment import augment_reverse


def test_augment_reverse_single_list():
    """Test augment_reverse with a single list of sequences."""
    sequences = ["AAC", "BBB", "ATCG"]
    result = augment_reverse(sequences)
    
    expected = (["AAC", "BBB", "ATCG", "CAA", "BBB", "GCTA"],)
    assert result == expected
    assert len(result) == 1
    assert len(result[0]) == 6


def test_augment_reverse_multiple_lists():
    """Test augment_reverse with multiple lists of sequences."""
    seq1 = ["ABC", "DEF"]
    seq2 = ["XYZ"]
    seq3 = ["123", "456", "789"]
    
    result = augment_reverse(seq1, seq2, seq3)
    
    expected = (
        ["ABC", "DEF", "CBA", "FED"],
        ["XYZ", "ZYX"],
        ["123", "456", "789", "321", "654", "987"]
    )
    assert result == expected
    assert len(result) == 3
    assert len(result[0]) == 4
    assert len(result[1]) == 2
    assert len(result[2]) == 6