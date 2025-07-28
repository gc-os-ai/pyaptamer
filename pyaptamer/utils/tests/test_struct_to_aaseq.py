import pytest

from pyaptamer.datasets.loader import load_pfoa_structure
from pyaptamer.utils.struct_to_aaseq import struct_to_aaseq


def test_struct_to_aaseq_runs_and_returns_expected_type():
    structure = load_pfoa_structure()

    try:
        sequences = struct_to_aaseq(structure)
    except Exception as e:
        pytest.fail(f"struct_to_aaseq raised an exception: {e}")

    assert isinstance(sequences, list), "Return value should be a list"
    for seq in sequences:
        assert isinstance(seq, str), "Each sequence should be a string"
