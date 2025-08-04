from pyaptamer.datasets import load_1gnh_structure
from pyaptamer.utils.struct_to_aaseq import struct_to_aaseq


def test_struct_to_aaseq():
    """
    Test that struct_to_aaseq correctly converts a Biopython Structure
    into a list of amino‑acid sequences.

    Asserts:
        - No exception is raised when calling the function.
        - The return value is a list.
        - Each element of the list is a string.
    """
    structure = load_1gnh_structure()

    sequences = struct_to_aaseq(structure)

    assert isinstance(sequences, list), "Return value should be a list"
    for seq in sequences:
        assert isinstance(seq, str), "Each sequence should be a string"
