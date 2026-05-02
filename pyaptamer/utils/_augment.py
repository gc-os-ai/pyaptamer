__author__ = ["nennomp"]
__all__ = ["augment_reverse", "augment_complement"]

import numpy as np

# Complement mapping tables for DNA and RNA
_DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
_RNA_COMPLEMENT = str.maketrans("ACGUacgu", "UGCAugca")


def augment_reverse(*sequence_arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Augment arrays of sequences by adding the reversed sequences.

    Parameters
    ----------
    *sequence_arrays : np.ndarray
        Variable number of numpy arrays of sequences (containing strings).

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of arrays, each containing sequences with their reversed sequences.
        added.
    """
    results = []
    for sequences in sequence_arrays:
        # create array of reversed sequences
        reversed_sequences = np.array([seq[::-1] for seq in sequences])
        # concatenate original and reversed sequences
        result = np.concatenate([sequences, reversed_sequences])
        results.append(result)

    return tuple(results)


def _reverse_complement(sequence: str, molecule_type: str) -> str:
    """Compute the reverse complement of a single sequence.

    Parameters
    ----------
    sequence : str
        A nucleotide sequence string.
    molecule_type : str
        Either ``"rna"`` or ``"dna"``.

    Returns
    -------
    str
        The reverse complement of the input sequence.
    """
    table = _RNA_COMPLEMENT if molecule_type == "rna" else _DNA_COMPLEMENT
    return sequence.translate(table)[::-1]


def augment_complement(
    *sequence_arrays: np.ndarray,
    molecule_type: str = "rna",
) -> tuple[np.ndarray, ...]:
    """Augment arrays of sequences by adding their reverse complements.

    For each input array, the reverse complement of every sequence is
    computed and concatenated with the originals. This is a biologically
    meaningful augmentation strategy because the reverse complement
    represents the complementary strand of a nucleic acid duplex
    (A↔U for RNA, A↔T for DNA, C↔G for both).

    Parameters
    ----------
    *sequence_arrays : np.ndarray
        Variable number of numpy arrays of sequences (containing strings).
    molecule_type : str, optional, default="rna"
        Type of molecule. Must be ``"rna"`` or ``"dna"``.
        Determines the complement mapping:

        - ``"rna"``: A↔U, C↔G
        - ``"dna"``: A↔T, C↔G

    Returns
    -------
    tuple[np.ndarray, ...]
        A tuple of arrays, each containing the original sequences followed
        by their reverse complements.

    Raises
    ------
    ValueError
        If ``molecule_type`` is not ``"rna"`` or ``"dna"``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.utils._augment import augment_complement
    >>> seqs = np.array(["AUGC", "GCAU"])
    >>> (result,) = augment_complement(seqs, molecule_type="rna")
    >>> result
    array(['AUGC', 'GCAU', 'GCAU', 'AUGC'], dtype='<U4')
    """
    if molecule_type not in ("rna", "dna"):
        raise ValueError(
            f"molecule_type must be 'rna' or 'dna', got '{molecule_type}'."
        )

    results = []
    for sequences in sequence_arrays:
        rc_sequences = np.array(
            [_reverse_complement(seq, molecule_type) for seq in sequences]
        )
        result = np.concatenate([sequences, rc_sequences])
        results.append(result)

    return tuple(results)
