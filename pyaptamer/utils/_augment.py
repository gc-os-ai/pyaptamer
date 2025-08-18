__author__ = ["nennomp"]
__all__ = ["augment_reverse"]


def augment_reverse(*sequence_lists: list[str]) -> tuple[list[str], ...]:
    """Augment lists of sequences by adding their reverse complement.

    Parameters
    ----------
    *sequence_lists : list[str]
        Variable number of lists of sequences.

    Returns
    -------
    tuple[list[str], ...]
        A tuple of lists, each containing sequences with their reverse complements
        added.
    """
    results = []
    for sequences in sequence_lists:
        result = sequences.copy()  # start with original sequences
        for seq in sequences:
            result.append(seq[::-1])  # add reverse complement
        results.append(result)

    return tuple(results)
