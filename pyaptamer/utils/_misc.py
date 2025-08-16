"Generic utilities."

__author__ = ["nennomp"]
__all__ = ["generate_triplets"]

from itertools import product

def generate_triplets(letters: list[str]) -> dict[str, int]:
    """Generate a dictionary of all possible triplets combinations from given letters.
    
    Parameters
    ----------
    letters : list[str]
        List of characters to form triplets from.

    Returns
    -------
    dict[str, int]
        A dictionary mapping each triplet to a unique integer ID.
    """
    triplets = {}
    for idx, triplet in enumerate(product(letters, repeat=3)):
        triplets["".join(triplet)] = idx + 1
        
    return triplets