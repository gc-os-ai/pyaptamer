from itertools import product

import numpy as np

from pyaptamer.pseaac import PSeAAC


def generate_kmer_vecs(aptamer_sequence, k=4):
    """
    Generate normalized k-mer frequency vectors for the aptamer sequence.

    For all possible k-mers from length 1 to k, count their occurrences in the sequence
    and normalize to form a frequency vector.

    Parameters
    ----------
    aptamer_sequence : str
        The DNA sequence of the aptamer.
    k : int, optional
        Maximum k-mer length (default is 4).

    Returns
    -------
    np.ndarray
        1D numpy array of normalized frequency vector for all possible k-mers from
        length 1 to k.
    """
    DNA_BASES = list("ACGT")

    # Generate all possible k-mers from 1 to k
    all_kmers = []
    for i in range(1, k + 1):
        all_kmers.extend(["".join(p) for p in product(DNA_BASES, repeat=i)])

    # Count occurrences of each k-mer in the aptamer_sequence
    kmer_counts = dict.fromkeys(all_kmers, 0)
    for i in range(len(aptamer_sequence)):
        for j in range(1, k + 1):
            if i + j <= len(aptamer_sequence):
                kmer = aptamer_sequence[i : i + j]
                if kmer in kmer_counts:
                    kmer_counts[kmer] += 1

    # Normalize counts to frequencies
    total_kmers = sum(kmer_counts.values())
    kmer_freq = np.array(
        [
            kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
            for kmer in all_kmers
        ]
    )

    return kmer_freq


def pairs_to_features(X, k=4):
    """
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.

    This function generates feature vectors for each (aptamer, protein) pair using:


    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence


    Parameters
    ----------
    X : list of tuple of str
        A list where each element is a tuple `(aptamer_sequence, protein_sequence)`.
        `aptamer_sequence` should be a string of nucleotides, and `protein_sequence`
        should be a string of amino acids.

    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated feature vector
        for a given (aptamer, protein) pair.
    """
    pseaac = PSeAAC()

    feats = []
    for aptamer_seq, protein_seq in X:
        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)


def rna2dna(sequence: str) -> str:
    """
    Convert an RNA sequence to a DNA sequence.

    Nucleotides 'U' in the RNA sequence are replaced with 'T' in the DNA sequence.
    Unknown nucleotides are replaced with 'N'. Other nucleotides ('A', 'C', 'G')
    remain unchanged.

    Parameters
    ----------
    sequence : str
        The RNA sequence to be converted.

    Returns
    -------
    str
        The converted DNA sequence.
    """
    # Replace nucleotides 'U' with 'T'
    result = sequence.translate(str.maketrans("U", "T"))

    # Replace any unknown characters with 'N'
    for char in result:
        if char not in "ACGT":
            result = result.replace(char, "N")

    return result
