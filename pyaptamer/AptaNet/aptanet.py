from AptaNet.pseaac import pseaac


class AptaNet:
    """
    AptaNet feature generator for aptamer-protein pairs.

    This class generates a combined feature vector using k-mer frequencies from the aptamer DNA sequence
    and PseAAC features from the protein sequence.

    Parameters
    ----------
    aptamer_sequence : str
        The DNA sequence of the aptamer.
    protein_sequence : str
        The protein sequence to be analyzed.

    Attributes
    ----------
    aptamer_sequence : str
        The DNA sequence of the aptamer.
    protein_sequence : str
        The protein sequence to be analyzed.
    amino_acid : list of str
        List of 20 native amino acids in alphabetical order.
    pseaac : pseaac
        PseAAC object for the protein sequence.
    """

    def __init__(self, aptamer_sequence, protein_sequence):
        """
        Initialize the AptaNet class with an aptamer sequence and a protein sequence.

        Parameters
        ----------
        aptamer_sequence : str
            The DNA sequence of the aptamer.
        protein_sequence : str
            The protein sequence to be analyzed.
        """
        self.aptamer_sequence = aptamer_sequence
        self.protein_sequence = protein_sequence

        # Define the 20 native amino acids according to the alphabetical order of their single-letter codes
        self.amino_acid = list("ACDEFGHIKLMNPQRSTVWY")

        # Initialize the PseAAC object with the protein sequence
        self.pseaac = pseaac(self, self.protein_sequence)

    def _generate_kmer_vecs(self, aptamer_sequence, k=4):
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
        list of float
            Normalized frequency vector for all possible k-mers from length 1 to k.
        """
        from itertools import product

        bases = ["A", "C", "G", "T"]

        # Generate all possible k-mers from 1 to k
        all_kmers = []
        for i in range(1, k + 1):
            all_kmers.extend(["".join(p) for p in product(bases, repeat=i)])

        # Count occurrences of each k-mer in the aptamer_sequence
        kmer_counts = {kmer: 0 for kmer in all_kmers}
        for i in range(len(aptamer_sequence)):
            for j in range(1, k + 1):
                if i + j <= len(aptamer_sequence):
                    kmer = aptamer_sequence[i : i + j]
                    if kmer in kmer_counts:
                        kmer_counts[kmer] += 1

        # Normalize counts to frequencies
        total_kmers = sum(kmer_counts.values())
        kmer_freq = [
            kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
            for kmer in all_kmers
        ]

        return kmer_freq

    def generate_final_vector(self):
        """
        Generate the final feature vector by concatenating k-mer and PseAAC features.

        Returns
        -------
        list of float
            Combined feature vector for the aptamer-protein pair.
        """
        final_vector = (
            self._generate_kmer_vecs(self.aptamer_sequence, k=4) + self.pseaac
        )

        return final_vector
