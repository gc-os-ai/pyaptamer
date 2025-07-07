from AptaNet.pseaac import pseaac

class AptaNet:
    def __init__(self, aptamer_sequence, protein_sequence):
        """
        Initialize the AptaNetInput class with an aptamer sequence and a protein sequence.
        :param aptamer_sequence: The DNA sequence of the aptamer.
        :param protein_sequence: The protein sequence to be analyzed.
        """
        self.aptamer_sequence = aptamer_sequence
        self.protein_sequence = protein_sequence

        # Define the 20 native amino acids according to the alphabetical order of their single-letter codes
        self.amino_acid = list("ACDEFGHIKLMNPQRSTVWY")

        # Initialize the PseAAC object with the protein sequence
        self.pseaac = pseaac(self, self.protein_sequence)

    # Do what repDNA is doing in the original code, i.e. generating all possible DNA pairs till k-mer length for the aptamer, and then normalizing the frequency to form a vector
    def generate_kmer_vecs(self, aptamer_sequence, k=4):
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
                    kmer = aptamer_sequence[i:i + j]
                    if kmer in kmer_counts:
                        kmer_counts[kmer] += 1

        # Normalize counts to frequencies (like normalize_aa)
        total_kmers = sum(kmer_counts.values())
        kmer_freq = [kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0 for kmer in all_kmers]

        return kmer_freq

    def generate_final_vector(self):
        final_vector = self.generate_kmer_vecs(self.aptamer_sequence, k=4) + self.pseaac
    
        return final_vector