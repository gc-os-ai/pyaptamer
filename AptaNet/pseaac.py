def pseaac(self, protein_sequence):
    # The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P1 = {
        "A": 0.62,
        "C": 0.29,
        "D": -0.90,
        "E": -0.74,
        "F": 1.19,
        "G": 0.48,
        "H": -0.40,
        "I": 1.38,
        "K": -1.50,
        "L": 1.06,
        "M": 0.64,
        "N": -0.78,
        "P": 0.12,
        "Q": -0.85,
        "R": -2.53,
        "S": -0.18,
        "T": -0.05,
        "V": 1.08,
        "W": 0.81,
        "Y": 0.26,
    }

    # The hydrophilicity values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P2 = {
        "A": -0.5,
        "C": -1.0,
        "D": 3.0,
        "E": 3.0,
        "F": -2.5,
        "G": 0.0,
        "H": -0.5,
        "I": -1.8,
        "K": 3.0,
        "L": -1.8,
        "M": -1.3,
        "N": 0.2,
        "P": 0.0,
        "Q": 0.2,
        "R": 3.0,
        "S": 0.3,
        "T": -0.4,
        "V": -1.5,
        "W": -3.4,
        "Y": -2.3,
    }

    # The side-chain mass for each of the 20 amino acids.
    P3 = {
        "A": 15.0,
        "C": 47.0,
        "D": 59.0,
        "E": 73.0,
        "F": 91.0,
        "G": 1.0,
        "H": 82.0,
        "I": 57.0,
        "K": 73.0,
        "L": 57.0,
        "M": 75.0,
        "N": 58.0,
        "P": 42.0,
        "Q": 72.0,
        "R": 101.0,
        "S": 31.0,
        "T": 45.0,
        "V": 43.0,
        "W": 130.0,
        "Y": 107.0,
    }

    # The Polarity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P4 = {
        "A": 0.5,
        "C": 2.5,
        "D": -1,
        "E": 2.5,
        "F": -2.5,
        "G": 0,
        "H": -0.5,
        "I": 1.8,
        "K": 3,
        "L": -1.8,
        "M": -1.3,
        "N": 0.2,
        "P": -1.4,
        "Q": 0.2,
        "R": 3,
        "S": 0.3,
        "T": -0.4,
        "V": -1.5,
        "W": -3.4,
        "Y": -2.3,
    }

    # The Molecular weight values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P5 = {
        "A": 5.3,
        "C": 3.6,
        "D": 1.3,
        "E": 3.3,
        "F": 2.3,
        "G": 4.8,
        "H": 1.4,
        "I": 3.1,
        "K": 4.1,
        "L": 4.7,
        "M": 1.1,
        "N": 3,
        "P": 2.5,
        "Q": 2.4,
        "R": 2.6,
        "S": 4.5,
        "T": 3.7,
        "V": 4.2,
        "W": 0.8,
        "Y": 2.3,
    }

    # The Meling point for each of the 20 amino acids.
    P6 = {
        "A": 0.81,
        "C": 0.71,
        "D": 1.17,
        "E": 0.53,
        "F": 1.2,
        "G": 0.88,
        "H": 0.92,
        "I": 1.48,
        "K": 0.77,
        "L": 1.24,
        "M": 1.05,
        "N": 0.62,
        "P": 0.61,
        "Q": 0.98,
        "R": 0.85,
        "S": 0.92,
        "T": 1.18,
        "V": 1.66,
        "W": 1.18,
        "Y": 1.23,
    }

    # The Transfer_free_energy values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P7 = {
        "A": 58,
        "C": -97,
        "D": 116,
        "E": -131,
        "F": 92,
        "G": -11,
        "H": -73,
        "I": 107,
        "K": -24,
        "L": 95,
        "M": 78,
        "N": -93,
        "P": -79,
        "Q": -139,
        "R": -184,
        "S": -34,
        "T": -7,
        "V": 100,
        "W": 59,
        "Y": -11,
    }

    # The Buriability values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P8 = {
        "A": 1.37,
        "C": 8.93,
        "D": -4.47,
        "E": 4.04,
        "F": -7.96,
        "G": 3.39,
        "H": -1.65,
        "I": -7.92,
        "K": 7.7,
        "L": -8.68,
        "M": -7.13,
        "N": 6.29,
        "P": 6.25,
        "Q": 3.88,
        "R": 1.33,
        "S": 4.08,
        "T": 4.02,
        "V": -6.94,
        "W": 0.79,
        "Y": -4.73,
    }

    # The Bulkiness for each of the 20 amino acids.
    P9 = {
        "A": 6.77,
        "C": 8.57,
        "D": 0.31,
        "E": 12.93,
        "F": 1.92,
        "G": 7.95,
        "H": 2.8,
        "I": 2.72,
        "K": 10.2,
        "L": 4.43,
        "M": 1.87,
        "N": 5.5,
        "P": 4.79,
        "Q": 5.24,
        "R": 6.87,
        "S": 5.41,
        "T": 5.36,
        "V": 3.57,
        "W": 0.54,
        "Y": 2.26,
    }

    # The Solvation_free_energy values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P10 = {
        "A": 0.87,
        "C": 0.66,
        "D": 1.52,
        "E": 0.67,
        "F": 2.87,
        "G": 0.1,
        "H": 0.87,
        "I": 3.15,
        "K": 1.64,
        "L": 2.17,
        "M": 1.67,
        "N": 0.09,
        "P": 2.77,
        "Q": 0,
        "R": 0.85,
        "S": 0.07,
        "T": 0.07,
        "V": 1.87,
        "W": 3.77,
        "Y": 2.67,
    }

    # The Relative_mutability values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P11 = {
        "A": 1.09,
        "C": 0.77,
        "D": 0.5,
        "E": 0.92,
        "F": 0.5,
        "G": 1.25,
        "H": 0.67,
        "I": 0.66,
        "K": 1.25,
        "L": 0.44,
        "M": 0.45,
        "N": 1.14,
        "P": 2.96,
        "Q": 0.83,
        "R": 0.97,
        "S": 1.21,
        "T": 1.33,
        "V": 0.56,
        "W": 0.62,
        "Y": 0.94,
    }

    # The Residue_volume for each of the 20 amino acids.
    P12 = {
        "A": 0.91,
        "C": 1.4,
        "D": 0.93,
        "E": 0.97,
        "F": 0.72,
        "G": 1.51,
        "H": 0.9,
        "I": 0.65,
        "K": 0.82,
        "L": 0.59,
        "M": 0.58,
        "N": 1.64,
        "P": 1.66,
        "Q": 0.94,
        "R": 1,
        "S": 1.23,
        "T": 1.04,
        "V": 0.6,
        "W": 0.67,
        "Y": 0.92,
    }

    # The volume values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P13 = {
        "A": 0.92,
        "C": 0.48,
        "D": 1.16,
        "E": 0.61,
        "F": 1.25,
        "G": 0.61,
        "H": 0.93,
        "I": 1.81,
        "K": 0.7,
        "L": 1.3,
        "M": 1.19,
        "N": 0.6,
        "P": 0.4,
        "Q": 0.95,
        "R": 0.93,
        "S": 0.82,
        "T": 1.12,
        "V": 1.81,
        "W": 1.54,
        "Y": 1.53,
    }

    # The Amino_acid_distribution values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P14 = {
        "A": 0.96,
        "C": 0.9,
        "D": 1.13,
        "E": 0.33,
        "F": 1.37,
        "G": 0.9,
        "H": 0.87,
        "I": 1.54,
        "K": 0.81,
        "L": 1.26,
        "M": 1.29,
        "N": 0.72,
        "P": 0.75,
        "Q": 1.18,
        "R": 0.67,
        "S": 0.77,
        "T": 1.23,
        "V": 1.41,
        "W": 1.13,
        "Y": 1.07,
    }

    # The Hydration_number for each of the 20 amino acids.
    P15 = {
        "A": 0.9,
        "C": 0.47,
        "D": 1.24,
        "E": 0.62,
        "F": 1.23,
        "G": 0.56,
        "H": 1.12,
        "I": 1.54,
        "K": 0.74,
        "L": 1.26,
        "M": 1.09,
        "N": 0.62,
        "P": 0.42,
        "Q": 1.18,
        "R": 1.02,
        "S": 0.87,
        "T": 1.3,
        "V": 1.53,
        "W": 1.75,
        "Y": 1.68,
    }

    # The Isoelectric_point values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P16 = {
        "A": 6,
        "C": 5.05,
        "D": 2.77,
        "E": 5.22,
        "F": 5.48,
        "G": 5.97,
        "H": 7.59,
        "I": 6.02,
        "K": 9.74,
        "L": 5.98,
        "M": 5.74,
        "N": 5.41,
        "P": 6.3,
        "Q": 5.65,
        "R": 10.76,
        "S": 5.68,
        "T": 5.66,
        "V": 5.96,
        "W": 5.89,
        "Y": 5.66,
    }

    # The Compressibility values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P17 = {
        "A": -25.5,
        "C": -32.82,
        "D": -33.12,
        "E": -36.17,
        "F": -34.54,
        "G": -27,
        "H": -31.84,
        "I": -31.78,
        "K": -32.4,
        "L": -31.78,
        "M": -31.18,
        "N": -30.9,
        "P": -23.25,
        "Q": -32.6,
        "R": -26.62,
        "S": -29.88,
        "T": -31.23,
        "V": -30.62,
        "W": -30.24,
        "Y": -35.01,
    }

    # The Chromatographic_index for each of the 20 amino acids.
    P18 = {
        "A": 9.9,
        "C": 2.8,
        "D": 2.8,
        "E": 3.2,
        "F": 18.8,
        "G": 5.6,
        "H": 8.2,
        "I": 17.1,
        "K": 3.5,
        "L": 17.6,
        "M": 14.7,
        "N": 5.4,
        "P": 14.8,
        "Q": 9,
        "R": 4.6,
        "S": 6.9,
        "T": 9.5,
        "V": 14.3,
        "W": 17,
        "Y": 15,
    }

    # The Unfolding_entropy_change values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    P19 = {
        "A": 0.54,
        "C": -4.14,
        "D": -0.26,
        "E": -0.19,
        "F": -4.66,
        "G": -0.31,
        "H": -0.23,
        "I": -0.27,
        "K": 1.13,
        "L": -0.24,
        "M": -2.36,
        "N": 1.74,
        "P": -0.08,
        "Q": 1.53,
        "R": 3.69,
        "S": -0.24,
        "T": -0.28,
        "V": -0.36,
        "W": -2.69,
        "Y": -2.82,
    }

    # The Unfolding_entalpy_change values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    P20 = {
        "A": 0.51,
        "C": 5.21,
        "D": 0.18,
        "E": 0.05,
        "F": 6.82,
        "G": -0.23,
        "H": 0.79,
        "I": 0.19,
        "K": -1.45,
        "L": 0.17,
        "M": 2.89,
        "N": -2.03,
        "P": 0.02,
        "Q": -1.76,
        "R": -4.4,
        "S": -0.16,
        "T": 0.04,
        "V": 0.3,
        "W": 4.47,
        "Y": 3.73,
    }

    # The Unfolding_Gibbs_free_energy_change for each of the 20 amino acids.
    P21 = {
        "A": -0.02,
        "C": 1.08,
        "D": -0.08,
        "E": -0.13,
        "F": 2.16,
        "G": 0.09,
        "H": 0.56,
        "I": -0.08,
        "K": -0.32,
        "L": -0.08,
        "M": 0.53,
        "N": -0.3,
        "P": -0.06,
        "Q": -0.23,
        "R": -0.71,
        "S": -0.4,
        "T": -0.24,
        "V": -0.06,
        "W": 1.78,
        "Y": -0.91,
    }

    # Define 7 selected groups of 3 properties each
    prop_groups = [
        (P1, P2, P3),
        (P4, P5, P6),
        (P7, P8, P9),
        (P10, P11, P12),
        (P13, P14, P15),
        (P16, P17, P18),
        (P19, P20, P21),
    ]

    def check_validity(self, seq):
        for aa in seq:
            if aa not in self.amino_acid:
                raise ValueError(
                    f"Invalid amino acid '{aa}' found in protein_sequence. Only {self.amino_acid} are allowed."
                )

    # Function to normalize amino acid composition
    def normalize_aa(self, seq):
        count = {aa: 0 for aa in self.amino_acid}
        for aa in seq:
            if aa in count:
                count[aa] += 1
        total = sum(count.values())
        return {aa: count[aa] / total if total > 0 else 0 for aa in count}

    # Normalize all physicochemical property dicts and store as a list
    def normalize_properties(*property_dicts):
        """
        Takes multiple amino acid property dictionaries and returns their normalized versions.
        Normalization: (value - mean) / std deviation
        Returns a list of normalized dictionaries in the same order.
        """
        normalized = []
        for prop in property_dicts:
            values = list(prop.values())
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            normalized.append(
                {aa: (val - mean_val) / std_val for aa, val in prop.items()}
            )
        return normalized

    def theta_RiRj(Ri, Rj, norm_props):
        return sum((prop[Rj] - prop[Ri]) ** 2 for prop in norm_props) / len(norm_props)

    def sum_theta_val(seq, seq_len, LVal, n, norm_props):
        return sum(
            theta_RiRj(seq[i], seq[i + n], norm_props) for i in range(seq_len - LVal)
        ) / (seq_len - n)

    check_validity(self, protein_sequence)

    lambda_val = 30
    weight = 0.15
    all_pseaac = []

    if len(protein_sequence) <= lambda_val:
        raise ValueError(
            f"Sequence too short for Lambda={lambda_val}. Must be > {lambda_val}."
        )

    for group in prop_groups:
        norm_props = normalize_properties(*group)
        aa_freq = normalize_aa(self, protein_sequence)
        sum_all_aa_freq = sum(aa_freq.values())

        all_theta_val = []
        sum_all_theta_val = 0
        for n in range(1, lambda_val + 1):
            theta_val = sum_theta_val(
                protein_sequence, len(protein_sequence), lambda_val, n, norm_props
            )
            all_theta_val.append(theta_val)
            sum_all_theta_val += theta_val

        denominator_val = sum_all_aa_freq + (weight * sum_all_theta_val)

        # First 20 features: normalized amino acid composition
        for aa in self.amino_acid:
            all_pseaac.append(round((aa_freq[aa] / denominator_val), 3))

        # Next 30 features: theta values
        for theta_val in all_theta_val:
            all_pseaac.append(round((weight * theta_val / denominator_val), 3))

    return all_pseaac
