class PSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.

    Parameters
    ----------
    protein_sequence : str
        Protein sequence using single-letter amino acid codes.
    """

    def __init__(self, protein_sequence):
        """
        Initialize PSeAAC with a protein sequence.

        Parameters
        ----------
        protein_sequence : str
            Protein sequence using single-letter amino acid codes.
        """
        self.protein_sequence = protein_sequence
        self.amino_acid = list("ACDEFGHIKLMNPQRSTVWY")

        # The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
        self.P1 = {
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
        self.P2 = {
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
        self.P3 = {
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
        self.P4 = {
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
        self.P5 = {
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
        self.P6 = {
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
        self.P7 = {
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
        self.P8 = {
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
        self.P9 = {
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
        self.P10 = {
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
        self.P11 = {
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
        self.P12 = {
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
        self.P13 = {
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
        self.P14 = {
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
        self.P15 = {
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
        self.P16 = {
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
        self.P17 = {
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
        self.P18 = {
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
        self.P19 = {
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
        self.P20 = {
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
        self.P21 = {
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

        # Normalized properties upto 3 decimal places
        self.NP1 = {
            "A": 0.636,
            "C": 0.298,
            "D": -0.924,
            "E": -0.759,
            "F": 1.221,
            "G": 0.493,
            "H": -0.41,
            "I": 1.416,
            "K": -1.539,
            "L": 1.088,
            "M": 0.657,
            "N": -0.8,
            "P": 0.123,
            "Q": -0.872,
            "R": -2.596,
            "S": -0.185,
            "T": -0.051,
            "V": 1.108,
            "W": 0.831,
            "Y": 0.267,
        }
        self.NP2 = {
            "A": -0.152,
            "C": -0.418,
            "D": 1.713,
            "E": 1.713,
            "F": -1.218,
            "G": 0.115,
            "H": -0.152,
            "I": -0.845,
            "K": 1.713,
            "L": -0.845,
            "M": -0.578,
            "N": 0.221,
            "P": 0.115,
            "Q": 0.221,
            "R": 1.713,
            "S": 0.274,
            "T": -0.099,
            "V": -0.685,
            "W": -1.697,
            "Y": -1.111,
        }
        self.NP3 = {
            "A": -1.592,
            "C": -0.53,
            "D": -0.131,
            "E": 0.334,
            "F": 0.931,
            "G": -2.057,
            "H": 0.632,
            "I": -0.198,
            "K": 0.334,
            "L": -0.198,
            "M": 0.4,
            "N": -0.164,
            "P": -0.696,
            "Q": 0.3,
            "R": 1.263,
            "S": -1.061,
            "T": -0.596,
            "V": -0.662,
            "W": 2.226,
            "Y": 1.462,
        }
        self.NP4 = {
            "A": 0.329,
            "C": 1.417,
            "D": -0.487,
            "E": 1.417,
            "F": -1.303,
            "G": 0.057,
            "H": -0.215,
            "I": 1.036,
            "K": 1.689,
            "L": -0.922,
            "M": -0.65,
            "N": 0.166,
            "P": -0.704,
            "Q": 0.166,
            "R": 1.689,
            "S": 0.22,
            "T": -0.16,
            "V": -0.759,
            "W": -1.792,
            "Y": -1.194,
        }
        self.NP5 = {
            "A": 1.755,
            "C": 0.429,
            "D": -1.365,
            "E": 0.195,
            "F": -0.585,
            "G": 1.365,
            "H": -1.287,
            "I": 0.039,
            "K": 0.819,
            "L": 1.287,
            "M": -1.521,
            "N": -0.039,
            "P": -0.429,
            "Q": -0.507,
            "R": -0.351,
            "S": 1.131,
            "T": 0.507,
            "V": 0.897,
            "W": -1.755,
            "Y": -0.585,
        }
        self.NP6 = {
            "A": -0.659,
            "C": -1.007,
            "D": 0.593,
            "E": -1.633,
            "F": 0.697,
            "G": -0.416,
            "H": -0.277,
            "I": 1.671,
            "K": -0.798,
            "L": 0.836,
            "M": 0.176,
            "N": -1.32,
            "P": -1.355,
            "Q": -0.068,
            "R": -0.52,
            "S": -0.277,
            "T": 0.628,
            "V": 2.297,
            "W": 0.628,
            "Y": 0.802,
        }
        self.NP7 = {
            "A": 0.735,
            "C": -0.968,
            "D": 1.372,
            "E": -1.341,
            "F": 1.108,
            "G": -0.023,
            "H": -0.704,
            "I": 1.273,
            "K": -0.166,
            "L": 1.141,
            "M": 0.955,
            "N": -0.924,
            "P": -0.77,
            "Q": -1.429,
            "R": -1.924,
            "S": -0.276,
            "T": 0.021,
            "V": 1.196,
            "W": 0.746,
            "Y": -0.023,
        }
        self.NP8 = {
            "A": 0.219,
            "C": 1.552,
            "D": -0.811,
            "E": 0.69,
            "F": -1.427,
            "G": 0.575,
            "H": -0.314,
            "I": -1.42,
            "K": 1.335,
            "L": -1.554,
            "M": -1.281,
            "N": 1.087,
            "P": 1.08,
            "Q": 0.662,
            "R": 0.212,
            "S": 0.697,
            "T": 0.686,
            "V": -1.247,
            "W": 0.117,
            "Y": -0.857,
        }
        self.NP9 = {
            "A": 0.56,
            "C": 1.13,
            "D": -1.484,
            "E": 2.509,
            "F": -0.975,
            "G": 0.933,
            "H": -0.696,
            "I": -0.722,
            "K": 1.645,
            "L": -0.181,
            "M": -0.991,
            "N": 0.158,
            "P": -0.067,
            "Q": 0.076,
            "R": 0.592,
            "S": 0.13,
            "T": 0.114,
            "V": -0.453,
            "W": -1.411,
            "Y": -0.867,
        }
        self.NP10 = {
            "A": -0.479,
            "C": -0.663,
            "D": 0.09,
            "E": -0.654,
            "F": 1.271,
            "G": -1.153,
            "H": -0.479,
            "I": 1.516,
            "K": 0.195,
            "L": 0.658,
            "M": 0.221,
            "N": -1.161,
            "P": 1.183,
            "Q": -1.24,
            "R": -0.496,
            "S": -1.179,
            "T": -1.179,
            "V": 0.396,
            "W": 2.058,
            "Y": 1.096,
        }
        self.NP11 = {
            "A": 0.253,
            "C": -0.338,
            "D": -0.836,
            "E": -0.061,
            "F": -0.836,
            "G": 0.548,
            "H": -0.522,
            "I": -0.541,
            "K": 0.548,
            "L": -0.947,
            "M": -0.928,
            "N": 0.345,
            "P": 3.703,
            "Q": -0.227,
            "R": 0.031,
            "S": 0.474,
            "T": 0.696,
            "V": -0.725,
            "W": -0.614,
            "Y": -0.024,
        }
        self.NP12 = {
            "A": -0.223,
            "C": 1.256,
            "D": -0.163,
            "E": -0.042,
            "F": -0.797,
            "G": 1.588,
            "H": -0.254,
            "I": -1.008,
            "K": -0.495,
            "L": -1.189,
            "M": -1.219,
            "N": 1.98,
            "P": 2.04,
            "Q": -0.133,
            "R": 0.048,
            "S": 0.743,
            "T": 0.169,
            "V": -1.159,
            "W": -0.948,
            "Y": -0.193,
        }
        self.NP13 = {
            "A": -0.277,
            "C": -1.356,
            "D": 0.311,
            "E": -1.037,
            "F": 0.532,
            "G": -1.037,
            "H": -0.253,
            "I": 1.905,
            "K": -0.816,
            "L": 0.655,
            "M": 0.385,
            "N": -1.062,
            "P": -1.552,
            "Q": -0.203,
            "R": -0.253,
            "S": -0.522,
            "T": 0.213,
            "V": 1.905,
            "W": 1.243,
            "Y": 1.219,
        }
        self.NP14 = {
            "A": -0.187,
            "C": -0.393,
            "D": 0.397,
            "E": -2.352,
            "F": 1.221,
            "G": -0.393,
            "H": -0.496,
            "I": 1.805,
            "K": -0.703,
            "L": 0.843,
            "M": 0.947,
            "N": -1.012,
            "P": -0.909,
            "Q": 0.569,
            "R": -1.184,
            "S": -0.84,
            "T": 0.74,
            "V": 1.359,
            "W": 0.397,
            "Y": 0.191,
        }
        self.NP15 = {
            "A": -0.402,
            "C": -1.503,
            "D": 0.469,
            "E": -1.119,
            "F": 0.443,
            "G": -1.273,
            "H": 0.161,
            "I": 1.237,
            "K": -0.812,
            "L": 0.52,
            "M": 0.085,
            "N": -1.119,
            "P": -1.631,
            "Q": 0.315,
            "R": -0.095,
            "S": -0.479,
            "T": 0.622,
            "V": 1.211,
            "W": 1.775,
            "Y": 1.595,
        }
        self.NP16 = {
            "A": -0.078,
            "C": -0.667,
            "D": -2.081,
            "E": -0.562,
            "F": -0.401,
            "G": -0.097,
            "H": 0.907,
            "I": -0.066,
            "K": 2.24,
            "L": -0.091,
            "M": -0.24,
            "N": -0.444,
            "P": 0.108,
            "Q": -0.295,
            "R": 2.872,
            "S": -0.277,
            "T": -0.289,
            "V": -0.103,
            "W": -0.147,
            "Y": -0.289,
        }
        self.NP17 = {
            "A": 1.728,
            "C": -0.604,
            "D": -0.7,
            "E": -1.671,
            "F": -1.152,
            "G": 1.25,
            "H": -0.292,
            "I": -0.273,
            "K": -0.47,
            "L": -0.273,
            "M": -0.082,
            "N": 0.008,
            "P": 2.445,
            "Q": -0.534,
            "R": 1.371,
            "S": 0.333,
            "T": -0.097,
            "V": 0.097,
            "W": 0.218,
            "Y": -1.302,
        }
        self.NP18 = {
            "A": -0.025,
            "C": -1.322,
            "D": -1.322,
            "E": -1.249,
            "F": 1.601,
            "G": -0.81,
            "H": -0.335,
            "I": 1.291,
            "K": -1.194,
            "L": 1.382,
            "M": 0.852,
            "N": -0.847,
            "P": 0.871,
            "Q": -0.189,
            "R": -0.993,
            "S": -0.573,
            "T": -0.098,
            "V": 0.779,
            "W": 1.273,
            "Y": 0.907,
        }
        self.NP19 = {
            "A": 0.549,
            "C": -1.863,
            "D": 0.137,
            "E": 0.173,
            "F": -2.131,
            "G": 0.111,
            "H": 0.152,
            "I": 0.131,
            "K": 0.853,
            "L": 0.147,
            "M": -0.946,
            "N": 1.167,
            "P": 0.229,
            "Q": 1.059,
            "R": 2.172,
            "S": 0.147,
            "T": 0.126,
            "V": 0.085,
            "W": -1.116,
            "Y": -1.183,
        }
        self.NP20 = {
            "A": -0.099,
            "C": 1.717,
            "D": -0.227,
            "E": -0.277,
            "F": 2.339,
            "G": -0.385,
            "H": 0.009,
            "I": -0.223,
            "K": -0.857,
            "L": -0.231,
            "M": 0.82,
            "N": -1.081,
            "P": -0.289,
            "Q": -0.976,
            "R": -1.996,
            "S": -0.358,
            "T": -0.281,
            "V": -0.18,
            "W": 1.431,
            "Y": 1.145,
        }
        self.NP21 = {
            "A": -0.2,
            "C": 1.276,
            "D": -0.28,
            "E": -0.348,
            "F": 2.725,
            "G": -0.052,
            "H": 0.578,
            "I": -0.28,
            "K": -0.602,
            "L": -0.28,
            "M": 0.538,
            "N": -0.576,
            "P": -0.254,
            "Q": -0.482,
            "R": -1.126,
            "S": -0.71,
            "T": -0.495,
            "V": -0.254,
            "W": 2.215,
            "Y": -1.394,
        }

        # Define 7 selected groups of 3 properties each
        self.prop_groups = [
            (self.NP1, self.NP2, self.NP3),
            (self.NP4, self.NP5, self.NP6),
            (self.NP7, self.NP8, self.NP9),
            (self.NP10, self.NP11, self.NP12),
            (self.NP13, self.NP14, self.NP15),
            (self.NP16, self.NP17, self.NP18),
            (self.NP19, self.NP20, self.NP21),
        ]

    def _check_validity(self, seq):
        """
        Check if the sequence contains only valid amino acids.

        Parameters
        ----------
        seq : str
            Protein sequence.

        Raises
        ------
        ValueError
            If an invalid amino acid is found.
        """
        for aa in seq:
            if aa not in self.amino_acid:
                raise ValueError(
                    f"Invalid amino acid '{aa}' found in protein_sequence. Only {self.amino_acid} are allowed."
                )

    # Function to average the amino acid composition
    def _average_aa(self, seq):
        """
        Compute the average amino acid composition for a sequence.

        Parameters
        ----------
        seq : str
            Protein sequence.

        Returns
        -------
        dict
            Dictionary mapping amino acid to its average frequency.
        """
        count = {aa: 0 for aa in self.amino_acid}
        for aa in seq:
            if aa in count:
                count[aa] += 1
        total = len(self.amino_acid)

        return {aa: count[aa] / total if total > 0 else 0 for aa in count}

    def _theta_RiRj(self, Ri, Rj, prop_group):
        """
        Compute the theta value between two amino acids for a group of properties.

        Parameters
        ----------
        Ri : str
            First amino acid.
        Rj : str
            Second amino acid.
        prop_group : tuple of dict
            Tuple of property dictionaries.

        Returns
        -------
        float
            Theta value.
        """
        return sum((prop[Rj] - prop[Ri]) ** 2 for prop in prop_group) / len(prop_group)

    def _sum_theta_val(self, seq, seq_len, lambda_val, n, prop_group):
        """
        Compute the average theta value for a sequence and property group.

        Parameters
        ----------
        seq : str
            Protein sequence.
        seq_len : int
            Length of the sequence.
        lambda_val : int
            Lambda parameter.
        n : int
            Offset for theta calculation.
        prop_group : tuple of dict
            Tuple of property dictionaries.

        Returns
        -------
        float
            Average theta value.
        """
        return sum(
            self._theta_RiRj(seq[i], seq[i + n], prop_group)
            for i in range(seq_len - lambda_val)
        ) / (seq_len - n)

    def vectorize(self):
        """
        Generate the PseAAC feature vector for the protein sequence.

        Returns
        -------
        list of float
            PseAAC feature vector.
        """
        self._check_validity(self.protein_sequence)

        lambda_val = 30
        weight = 0.15
        all_pseaac = []

        if len(self.protein_sequence) <= lambda_val:
            raise ValueError(
                f"Sequence too short for Lambda={lambda_val}. Must be > {lambda_val}."
            )

        for prop_group in self.prop_groups:
            aa_freq = self._average_aa(self.protein_sequence)
            sum_all_aa_freq = sum(aa_freq.values())

            all_theta_val = []
            sum_all_theta_val = 0
            for n in range(1, lambda_val + 1):
                theta_val = self._sum_theta_val(
                    self.protein_sequence,
                    len(self.protein_sequence),
                    lambda_val,
                    n,
                    prop_group,
                )
                all_theta_val.append(theta_val)
                sum_all_theta_val += theta_val

            denominator_val = sum_all_aa_freq + (weight * sum_all_theta_val)
            print(f"Denominator value: {denominator_val}")
            # First 20 features: normalized amino acid composition
            for aa in self.amino_acid:
                all_pseaac.append(round((aa_freq[aa] / denominator_val), 3))

            # Next 30 features: theta values
            for theta_val in all_theta_val:
                all_pseaac.append(round((weight * theta_val / denominator_val), 3))

        return all_pseaac
