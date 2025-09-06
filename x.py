#!/usr/bin/env python3
"""
Example script to run Pseudo Amino Acid Composition (PseAAC) feature extraction.
"""

import numpy as np

from pyaptamer.pseaac import PSeAAC


def main():
    # Example protein sequence
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 amino acids

    # Initialize PSeAAC with default parameters
    pseaac = PSeAAC(lambda_val=30, weight=0.05)

    # Compute feature vector
    features = pseaac.transform(sequence)

    # Show basic info
    print("=== PSeAAC Example ===")
    print(f"Input sequence length: {len(sequence)}")
    print(f"Lambda value: {pseaac.lambda_val}")
    print(f"Weight: {pseaac.weight}")
    print(f"Output feature vector length: {len(features)}")

    # Show first 20 values as a sample
    print("\nFirst 20 features:")
    print(np.round(features, 3))

    # Optionally, save full feature vector to file
    np.savetxt("pseaac_features.txt", features, fmt="%.5f")
    print("\nFull feature vector saved to 'pseaac_features.txt'")


if __name__ == "__main__":
    main()
