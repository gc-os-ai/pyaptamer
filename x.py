import numpy as np

from pyaptamer.utils._deepaptamer_utils import run_deepdna_prediction


def main():
    seq = "GTACGTACGTACGTACGTACGTACGTACGTACGTA"  # length 35
    features = ["MGW", "Roll", "ProT", "HelT"]
    layer = 2

    results = {}

    print(f"DNA sequence length: {len(seq)}\n")

    # Run predictions for each feature
    for feature in features:
        preds = run_deepdna_prediction(seq, feature, layer, mode="cpu")
        preds = np.array(preds)

        results[feature] = preds
        print(preds)
        print(f"{feature}: shape = {preds.shape}")

    # Flatten all features into one vector
    flattened = np.concatenate([results[f] for f in features])
    print("\nFlattened vector shape:", flattened.shape)


if __name__ == "__main__":
    main()
