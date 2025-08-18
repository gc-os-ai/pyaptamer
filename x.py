from pyaptamer.utils._deepaptamer_utils import run_deepdna_prediction


def main():
    seq = "GTACGTACGTACGTACGTACGTACGTACGTACGTA"  # length 35

    print(run_deepdna_prediction(seq))


if __name__ == "__main__":
    main()
