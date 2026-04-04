import numpy as np

from pyaptamer.utils import generate_nplets, seq2vec


def _seq2vec_legacy(
    sequence_list: tuple[list[str], list[str]],
    words: dict[str, int],
    seq_max_len: int,
    word_max_len: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation used to verify backward compatibility."""
    words_ss = generate_nplets(
        letters=["H", "B", "E", "G", "I", "T", "S", "-"], repeat=range(1, 4)
    )
    outputs = []
    outputs_ss = []
    for seq, ss in zip(*sequence_list, strict=False):
        output = []
        output_ss = []
        i = 0
        while i < len(seq):
            matched = False
            for j in range(word_max_len, 0, -1):
                if i + j <= len(seq):
                    substring = seq[i : i + j]
                    word_idx = words.get(substring, 0)
                    if word_idx != 0:
                        matched = True
                        output.append(word_idx)
                        output_ss.append(words_ss.get(ss[i : i + j], 0))
                        if len(output) == seq_max_len:
                            outputs.append(np.array(output))
                            outputs_ss.append(np.array(output_ss))
                            output = []
                            output_ss = []
                        i += j
                        break
            if not matched:
                i += 1
        if len(output) > 0:
            outputs.append(np.array(output))
            outputs_ss.append(np.array(output_ss))
    if outputs:
        padded_outputs = np.zeros((len(outputs), seq_max_len))
        padded_outputs_ss = np.zeros((len(outputs_ss), seq_max_len))
        for idx, (seq_array, ss_array) in enumerate(
            zip(outputs, outputs_ss, strict=False)
        ):
            seq_len = len(seq_array)
            padded_outputs[idx, :seq_len] = seq_array
            padded_outputs_ss[idx, :seq_len] = ss_array
        return padded_outputs, padded_outputs_ss
    return np.zeros((0, seq_max_len)), np.zeros((0, seq_max_len))


def test_seq2vec_matches_legacy_behavior():
    """Ensure the optimized implementation matches the old behavior."""
    words = {
        "A": 1,
        "C": 2,
        "G": 3,
        "U": 4,
        "AA": 5,
        "CG": 6,
        "ACG": 7,
    }
    sequences = (
        ["AAACGUX", "NNNACG", "ACGACGACG"],
        ["HHHSSS-", "TTTHHH", "HHHBBBEEE"],
    )
    legacy_seq, legacy_ss = _seq2vec_legacy(
        sequence_list=sequences,
        words=words,
        seq_max_len=4,
        word_max_len=3,
    )
    new_seq, new_ss = seq2vec(
        sequence_list=sequences,
        words=words,
        seq_max_len=4,
        word_max_len=3,
    )

    np.testing.assert_array_equal(new_seq, legacy_seq)
    np.testing.assert_array_equal(new_ss, legacy_ss)


def test_seq2vec_empty_vocab_guard():
    """Empty or effectively empty vocabularies should return empty outputs."""
    sequences = (["ACGU"], ["HHHH"])
    out_seq, out_ss = seq2vec(
        sequence_list=sequences,
        words={},
        seq_max_len=5,
    )

    np.testing.assert_array_equal(out_seq, np.zeros((0, 5)))
    np.testing.assert_array_equal(out_ss, np.zeros((0, 5)))


def test_seq2vec_skips_unknown_characters():
    """Unknown sequence regions should be skipped and not create tokens."""
    words = {"AC": 1, "GU": 2}
    sequences = (["XXACYYGUZZ"], ["--HH--TT--"])

    out_seq, out_ss = seq2vec(
        sequence_list=sequences,
        words=words,
        seq_max_len=4,
        word_max_len=2,
    )
    np.testing.assert_array_equal(out_seq, np.array([[1.0, 2.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(out_ss, np.array([[9.0, 54.0, 0.0, 0.0]]))


def test_seq2vec_shorter_than_pattern():
    """Sequence shorter than any usable token should produce empty output rows."""
    words = {"AC": 1, "GU": 2}
    sequences = (["A", "G"], ["H", "T"])
    out_seq, out_ss = seq2vec(
        sequence_list=sequences,
        words=words,
        seq_max_len=3,
        word_max_len=2,
    )
    np.testing.assert_array_equal(out_seq, np.zeros((0, 3)))
    np.testing.assert_array_equal(out_ss, np.zeros((0, 3)))


def test_seq2vec_chunking_and_padding():
    """Long outputs should be chunked to seq_max_len and the tail padded."""
    words = {"A": 1}
    sequences = (["AAAAA"], ["HHHHH"])
    out_seq, out_ss = seq2vec(
        sequence_list=sequences,
        words=words,
        seq_max_len=2,
        word_max_len=1,
    )

    np.testing.assert_array_equal(
        out_seq,
        np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        out_ss,
        np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )
