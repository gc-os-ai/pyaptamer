from pyaptamer.utils._aptatrans_utils import seq2vec


def test_seq2vec_basic():
    words = {"AA": 1, "AC": 2, "A": 3}
    sequences = (["AAAC"], ["HHHC"])

    padded_seq, padded_ss = seq2vec(sequences, words, seq_max_len=4)

    assert padded_seq.shape == (1, 4)
    assert padded_ss.shape == (1, 4)
    assert padded_seq[0][0] == 1
    assert padded_seq[0][1] == 2


def test_seq2vec_empty():
    words = {"A": 1}
    sequences = ([], [])

    padded_seq, padded_ss = seq2vec(sequences, words, seq_max_len=5)

    assert padded_seq.shape == (0, 5)


def test_seq2vec_long_sequence():
    words = {"A": 1, "C": 2}
    sequences = (["AAAA"], ["HHHH"])
    padded, ss = seq2vec(sequences, words, seq_max_len=2)
    assert padded.shape == (2, 2)


def test_seq2vec_empty_or_no_match():
    words = {"G": 1}
    sequences = (["AAA"], ["HHH"])
    padded, ss = seq2vec(sequences, words, seq_max_len=5)
    assert padded.shape == (0, 5)