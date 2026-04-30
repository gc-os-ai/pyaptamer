"""Shared greedy tokenizer helper."""

__author__ = ["Iskaban10"]
__all__ = ["greedy_tokenize_sequence"]


def greedy_tokenize_sequence(
    seq: str,
    words: dict[str, int],
    word_max_len: int,
    max_len: int | None = None,
    unknown_token: int | None = 0,
) -> list[int]:
    """Tokenize a single string using greedy longest-match against a vocabulary.

    Scans ``seq`` from left to right. At each position, attempts to match the
    longest possible substring (up to ``word_max_len`` characters) that exists
    in ``words``. On a successful match, the corresponding index is appended to
    the token list and the scan advances by the matched length. On failure:

    - if ``unknown_token`` is not ``None``, appends ``unknown_token`` and
      advances by one character;
    - if ``unknown_token`` is ``None``, silently skips the character (advances
      by one without appending anything).

    Tokenization stops early once ``max_len`` tokens have been collected (if
    provided). No padding is applied — the caller is responsible for padding
    the returned list to a uniform length.

    Parameters
    ----------
    seq : str
        Input string to tokenize.
    words : dict[str, int]
        Vocabulary mapping substrings (k-mers) to unique integer indices.
        Index ``0`` is reserved as the unknown/padding token and should not
        appear as a value in ``words``.
    word_max_len : int
        Maximum substring length to attempt during matching.
    max_len : int or None, optional, default=None
        If provided, tokenization stops once this many tokens are collected.
        The returned list will have at most ``max_len`` entries.
    unknown_token : int or None, optional, default=0
        Token appended when no vocabulary match is found at the current
        position. Pass ``None`` to silently skip unmatched characters instead
        of recording them (the behaviour used by :func:`seq2vec`).

    Returns
    -------
    list[int]
        List of token indices. Length is at most ``max_len`` (if given).

    Examples
    --------
    >>> from pyaptamer.utils._greedy_tokenize import greedy_tokenize_sequence
    >>> words = {"A": 1, "C": 2, "AC": 3}
    >>> greedy_tokenize_sequence("ACX", words, word_max_len=2)
    [3, 0]
    >>> greedy_tokenize_sequence("ACX", words, word_max_len=2, unknown_token=None)
    [3]
    >>> greedy_tokenize_sequence("ACACAC", words, word_max_len=2, max_len=2)
    [3, 3]
    """
    tokens: list[int] = []
    i = 0
    n = len(seq)

    while i < n:
        matched = False

        for pattern_len in range(min(word_max_len, n - i), 0, -1):
            pattern = seq[i : i + pattern_len]
            if pattern in words:
                tokens.append(words[pattern])
                i += pattern_len
                matched = True
                break

        if not matched:
            if unknown_token is not None:
                tokens.append(unknown_token)
            i += 1

        if max_len is not None and len(tokens) >= max_len:
            tokens = tokens[:max_len]
            break

    return tokens
