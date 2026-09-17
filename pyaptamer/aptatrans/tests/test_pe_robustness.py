import pytest

from pyaptamer.aptatrans.layers._encoder import PositionalEncoding


def test_positional_encoding_odd_dimensions():
    """
    Test that PositionalEncoding handles odd-numbered model dimensions.
    Regression test for Issue where odd d_model caused shape mismatch crash.
    """
    # Test standard even dimension
    try:
        pe_even = PositionalEncoding(d_model=128, max_len=10)
        assert pe_even.pe.shape == (1, 10, 128)
    except Exception as e:
        pytest.fail(f"PositionalEncoding failed on even d_model: {e}")

    # Test odd dimension (previously crashed)
    try:
        pe_odd = PositionalEncoding(d_model=127, max_len=10)
        assert pe_odd.pe.shape == (1, 10, 127)
    except Exception as e:
        pytest.fail(f"PositionalEncoding failed on odd d_model: {e}")


def test_positional_encoding_small_dimensions():
    """Test very small dimensions."""
    pe_1 = PositionalEncoding(d_model=1, max_len=5)
    assert pe_1.pe.shape == (1, 5, 1)

    pe_2 = PositionalEncoding(d_model=2, max_len=5)
    assert pe_2.pe.shape == (1, 5, 2)
