import pytest
import torch

from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig


@pytest.mark.parametrize("seq_len", [1, 2])
def test_short_sequences(seq_len):
    """
    Test that AptaTrans handles very short sequences without crashing.
    This is a regression test for the issue where conv1 with kernel_size=3
    and no padding would crash on sequences shorter than 3.
    """
    # Configure tiny encoders
    apta_config = EncoderPredictorConfig(
        num_embeddings=10, target_dim=8, max_len=seq_len
    )
    prot_config = EncoderPredictorConfig(
        num_embeddings=10, target_dim=8, max_len=seq_len
    )

    # Initialize AptaTrans
    model = AptaTrans(apta_config, prot_config, in_dim=32, n_heads=4)
    model.eval()

    # Input sequences
    x_apta = torch.randint(0, 10, (1, seq_len))
    x_prot = torch.randint(0, 10, (1, seq_len))

    # This should not crash
    with torch.no_grad():
        output = model(x_apta, x_prot)

    assert output.shape == (1, 1)
