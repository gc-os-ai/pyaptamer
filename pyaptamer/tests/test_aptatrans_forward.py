import torch
from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig


def test_forward_imap_shape():
    apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)

    model = AptaTrans(apta_embedding, prot_embedding, pretrained=False)

    x_apta = torch.randint(0, 16, (2, 10))
    x_prot = torch.randint(0, 16, (2, 12))

    imap = model.forward_imap(x_apta, x_prot)

    assert imap.dim() == 4
    assert imap.shape[1] == 1  # channel dimension must exist


def test_forward_runs_without_error():
    apta_embedding = EncoderPredictorConfig(128, 16, max_len=128)
    prot_embedding = EncoderPredictorConfig(128, 16, max_len=128)

    model = AptaTrans(apta_embedding, prot_embedding, pretrained=False)

    x_apta = torch.randint(0, 16, (2, 10))
    x_prot = torch.randint(0, 16, (2, 12))

    out = model(x_apta, x_prot)

    # Just ensure it runs
    assert out is not None