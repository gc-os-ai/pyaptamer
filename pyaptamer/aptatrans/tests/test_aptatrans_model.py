"""Tests for the AptaTrans model itself (non-Lightning)."""

import os
import torch
import pytest

from pyaptamer.aptatrans import AptaTrans, EncoderPredictorConfig


def make_small_model():
    # very small embedding sizes so that the model is lightweight
    apta_embedding = EncoderPredictorConfig(num_embeddings=4, target_dim=2, max_len=8)
    prot_embedding = EncoderPredictorConfig(num_embeddings=4, target_dim=2, max_len=8)
    return AptaTrans(apta_embedding, prot_embedding, in_dim=4, n_encoder_layers=1, n_heads=1, conv_layers=[1,1,1])


def test_save_and_load_pretrained(tmp_path, monkeypatch):
    """Saving weights should write a file and loading from that path should restore state."""
    model = make_small_model()

    # make a deterministic change to parameters so we can detect load
    for p in model.parameters():
        p.data.fill_(1.234)

    save_path = tmp_path / "custom.pt"
    model.save_pretrained(str(save_path))
    assert save_path.exists()

    # create another model and ensure it's different
    new_model = make_small_model()
    # compare before loading: at least one parameter should differ
    assert any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(model.parameters(), new_model.parameters())
    )

    # load weights from file
    new_model.load_pretrained_weights(path=str(save_path))

    # after loading all parameters should match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_load_pretrained_download(monkeypatch, tmp_path):
    """When the local file is missing, ``load_pretrained_weights`` should call
    ``torch.hub.load_state_dict_from_url`` and load that dict.  We simulate the
    download by returning a fake state dict.
    """
    model = make_small_model()
    fake_state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    # ensure local path points to non-existent file
    missing = tmp_path / "nope.pt"
    assert not missing.exists()

    called = {}

    def fake_download(url, model_dir, map_location):
        called['url'] = url
        return fake_state

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_download)

    # call load_pretrained_weights with missing path
    model.load_pretrained_weights(path=str(missing))

    # after loading, model parameters should equal fake_state
    for k, v in model.state_dict().items():
        assert torch.allclose(v, fake_state[k])
    assert 'url' in called


def test_load_pretrained_invalid_path(monkeypatch, tmp_path):
    """If download also fails (raises), the exception should propagate."""
    model = make_small_model()
    missing = tmp_path / "nope2.pt"

    def raise_download(url, model_dir, map_location):
        raise RuntimeError("bad network")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", raise_download)

    with pytest.raises(RuntimeError):
        model.load_pretrained_weights(path=str(missing))
