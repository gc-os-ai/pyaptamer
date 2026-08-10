"""Tests for APIDataset."""

import numpy as np
import pytest
import torch

from pyaptamer.datasets.dataclasses._api import APIDataset

APTA_MAX_LEN = 10
PROT_MAX_LEN = 5
PROT_WORDS = {"A": 1, "C": 2, "D": 3, "AC": 4}

X_APTA = np.array(["AUGCAUGC", "GCUAGCUA"])
X_PROT = np.array(["ACD", "DCA"])
Y = np.array(["positive", "negative"])


def make_dataset(split="train", x_apta=X_APTA, x_prot=X_PROT, y=Y):
    """Build an APIDataset from the module-level fixtures."""
    return APIDataset(
        x_apta=x_apta,
        x_prot=x_prot,
        y=y,
        apta_max_len=APTA_MAX_LEN,
        prot_max_len=PROT_MAX_LEN,
        prot_words=PROT_WORDS,
        split=split,
    )


def test_unknown_split_raises():
    """An unrecognised split is rejected before any data is prepared."""
    with pytest.raises(ValueError, match="Unknown split: valid"):
        make_dataset(split="valid")


@pytest.mark.parametrize(
    "split, expected_len",
    [
        ("train", 2 * len(X_APTA)),  # reversed sequences are appended
        ("test", len(X_APTA)),  # no augmentation
    ],
)
def test_split_controls_augmentation(split, expected_len):
    """Split train doubles the dataset with reversed aptamers, test does not."""
    dataset = make_dataset(split=split)
    assert len(dataset) == expected_len
    assert dataset.x_apta.shape == (expected_len, APTA_MAX_LEN)
    assert dataset.x_prot.shape == (expected_len, PROT_MAX_LEN)
    assert dataset.y.shape == (expected_len,)


def test_train_split_duplicates_proteins_and_labels():
    """Augmentation repeats x_prot and y so all three arrays stay aligned."""
    dataset = make_dataset(split="train")
    n = len(X_APTA)
    assert torch.equal(dataset.x_prot[:n], dataset.x_prot[n:])
    assert torch.equal(dataset.y[:n], dataset.y[n:])


def test_labels_are_encoded_as_binary():
    """Labels equal to positive map to 1 and anything else maps to 0."""
    dataset = make_dataset(split="test")
    assert torch.equal(dataset.y, torch.tensor([1, 0]))


def test_getitem_returns_aligned_triple():
    """__getitem__ returns the aptamer, protein and label at the same index."""
    dataset = make_dataset(split="test")
    x_apta, x_prot, y = dataset[1]
    assert torch.equal(x_apta, dataset.x_apta[1])
    assert torch.equal(x_prot, dataset.x_prot[1])
    assert torch.equal(y, dataset.y[1])
