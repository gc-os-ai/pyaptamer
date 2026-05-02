"""Tests for APIDataset class."""

import numpy as np
import pandas as pd
import pytest
import torch

from pyaptamer.datasets.dataclasses import APIDataset


class TestAPIDataset:
    """Tests for APIDataset."""

    @pytest.fixture
    def dummy_data(self):
        apta = np.array(["AUGC", "GCAU"])
        prot = np.array(["ACD", "EFG"])
        y = np.array(["positive", "negative"])
        words = {"ACD": 1, "EFG": 2}
        return apta, prot, y, words

    def test_init_train(self, dummy_data):
        """Test initialization with train split (augmentation)."""
        apta, prot, y, words = dummy_data
        ds = APIDataset(
            x_apta=apta,
            x_prot=prot,
            y=y,
            apta_max_len=10,
            prot_max_len=10,
            prot_words=words,
            split="train",
        )
        # Augmentation (reverse) doubles the size
        assert len(ds) == 4
        assert isinstance(ds[0][0], torch.Tensor)
        assert isinstance(ds[0][1], torch.Tensor)
        assert isinstance(ds[0][2], torch.Tensor)

    def test_init_test(self, dummy_data):
        """Test initialization with test split (no augmentation)."""
        apta, prot, y, words = dummy_data
        ds = APIDataset(
            x_apta=apta,
            x_prot=prot,
            y=y,
            apta_max_len=10,
            prot_max_len=10,
            prot_words=words,
            split="test",
        )
        assert len(ds) == 2

    def test_from_dataframe(self, dummy_data):
        """Test from_dataframe factory method."""
        apta, prot, y, words = dummy_data
        df = pd.DataFrame({"apta": apta, "prot": prot, "label": y})
        
        ds = APIDataset.from_dataframe(
            df=df,
            apta_col="apta",
            prot_col="prot",
            label_col="label",
            apta_max_len=10,
            prot_max_len=10,
            prot_words=words,
            split="test",
        )
        
        assert len(ds) == 2
        # Check that data was correctly passed
        # Label 'positive' becomes 1, 'negative' becomes 0
        assert ds[0][2].item() == 1
        assert ds[1][2].item() == 0

    def test_invalid_split(self, dummy_data):
        """Test that invalid split raises ValueError."""
        apta, prot, y, words = dummy_data
        with pytest.raises(ValueError, match="Unknown split"):
            APIDataset(apta, prot, y, 10, 10, words, split="val")
