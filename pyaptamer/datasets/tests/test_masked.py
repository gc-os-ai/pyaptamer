import numpy as np
import pytest

from pyaptamer.datasets.dataclasses._masked import MaskedDataset


def test_masked_dataset_y_initialization():
    """
    Test that y_masked is correctly initialized from the y array, 
    not the x array (regression test for PR #601).
    """
    # Create input and target arrays with distinctly different values
    x = np.array([[1, 2, 3, 4, 0]])
    y = np.array([[5, 6, 7, 8, 0]])
    
    # We set masked_rate to 1.0 to ensure mask_positions is populated
    dataset = MaskedDataset(x, y, max_len=5, mask_idx=9, masked_rate=1.0, is_rna=False)
    
    x_masked, y_masked, x_orig, y_orig = dataset[0]
    
    # At masked positions, y_masked should retain its original target values (from y)
    # At unmasked positions, y_masked is explicitly set to 0.
    # We verify that any non-zero value in y_masked must come from y, not x.
    y_masked_np = y_masked.numpy()
    y_orig_np = y_orig.numpy()
    x_orig_np = x_orig.numpy()
    
    non_zero_mask = y_masked_np != 0
    assert np.any(non_zero_mask), "Expected at least one non-zero value in y_masked after masking"
    
    assert np.array_equal(y_masked_np[non_zero_mask], y_orig_np[non_zero_mask])
    
    # Explicitly check that it didn't clone from x
    assert not np.array_equal(y_masked_np[non_zero_mask], x_orig_np[non_zero_mask])
