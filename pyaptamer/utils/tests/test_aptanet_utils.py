import numpy as np
import pytest
from pyaptamer.pseaac import AptaNetPSeAAC, PSeAAC
from pyaptamer.utils._aptanet_utils import pairs_to_features

def test_pairs_to_features_default_behavior():
    """
    Test that default behavior uses 18 properties (6 groups).
    For a k=4, there are 340 k-mer features (4+16+64+256) and for PSeAAC
    with default lambda=30, each group produces 20 + 30 = 50 features.
    With 6 groups, that's 300 features. 
    Total features expected: 340 + 300 = 640.
    Before this issue fix, it would have been 340 + 350 = 690.
    """
    aptamer_seq = "ACGTACGTACGT"
    protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM" # Length 31 (must be > lambda=30)
    
    pairs = [(aptamer_seq, protein_seq)]
    feats = pairs_to_features(pairs, k=4)
    
    assert feats.shape == (1, 640), f"Expected shape (1, 640), got {feats.shape}"

def test_pairs_to_features_custom_pseaac():
    """
    Test that a custom PSeAAC instance can be passed and is used correctly.
    We pass a generic PSeAAC with only 1 property group (3 properties).
    This should produce 50 PSeAAC features (instead of 300).
    Total features expected: 340 + 50 = 390.
    """
    aptamer_seq = "ACGTACGTACGT"
    protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    
    pairs = [(aptamer_seq, protein_seq)]
    custom_pseaac = PSeAAC(prop_indices=[0, 1, 2], group_props=3)
    feats = pairs_to_features(pairs, k=4, pseaac=custom_pseaac)
    
    assert feats.shape == (1, 390), f"Expected shape (1, 390), got {feats.shape}"

def test_pairs_to_features_output_type():
    """
    Test that the output type is unchanged and is a float32 numpy array.
    """
    aptamer_seq = "ACGT"
    protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    
    pairs = [(aptamer_seq, protein_seq)]
    feats = pairs_to_features(pairs)
    
    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32
