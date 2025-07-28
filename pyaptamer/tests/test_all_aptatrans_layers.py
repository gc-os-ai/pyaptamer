"""Test suite for all AptaTrans' architectural components and layers."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn

from pyaptamer.aptatrans.layers._convolutional import conv3x3, ConvBlock
from pyaptamer.aptatrans.layers._encoder import (
    EncoderPredictorConfig, 
    PositionalEncoding, 
    TokenPredictor
)
from pyaptamer.aptatrans.layers._interaction_map import InteractionMap


@pytest.mark.parametrize("in_c, out_c", [(8, 16), (16, 32), (32, 64)]) 
def test_conv3x3(in_c, out_c): 
    """Check correct initialization of a 3x3 convolution layer.""" 
    conv = conv3x3(in_c, out_c) 
    assert isinstance(conv, nn.Conv2d) 
    assert conv.in_channels == in_c 
    assert conv.out_channels == out_c 
    assert conv.kernel_size == (3, 3) 
    assert conv.padding == "same"

@pytest.mark.parametrize( 
    "in_c, out_c, pooling, x", 
    [(8, 16, None, torch.randn(1, 8, 32, 32)), 
     (16, 32, None, torch.randn(2, 16, 64, 64)), 
     (32, 64, None, torch.randn(1, 32, 128, 128))] 
) 
def test_convblock_without_pooling(in_c, out_c, pooling, x): 
    """Check ConvBlock() works as intended without pooling.""" 
    convblock = ConvBlock(in_c=in_c, out_c=out_c, pooling=pooling) 

    assert isinstance(convblock.block, nn.Sequential) 
    assert len(convblock.block) == 6 # without pooling

    # check types in the container 
    expected_types = [ 
        nn.Conv2d, 
        nn.BatchNorm2d, 
        nn.GELU, 
        nn.Conv2d, 
        nn.BatchNorm2d, 
        nn.GELU, 
    ] 
    assert all( 
        isinstance(convblock.block[i], expected_types[i]) 
        for i in range(len(convblock.block)) 
    )

    # check forward pass 
    out = convblock(x) 
    # no pooling, no downsampling
    assert out.shape == (x.shape[0], out_c, x.shape[2], x.shape[3])

@pytest.mark.parametrize( 
    "in_c, out_c, pooling, x", 
    [
        (8, 16, nn.MaxPool2d(2), torch.randn(1, 8, 32, 32)), 
        (16, 32, nn.MaxPool2d(2), torch.randn(2, 16, 64, 64)), 
        (32, 64, nn.MaxPool2d(2), torch.randn(1, 32, 128, 128))
    ] 
) 
def test_convblock_with_pooling(in_c, out_c, pooling, x): 
    """Check ConvBlock() works as intended with pooling.""" 
    convblock = ConvBlock(in_c=in_c, out_c=out_c, pooling=pooling) 

    assert isinstance(convblock.block, nn.Sequential) 
    assert len(convblock.block) == 7 # with pooling

    # check types in the container 
    expected_types = [ 
        type(pooling), 
        nn.Conv2d, 
        nn.BatchNorm2d, 
        nn.GELU, 
        nn.Conv2d, 
        nn.BatchNorm2d, 
        nn.GELU, 
    ] 
    assert all( 
        isinstance(convblock.block[i], expected_types[i]) 
        for i in range(len(convblock.block)) 
    )

    # check forward pass 
    out = convblock(x) 
    # downsampling due to pooling
    assert out.shape == (x.shape[0], out_c, x.shape[2]//2, x.shape[3]//2)

@pytest.mark.parametrize(
    "num_embeddings, target_dim, max_len",
    [(100, 50, 512), (1000, 200, 1024), (5000, 1000, 2048)]
)
def test_embedding_config(num_embeddings, target_dim, max_len):
    """Check correct initialization of EncoderPredictorConfig() dataclass."""
    config = EncoderPredictorConfig(
        num_embeddings=num_embeddings,
        target_dim=target_dim,
        max_len=max_len
    )
    assert config.num_embeddings == num_embeddings
    assert config.target_dim == target_dim
    assert config.max_len == max_len


@pytest.mark.parametrize(
    "d_model, dropout, max_len, x",
    [
        (128, 0.0, 512, torch.randn(10, 2, 128)),
        (256, 0.1, 1024, torch.randn(50, 4, 256)),
        (512, 0.2, 2048, torch.randn(100, 8, 512))
    ]
)
def test_positional_encoding(d_model, dropout, max_len, x):
    """Check PositionalEncoding() works as intended."""
    pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
    
    assert isinstance(pe.dropout, nn.Dropout)
    assert pe.dropout.p == dropout
    
    # check positional encoding buffer
    assert hasattr(pe, 'pe')
    assert 'pe' in dict(pe.named_buffers())
    assert 'pe' not in dict(pe.named_parameters())
    assert pe.pe.shape == (max_len, 1, d_model)
    
    # check forward pass
    out = pe(x)
    assert out.shape == x.shape
    
    # check that positional encoding was added (output != input when dropout=0)
    if dropout == 0.0:
        assert not torch.allclose(out, x)


@pytest.mark.parametrize(
    "d_model, d_out_mt, d_out_ss, x_mt, x_ss",
    [(128, 5, 3, torch.randn(2, 10, 128), torch.randn(2, 10, 128)),
     (256, 10, 5, torch.randn(4, 50, 256), torch.randn(4, 50, 256)),
     (512, 20, 8, torch.randn(8, 100, 512), torch.randn(8, 100, 512))]
)
def test_token_predictor(d_model, d_out_mt, d_out_ss, x_mt, x_ss):
    """Check TokenPredictor() works as intended."""
    predictor = TokenPredictor(d_model=d_model, d_out_mt=d_out_mt, d_out_ss=d_out_ss)
    
    # check linear layers
    assert isinstance(predictor.fc_mt, nn.Linear)
    assert isinstance(predictor.fc_ss, nn.Linear)
    assert predictor.fc_mt.in_features == d_model
    assert predictor.fc_mt.out_features == d_out_mt
    assert predictor.fc_ss.in_features == d_model
    assert predictor.fc_ss.out_features == d_out_ss
    
    # check forward pass
    out_mt, out_ss = predictor(x_mt, x_ss)
    assert out_mt.shape == (x_mt.shape[0], x_mt.shape[1], d_out_mt)
    assert out_ss.shape == (x_ss.shape[0], x_ss.shape[1], d_out_ss)

@pytest.mark.parametrize(
    "x_apta, x_prot",
    [(torch.randn(2, 10, 128), torch.randn(2, 15, 128)),
     (torch.randn(4, 50, 256), torch.randn(4, 75, 256)),
     (torch.randn(8, 100, 512), torch.randn(8, 150, 512))]
)
def test_interaction_map(x_apta, x_prot):
    """Check InteractionMap() works as intended."""
    interaction_map = InteractionMap()
    
    # check batch normalization layer
    assert isinstance(interaction_map.batchnorm, nn.BatchNorm2d)
    assert interaction_map.batchnorm.num_features == 1
    
    # check forward pass
    out = interaction_map(x_apta, x_prot)
    assert isinstance(out, torch.Tensor)
    # shape should be (batch_size, 1, seq_len_apta, seq_len_prot)
    assert out.shape == (x_apta.shape[0], 1, x_apta.shape[1], x_prot.shape[1])


@pytest.mark.parametrize(
    "x_apta, x_prot",
    [(torch.randn(2, 10, 128), torch.randn(2, 15, 256)),
     (torch.randn(4, 50, 256), torch.randn(4, 75, 512))]
)
def test_interaction_map_mismatch_input_features(x_apta, x_prot):
    """Check InteractionMap() raises error for mismatched feature dimensions."""
    interaction_map = InteractionMap()
    
    # should raise assertion error due to mismatched features
    with pytest.raises(
        AssertionError, 
        match="The number of features of `x_apta` and `x_prot` must match"
    ):
        interaction_map(x_apta, x_prot)