# tests/test_aptable.py
# Author: Dashpreet Singh <dashpreetsinghhanda@gmail.com>
#
# Tests for the aptable module: AptaBLE model, CrossAttentionInteractionMap,
# and AptaBLELightning.
#
# Run: python -m pytest pyaptamer/aptable/tests/test_aptable.py -v
# No GPU, no pretrained weights required.

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pyaptamer.aptable import AptaBLE, AptaBLELightning, CrossAttentionInteractionMap
from pyaptamer.aptable.layers._cross_attention_map import CrossAttentionInteractionMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aptable(in_dim=32, n_heads=4):
    from pyaptamer.aptatrans.layers._encoder import EncoderPredictorConfig
    cfg_a = EncoderPredictorConfig(num_embeddings=16, target_dim=8, max_len=32)
    cfg_p = EncoderPredictorConfig(num_embeddings=25, target_dim=8, max_len=32)
    return AptaBLE(
        cfg_a, cfg_p,
        in_dim=in_dim,
        n_encoder_layers=1,
        n_heads=n_heads,
        cross_attention_heads=n_heads,
        conv_layers=[1, 1, 1],
    )


# ---------------------------------------------------------------------------
# CrossAttentionInteractionMap
# ---------------------------------------------------------------------------

class TestCrossAttentionInteractionMap(unittest.TestCase):

    def setUp(self):
        self.imap = CrossAttentionInteractionMap(d_model=32, n_heads=4)
        self.imap.eval()

    def test_output_shape(self):
        x_a = torch.randn(2, 10, 32)
        x_p = torch.randn(2, 15, 32)
        out = self.imap(x_a, x_p)
        self.assertEqual(out.shape, (2, 1, 10, 15))

    def test_asymmetric_seq_len(self):
        x_a = torch.randn(3, 8, 32)
        x_p = torch.randn(3, 20, 32)
        out = self.imap(x_a, x_p)
        self.assertEqual(out.shape, (3, 1, 8, 20))

    def test_no_nan(self):
        x_a = torch.randn(2, 10, 32)
        x_p = torch.randn(2, 12, 32)
        out = self.imap(x_a, x_p)
        self.assertFalse(torch.isnan(out).any())

    def test_wrong_d_model_raises(self):
        with self.assertRaises(ValueError):
            CrossAttentionInteractionMap(d_model=33, n_heads=4)

    def test_wrong_feature_dim_raises(self):
        x_a = torch.randn(2, 10, 64)   # wrong
        x_p = torch.randn(2, 10, 32)
        with self.assertRaises(ValueError):
            self.imap(x_a, x_p)

    def test_return_attention_shapes(self):
        x_a = torch.randn(2, 10, 32)
        x_p = torch.randn(2, 12, 32)
        imap, attn_a2p, attn_p2a = self.imap(x_a, x_p, return_attention=True)
        self.assertEqual(imap.shape, (2, 1, 10, 12))
        self.assertEqual(attn_a2p.shape, (2, 10, 12))
        self.assertEqual(attn_p2a.shape, (2, 12, 10))

    def test_attention_weights_sum_to_one(self):
        x_a = torch.randn(2, 8, 32)
        x_p = torch.randn(2, 10, 32)
        _, attn_a2p, _ = self.imap(x_a, x_p, return_attention=True)
        sums = attn_a2p.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_gradient_flows(self):
        imap = CrossAttentionInteractionMap(d_model=32, n_heads=4)
        x_a = torch.randn(2, 8, 32, requires_grad=True)
        x_p = torch.randn(2, 8, 32, requires_grad=True)
        out = imap(x_a, x_p)
        out.sum().backward()
        self.assertFalse(torch.isnan(x_a.grad).any())

    def test_padding_mask_accepted(self):
        x_a = torch.randn(2, 10, 32)
        x_p = torch.randn(2, 12, 32)
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[:, -2:] = True
        out = self.imap(x_a, x_p, prot_key_padding_mask=mask)
        self.assertEqual(out.shape, (2, 1, 10, 12))


# ---------------------------------------------------------------------------
# AptaBLE model
# ---------------------------------------------------------------------------

class TestAptaBLE(unittest.TestCase):

    def setUp(self):
        self.model = _make_aptable()
        self.model.eval()

    def test_forward_shape(self):
        x_a = torch.randint(1, 16, (2, 10))
        x_p = torch.randint(1, 25, (2, 12))
        out = self.model(x_a, x_p)
        self.assertEqual(out.shape, (2, 1))

    def test_output_in_zero_one(self):
        x_a = torch.randint(1, 16, (2, 10))
        x_p = torch.randint(1, 25, (2, 12))
        out = self.model(x_a, x_p)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_no_nan(self):
        x_a = torch.randint(1, 16, (2, 10))
        x_p = torch.randint(1, 25, (2, 12))
        out = self.model(x_a, x_p)
        self.assertFalse(torch.isnan(out).any())

    def test_forward_imap_shape(self):
        x_a = torch.randint(1, 16, (2, 10))
        x_p = torch.randint(1, 25, (2, 12))
        imap = self.model.forward_imap(x_a, x_p)
        self.assertEqual(imap.shape[0], 2)
        self.assertEqual(imap.shape[1], 1)

    def test_forward_imap_return_attention(self):
        x_a = torch.randint(1, 16, (2, 10))
        x_p = torch.randint(1, 25, (2, 12))
        result = self.model.forward_imap(x_a, x_p, return_attention=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        imap, attn_a2p, attn_p2a = result
        self.assertEqual(imap.shape[1], 1)
        self.assertEqual(attn_a2p.shape[0], 2)

    def test_imap_is_cross_attention(self):
        self.assertIsInstance(self.model.imap, CrossAttentionInteractionMap)

    def test_invalid_in_dim_raises(self):
        from pyaptamer.aptatrans.layers._encoder import EncoderPredictorConfig
        cfg = EncoderPredictorConfig(16, 8, 32)
        with self.assertRaises(ValueError):
            AptaBLE(cfg, cfg, in_dim=33, n_heads=4)

    def test_gradient_flows_end_to_end(self):
        model = _make_aptable()
        x_a = torch.randint(1, 16, (2, 8))
        x_p = torch.randint(1, 25, (2, 10))
        out = model(x_a, x_p)
        out.sum().backward()
        for name, p in model.imap.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No grad for imap.{name}")

    def test_forward_encoder_apta(self):
        x_mlm = torch.randint(1, 16, (2, 10))
        x_ssp = torch.randint(1, 16, (2, 10))
        out_mt, out_ss = self.model.forward_encoder((x_mlm, x_ssp), "apta")
        self.assertEqual(out_mt.shape[0], 2)

    def test_forward_encoder_invalid_type_raises(self):
        x = torch.randint(1, 16, (2, 10))
        with self.assertRaises(ValueError):
            self.model.forward_encoder((x, x), "invalid")

    def test_batch_size_one(self):
        x_a = torch.randint(1, 16, (1, 10))
        x_p = torch.randint(1, 25, (1, 12))
        out = self.model(x_a, x_p)
        self.assertEqual(out.shape, (1, 1))

    def test_load_aptatrans_encoders(self):
        from pyaptamer.aptatrans._model import AptaTrans
        from pyaptamer.aptatrans.layers._encoder import EncoderPredictorConfig
        cfg_a = EncoderPredictorConfig(16, 8, 32)
        cfg_p = EncoderPredictorConfig(25, 8, 32)
        aptatrans = AptaTrans(
            cfg_a, cfg_p, in_dim=32, n_encoder_layers=1,
            n_heads=4, conv_layers=[1, 1, 1], pretrained=False
        )
        aptable = AptaBLE(
            cfg_a, cfg_p, in_dim=32, n_encoder_layers=1,
            n_heads=4, cross_attention_heads=4, conv_layers=[1, 1, 1]
        )
        aptable.load_aptatrans_encoders(aptatrans.state_dict())
        # Encoder weights should now match AptaTrans
        for key in ["encoder_apta", "encoder_prot"]:
            for at_name, at_param in aptatrans.named_parameters():
                if at_name.startswith(key):
                    ab_param = dict(aptable.named_parameters())[at_name]
                    self.assertTrue(
                        torch.allclose(at_param, ab_param),
                        f"Mismatch in {at_name} after load_aptatrans_encoders"
                    )


# ---------------------------------------------------------------------------
# AptaBLELightning
# ---------------------------------------------------------------------------

class TestAptaBLELightning(unittest.TestCase):

    def setUp(self):
        self.model = _make_aptable()
        self.lit = AptaBLELightning(self.model)

    def test_configure_optimizers_returns_adam(self):
        opt = self.lit.configure_optimizers()
        self.assertIsInstance(opt, torch.optim.Adam)

    def test_token_predictor_excluded_from_optimizer(self):
        opt = self.lit.configure_optimizers()
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        for name, param in self.model.named_parameters():
            if "token_predictor" in name:
                self.assertNotIn(
                    id(param), opt_param_ids,
                    f"token_predictor param {name} should be excluded from optimizer"
                )

    def test_lr_propagated(self):
        lit = AptaBLELightning(self.model, lr=3e-4)
        opt = lit.configure_optimizers()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 3e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
