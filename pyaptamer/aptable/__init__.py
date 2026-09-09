"""
AptaBLE-style architecture for aptamer-protein interaction prediction.

This module implements the symmetric bidirectional cross-attention
interaction map introduced in AptaBLE (NeurIPS 2024, January 2026),
as a self-contained extension to pyaptamer's existing AptaTrans module.

The core innovation over AptaTrans is replacing the dot-product
``InteractionMap`` with ``CrossAttentionInteractionMap``, which allows
aptamers and proteins to attend to each other's positions simultaneously.
This produces interaction maps that recapitulate experimental binding
interfaces, as demonstrated by AptaBLE on SARS-CoV-2 Spike glycoprotein
aptamers.

Intended Integration
--------------------
This module is structured identically to ``pyaptamer.aptatrans`` and
``pyaptamer.aptanet`` and is intended to be merged into the main
pyaptamer package under ``pyaptamer.aptable``.

The ``AptaBLE`` model is a drop-in replacement for ``AptaTrans``:
- Same encoder architecture (pretrained AptaTrans encoders are reusable)
- Same convolutional head
- Same output shape and interface
- Only the interaction map layer differs

Usage
-----
>>> from pyaptamer.aptable import AptaBLE, AptaBLELightning
>>> from pyaptamer.aptatrans import EncoderPredictorConfig
>>> apta_cfg = EncoderPredictorConfig(128, 16, max_len=275)
>>> prot_cfg = EncoderPredictorConfig(128, 16, max_len=1000)
>>> model = AptaBLE(apta_cfg, prot_cfg)
>>> # Load pretrained AptaTrans encoder weights (compatible)
>>> # model.load_aptatrans_encoders(aptatrans_state_dict)

References
----------
AptaBLE: Aptamer Binding and Likelihood Estimation.
NeurIPS 2024 workshop, updated January 2026.
"""

__author__ = ["DZDasherKTB"]
__all__ = [
    "AptaBLE",
    "AptaBLELightning",
    "CrossAttentionInteractionMap",
]

from pyaptamer.aptable._model import AptaBLE
from pyaptamer.aptable._model_lightning import AptaBLELightning
from pyaptamer.aptable.layers._cross_attention_map import CrossAttentionInteractionMap
