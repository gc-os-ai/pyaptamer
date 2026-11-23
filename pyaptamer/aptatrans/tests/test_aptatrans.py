"""Test suite for AptaTrans' deep neural network and pipeline."""

__author__ = ["nennomp", "siddharth7113"]

from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptatrans import (
    AptaTrans,
    AptaTransClassifier,
    AptaTransPipeline,
    EncoderPredictorConfig,
)
from pyaptamer.data import MoleculeLoader

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
SMALL_NET = {
    "in_dim": 8,
    "n_heads": 2,
    "n_encoder_layers": 1,
    "conv_layers": [1, 1, 1],
    "max_epochs": 1,
    "random_state": 0,
}


class TestAptaTransModel:
    """Tests for the AptaTrans() class."""

    @pytest.fixture
    def embeddings(self) -> tuple[EncoderPredictorConfig, EncoderPredictorConfig]:
        """Create dummy embeddings configurations for testing purposes."""
        embedding = EncoderPredictorConfig(
            num_embeddings=16,
            target_dim=16,
            max_len=16,
        )
        return (embedding, embedding)

    def test_init_input_dim_not_divisible_by_heads(
        self,
        embeddings: tuple[EncoderPredictorConfig, EncoderPredictorConfig],
    ):
        """
        Check that an AssertionError is raised if input dimension is not divisible by
        the number of heads.
        """
        with pytest.raises(
            AssertionError,
            match="Input dimension 128 must be divisible by number of heads 3.",
        ):
            AptaTrans(
                apta_embedding=embeddings[0],
                prot_embedding=embeddings[1],
                in_dim=128,
                n_heads=3,
            )

    @pytest.mark.parametrize(
        "batch_size, seq_len_apta, seq_len_prot, in_dim",
        [(2, 50, 100, 128), (4, 100, 150, 256), (8, 75, 125, 512)],
    )
    def test_forward_encoder(self, batch_size, seq_len_apta, seq_len_prot, in_dim):
        """Check forward_encoder() produces correct outputs for pretraining."""
        apta_embedding = EncoderPredictorConfig(
            num_embeddings=125, target_dim=8, max_len=seq_len_apta
        )
        prot_embedding = EncoderPredictorConfig(
            num_embeddings=1000, target_dim=12, max_len=seq_len_prot
        )
        model = AptaTrans(
            apta_embedding=apta_embedding,
            prot_embedding=prot_embedding,
            in_dim=in_dim,
        )

        x_apta_mlm = torch.randint(0, 125, (batch_size, seq_len_apta))
        x_apta_ss = torch.randint(0, 125, (batch_size, seq_len_apta))
        x_prot_mlm = torch.randint(0, 1000, (batch_size, seq_len_prot))
        x_prot_ss = torch.randint(0, 1000, (batch_size, seq_len_prot))

        # check output shapes for aptamer predictions
        y_apta_mlm, y_apta_ss = model.forward_encoder(
            x=(x_apta_mlm, x_apta_ss), encoder_type="apta"
        )
        assert y_apta_mlm.shape == (batch_size, seq_len_apta, 125)
        assert y_apta_ss.shape == (batch_size, seq_len_apta, 8)

        # check output shapes for protein predictions
        y_prot_mlm, y_prot_ss = model.forward_encoder(
            x=(x_prot_mlm, x_prot_ss), encoder_type="prot"
        )
        assert y_prot_mlm.shape == (batch_size, seq_len_prot, 1000)
        assert y_prot_ss.shape == (batch_size, seq_len_prot, 12)

    @pytest.mark.parametrize(
        "batch_size, seq_len_apta, seq_len_prot, in_dim",
        [(2, 50, 100, 128), (4, 100, 150, 256), (8, 75, 125, 512)],
    )
    def test_forward_imap(self, batch_size, seq_len_apta, seq_len_prot, in_dim):
        """Check forward_imap() computes interaction map correctly."""
        apta_embedding = EncoderPredictorConfig(
            num_embeddings=125, target_dim=8, max_len=seq_len_apta
        )
        prot_embedding = EncoderPredictorConfig(
            num_embeddings=1000, target_dim=12, max_len=seq_len_prot
        )
        model = AptaTrans(
            apta_embedding=apta_embedding,
            prot_embedding=prot_embedding,
            in_dim=in_dim,
        )

        x_apta = torch.randint(high=125, size=(batch_size, seq_len_apta))
        x_prot = torch.randint(high=1000, size=(batch_size, seq_len_prot))

        imap = model.forward_imap(x_apta, x_prot)

        assert isinstance(imap, torch.Tensor)
        assert imap.shape == (batch_size, 1, seq_len_apta, seq_len_prot)

    @pytest.mark.parametrize(
        "device, batch_size, in_dim, seq_len",
        [
            (torch.device("cpu"), 4, 32, 10),
            pytest.param(
                torch.device("cuda"),
                4,
                32,
                10,
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    @torch.no_grad()
    def test_forward(
        self,
        embeddings: tuple[EncoderPredictorConfig, EncoderPredictorConfig],
        device: torch.device,
        batch_size: int,
        in_dim: int,
        seq_len: int,
    ) -> None:
        """Check forward pass on specified device."""
        aptatrans = AptaTrans(
            apta_embedding=embeddings[0],
            prot_embedding=embeddings[1],
            in_dim=in_dim,
            n_encoder_layers=2,
            n_heads=4,
            conv_layers=[2, 2, 2],
            dropout=0.1,
        ).to(device)

        # dummy input tensors
        x_apta = torch.randint(
            high=embeddings[0].num_embeddings,
            size=(batch_size, seq_len),
            dtype=torch.long,
        ).to(device)
        x_prot = torch.randint(
            high=embeddings[1].num_embeddings,
            size=(batch_size, seq_len),
            dtype=torch.long,
        ).to(device)

        # forward pass
        output = aptatrans(x_apta, x_prot)

        assert output.shape == (batch_size, 1)
        # output should be in [0, 1] (sigmoid activation)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
        assert not torch.allclose(output[0], output[1], atol=1e-5)

    @pytest.mark.parametrize(
        "batch_size, seq_len_apta, seq_len_prot, in_dim",
        [
            # Asymmetric sequence lengths
            (2, 10, 20, 32),
            (3, 20, 10, 32),
            (4, 15, 25, 64),
        ],
    )
    @torch.no_grad()
    def test_forward_output_shape_asymmetric(
        self, batch_size: int, seq_len_apta: int, seq_len_prot: int, in_dim: int
    ) -> None:
        """Check forward() output is (batch, 1) for asymmetric sequence lengths.

        Regression test for the spurious squeeze(dim=2) in forward(): the interaction
        map from forward_imap is already (batch, 1, s1, s2) — the shape conv1 expects.
        """
        apta_embedding = EncoderPredictorConfig(
            num_embeddings=16, target_dim=8, max_len=seq_len_apta
        )
        prot_embedding = EncoderPredictorConfig(
            num_embeddings=16, target_dim=8, max_len=seq_len_prot
        )
        model = AptaTrans(
            apta_embedding=apta_embedding,
            prot_embedding=prot_embedding,
            in_dim=in_dim,
            n_encoder_layers=2,
            n_heads=4,
            conv_layers=[2, 2, 2],
        )

        x_apta = torch.randint(0, 16, (batch_size, seq_len_apta), dtype=torch.long)
        x_prot = torch.randint(0, 16, (batch_size, seq_len_prot), dtype=torch.long)

        output = model(x_apta, x_prot)

        assert output.shape == (batch_size, 1), (
            f"Expected ({batch_size}, 1), got {tuple(output.shape)}. "
            f"seq_len_apta={seq_len_apta}, seq_len_prot={seq_len_prot}"
        )

    @pytest.mark.parametrize("train", [True, False])
    def test_forward_row_of_only_padding_is_finite(self, train):
        """A sequence of only padding ids gives a finite score."""
        config = EncoderPredictorConfig(num_embeddings=16, target_dim=8, max_len=8)
        model = AptaTrans(config, config, in_dim=8, n_heads=2, n_encoder_layers=1)
        model.train(train)
        x_apta = torch.tensor([[0] * 8, [3, 4, 0, 0, 0, 0, 0, 0]])
        x_prot = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]] * 2)

        with torch.no_grad():
            output = model(x_apta, x_prot)

        assert torch.isfinite(output).all()

    def test_eval_forward_without_nested_tensor_support(self, monkeypatch):
        """Evaluation runs where nested tensors are missing, as on Apple MPS."""

        def unsupported(*args, **kwargs):
            raise NotImplementedError("nested tensors are not supported")

        monkeypatch.setattr(torch, "_nested_tensor_from_mask_left_aligned", unsupported)
        config = EncoderPredictorConfig(num_embeddings=16, target_dim=8, max_len=8)
        model = AptaTrans(config, config, in_dim=8, n_heads=2, n_encoder_layers=1)
        model.eval()
        x = torch.tensor([[3, 4, 5, 0, 0, 0, 0, 0]] * 2)

        with torch.no_grad():
            output = model(x, x)

        assert output.shape == (2, 1)


@parametrize_with_checks([AptaTransClassifier(apta_max_len=1, **SMALL_NET)])
def test_sklearn_compatible_classifier(estimator, check):
    """AptaTransClassifier passes the scikit-learn estimator checks."""
    check(estimator)


def test_classifier_token_id_above_vocabulary_raises():
    """A token id outside the vocabulary raises a ValueError naming the parameter."""
    X = np.array([[1, 1], [200, 1]])
    clf = AptaTransClassifier(apta_max_len=1, **SMALL_NET)
    with pytest.raises(ValueError, match="n_apta_tokens"):
        clf.fit(X, [0, 1])


def _ids(n_rows=8, n_cols=16):
    """Token-id rows with ids 1 to 19 and alternating labels."""
    rng = np.random.default_rng(0)
    return rng.integers(1, 20, size=(n_rows, n_cols)), np.array([0, 1] * (n_rows // 2))


def test_classifier_same_random_state_reproduces_the_fit():
    """The same random_state gives the same probabilities; another seed differs."""
    X, y = _ids()

    def proba(seed):
        params = {**SMALL_NET, "random_state": seed}
        return AptaTransClassifier(apta_max_len=8, **params).fit(X, y).predict_proba(X)

    np.testing.assert_array_equal(proba(0), proba(0))
    assert not np.array_equal(proba(0), proba(1))


def test_classifier_n_epochs_counts_the_epochs_run():
    """Without early stopping, n_epochs_ is max_epochs."""
    X, y = _ids()
    params = {**SMALL_NET, "max_epochs": 3}
    clf = AptaTransClassifier(apta_max_len=8, patience=None, **params).fit(X, y)
    assert clf.n_epochs_ == 3


def test_classifier_model_starts_from_the_given_network():
    """fit trains a copy of model with its sizes and leaves model unchanged."""
    config = EncoderPredictorConfig(num_embeddings=20, target_dim=8, max_len=8)
    model = AptaTrans(
        config, config, in_dim=8, n_heads=2, n_encoder_layers=1, conv_layers=[1, 1, 1]
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    X, y = _ids()
    clf = AptaTransClassifier(
        apta_max_len=8,
        n_apta_tokens=20,
        n_prot_tokens=20,
        model=model,
        max_epochs=1,
        random_state=0,
    ).fit(X, y)

    assert clf.model_ is not model
    fitted_shapes = {k: v.shape for k, v in clf.model_.state_dict().items()}
    assert fitted_shapes == {k: v.shape for k, v in before.items()}
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


@pytest.fixture(scope="module")
def pairs():
    """Eight aptamer-protein pairs with alternating labels."""
    X = MoleculeLoader(data={"aptamer": [APTAMER] * 8, "protein": [PROTEIN] * 8})
    return X, np.array([0, 1] * 4)


def _small_pipeline(**kwargs):
    """Build a pipeline with a network small enough to train in a test."""
    return AptaTransPipeline(
        apta_max_len=12,
        prot_max_len=10,
        estimator=AptaTransClassifier(apta_max_len=12, **SMALL_NET),
        **kwargs,
    )


@pytest.fixture(scope="module")
def fitted(pairs):
    """A small AptaTransPipeline fitted on pairs."""
    return _small_pipeline().fit(*pairs)


def test_pipeline_dna_and_rna_spelling_score_the_same(fitted):
    """An aptamer spelled with T scores the same as with U."""
    dna = pd.DataFrame({"aptamer": [APTAMER], "protein": [PROTEIN]})
    rna = pd.DataFrame({"aptamer": [APTAMER.replace("T", "U")], "protein": [PROTEIN]})
    np.testing.assert_allclose(fitted.predict_proba(dna), fitted.predict_proba(rna))


def test_pipeline_default_protein_words_are_the_bundled_ones(fitted):
    """With prot_words=None the protein encoder uses the 713 bundled words."""
    words = fitted.pipeline_["features"].named_transformers_["protein"].words
    assert len(words) == 713


def test_pipeline_default_classifier_vocabulary_follows_the_words():
    """The default classifier's protein vocabulary is the kept words, padding and
    mask, also with more than the 713 bundled words."""
    amino = "ACDEFGHIKLMNPQRSTVWY"
    words = {"".join(w): 2.0 for w in product(amino, repeat=2)}
    words.update({"".join(w): 2.0 for w in list(product(amino, repeat=3))[:400]})
    words["A"] = 0.0
    clf = AptaTransPipeline(prot_words=words)._build_pipeline()["clf"]
    assert clf.n_prot_tokens == 802


def test_pipeline_interaction_map_shape(fitted, pairs):
    """interaction_map gives one aptamer-by-protein map per row."""
    imap = fitted.interaction_map(pairs[0])
    assert imap.shape == (8, 1, 12, 10)


def test_pipeline_generate_returns_distinct_candidates_by_score(fitted):
    """generate returns n distinct candidates, highest score first."""
    candidates = fitted.generate(PROTEIN, n_candidates=2, depth=4, n_iterations=2)
    assert len({c[0] for c in candidates}) == 2
    assert [c[2] for c in candidates] == sorted(
        (c[2] for c in candidates), reverse=True
    )


def test_pipeline_generate_rejects_depth_below_three(fitted):
    """depth below 3 raises a ValueError, since no triplet fits."""
    with pytest.raises(ValueError, match="depth"):
        fitted.generate(PROTEIN, depth=2)
