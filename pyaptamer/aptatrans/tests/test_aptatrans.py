"""Test suite for AptaTrans' deep neural network and pipeline."""

__author__ = ["nennomp"]

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pyaptamer.aptatrans import AptaTrans, AptaTransPipeline, EncoderPredictorConfig


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


class MockAptaTransNeuralNet(nn.Module):
    """Mock AptaTrans model for testing pipeline."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        # mock embeddings with required attributes
        self.apta_embedding = type("MockEmbedding", (), {"max_len": 100})()
        self.prot_embedding = type("MockEmbedding", (), {"max_len": 150})()

    def forward_imap(self, x_apta, x_prot):
        batch_size = x_apta.shape[0]
        return torch.randn(
            batch_size,
            1,
            self.apta_embedding.max_len,
            self.prot_embedding.max_len,
            device=self.device,
        )

    def forward(self, x_apta, x_prot):
        # return deterministic scores based on input shapes
        batch_size = x_apta.shape[0]
        return torch.tensor([[0.8]], device=self.device).repeat(batch_size, 1)

    def to(self, device):
        self.device = device
        return self


class TestAptaTransPipeline:
    """Tests for the AptaTransPipeline() class."""

    @pytest.mark.parametrize(
        "device, prot_words",
        [
            (
                torch.device("cpu"),
                {"AAA": 0.5, "AAC": 0.3, "AAG": 0.8, "ACA": 0.2, "ACC": 0.9},
            ),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                {"CCC": 0.6, "CCG": 0.4, "CGC": 0.7, "GGG": 0.1, "GGC": 0.85},
            ),
        ],
    )
    def test_initialization(self, device, prot_words):
        """Check AptaTransPipeline() initializes correctly."""
        model = MockAptaTransNeuralNet(device)
        pipeline = AptaTransPipeline(device=device, model=model, prot_words=prot_words)

        assert isinstance(pipeline, AptaTransPipeline)
        assert pipeline.device.type == device.type
        assert pipeline.model is model
        assert pipeline.model.device.type == device.type

        # check word dictionaries
        assert isinstance(pipeline.prot_words, dict)

        # check protein words filtering (only above average frequency)
        mean_freq = sum(prot_words.values()) / len(prot_words)
        expected_prot_count = sum(1 for freq in prot_words.values() if freq > mean_freq)
        pipeline._init_vocabularies()
        assert len(pipeline.prot_words_) == expected_prot_count

    @pytest.mark.parametrize(
        "device, target",
        [
            (torch.device("cpu"), "AUGCAUGC"),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "GCUAGCUA",
            ),
        ],
    )
    def test_init_aptamer_experiment(self, device, target, monkeypatch):
        """Check _init_aptamer_experiment() initializes Aptamer experiment correctly."""
        model = MockAptaTransNeuralNet(device)
        prot_words = {"AUG": 0.8, "GCA": 0.6, "UGC": 0.4, "CUA": 0.2}
        pipeline = AptaTransPipeline(device=device, model=model, prot_words=prot_words)

        captured_args = {}

        class MockAptamer:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

        monkeypatch.setattr(
            "pyaptamer.aptatrans._pipeline.AptamerEvalAptaTrans", MockAptamer
        )

        experiment = pipeline._init_aptamer_experiment(target)

        assert isinstance(experiment, MockAptamer)
        assert captured_args["target"] == target
        assert captured_args["model"] is model
        assert captured_args["device"] == device

    @pytest.mark.parametrize(
        "device, X_train, y_train, X_test",
        [
            (
                torch.device("cpu"),
                pd.DataFrame(
                    {"aptamer": ["AUGCA", "GCUAG"], "protein": ["GCUAGCUA", "AUGCAUGC"]}
                ),
                np.array([1.0, 0.0]),
                pd.DataFrame({"aptamer": ["AUGCA"], "protein": ["GCUAGCUA"]}),
            ),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                pd.DataFrame(
                    {
                        "aptamer": ["GCUAG", "AUGCA", "CUAGC"],
                        "protein": ["AUGCAUGC", "GCUAGCUA", "UGCAUGCA"],
                    }
                ),
                np.array([0.0, 1.0, 0.5]),
                pd.DataFrame(
                    {"aptamer": ["GCUAG", "CUAGC"], "protein": ["AUGCAUGC", "UGCAUGCA"]}
                ),
            ),
            (
                torch.device("cpu"),
                pd.DataFrame(
                    {"aptamer": ["AUGCA", "GCUAG"], "protein": ["GCUAGCUA", "AUGCAUGC"]}
                ),
                np.array([1.0, 0.0]),
                np.array(
                    [
                        ["AUGCA", "GCUAGCUA"],
                    ]
                ),
            ),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                pd.DataFrame(
                    {
                        "aptamer": ["GCUAG", "AUGCA", "CUAGC"],
                        "protein": ["AUGCAUGC", "GCUAGCUA", "UGCAUGCA"],
                    }
                ),
                np.array([0.0, 1.0, 0.5]),
                np.array([["GCUAG", "AUGCAUGC"], ["CUAGC", "UGCAUGCA"]]),
            ),
        ],
    )
    def test_fit_and_predict(self, device, X_train, y_train, X_test, monkeypatch):
        """Check fit trains model and predict returns interaction scores correctly."""
        model = MockAptaTransNeuralNet(device)
        prot_words = {"AUG": 0.8, "GCA": 0.6, "UGC": 0.4, "CUA": 0.2}
        pipeline = AptaTransPipeline(device=device, model=model, prot_words=prot_words)

        class MockTrainer:
            def __init__(self, max_epochs, log_every_n_steps):
                pass

            def fit(self, model, dataloader):
                # skip training
                pass

        monkeypatch.setattr("pyaptamer.aptatrans._pipeline.L.Trainer", MockTrainer)

        # test fit
        result = pipeline.fit(X_train, y_train)

        assert result is pipeline

        # test predict
        scores = pipeline.predict(X_test)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X_test)
        assert scores.shape == (len(X_test),)

    def test_predict_raises_type_error_for_invalid_input(self):
        """Check predict raises TypeError for invalid input types."""
        device = torch.device("cpu")
        model = MockAptaTransNeuralNet(device)
        prot_words = {"AUG": 0.8, "GCA": 0.6, "UGC": 0.4, "CUA": 0.2}
        pipeline = AptaTransPipeline(device=device, model=model, prot_words=prot_words)

        invalid_input = [["AUGCA", "GCUAGCUA"], ["GCUAG", "AUGCAUGC"]]

        with pytest.raises(
            TypeError, match="X must be a pandas DataFrame or numpy.ndarray"
        ):
            pipeline.predict(invalid_input)

    @pytest.mark.parametrize(
        "device, target, n_candidates, depth",
        [
            (torch.device("cpu"), "AUGCAUGC", 3, 5),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "GCUAGCUA",
                5,
                10,
            ),
        ],
    )
    def test_recommend(self, device, target, n_candidates, depth, monkeypatch):
        """Check AptaTransPipeline.recommend() generates candidate aptamers."""
        # setup
        model = MockAptaTransNeuralNet(device)
        prot_words = {"AUG": 0.8, "GCA": 0.6, "UGC": 0.4, "CUA": 0.2}
        pipeline = AptaTransPipeline(
            device=device, model=model, prot_words=prot_words, depth=depth
        )

        # mock Aptamer experiment
        class MockExperiment:
            def evaluate(self, candidate):
                return torch.tensor(0.75)

        def mock_aptamer(**kwargs):
            return MockExperiment()

        monkeypatch.setattr(
            "pyaptamer.aptatrans._pipeline.AptamerEvalAptaTrans", mock_aptamer
        )

        # mock MCTS to return deterministic candidates
        class MockMCTS:
            def __init__(self, **kwargs):
                self.counter = 0

            def run(self, verbose: bool = False):
                # Generate some mock candidates
                candidates = [
                    (f"APTA{i:03d}", f"sequence_{i}", torch.tensor(i) / 10)
                    for i in range(10)
                ]

                candidate_data = candidates[self.counter % len(candidates)]
                self.counter += 1

                return {
                    "candidate": candidate_data[0],  # reconstructed candidate
                    "sequence": candidate_data[1],  # original sequence
                    "score": candidate_data[2],  # evaluation score
                }

        monkeypatch.setattr("pyaptamer.aptatrans._pipeline.MCTS", MockMCTS)

        # test recommendation
        candidates = pipeline.recommend(target=target, n_candidates=n_candidates)

        # check output
        assert isinstance(candidates, set)
        assert len(candidates) == n_candidates  # should be exactly n_candidates

    @pytest.mark.parametrize(
        "device, candidate, target",
        [
            (torch.device("cpu"), "AUGCA", "GCUAGCUA"),
            (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "GCUAG",
                "AUGCAUGC",
            ),
        ],
    )
    def test_get_interaction_map(self, device, candidate, target):
        """
        Check AptaTransPipeline.get_interaction_map() generates aptamer-protein
        interaction map.
        """
        model = MockAptaTransNeuralNet(torch.device(device))
        pipeline = AptaTransPipeline(
            device=torch.device(device), model=model, prot_words={"AAA": 0.5}
        )

        imap = pipeline.get_interaction_map(candidate, target)
        assert imap.shape == (
            1,
            1,
            model.apta_embedding.max_len,
            model.prot_embedding.max_len,
        )
