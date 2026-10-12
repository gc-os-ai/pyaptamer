"""Test the AptaDiff end-to-end algorithm."""

__author__ = ["aditi-dsi"]

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.raptgen import CNNPHMMVAE
from pyaptamer.trafos.encode import SequenceOneHotEncoder
from pyaptamer.trafos.transform import PrimerTrimmer

SEQ_LEN = 40
EMBED_SIZE = 2
N_CLUSTERS = 3
VALID_READ_COUNT = 6
VALID_CLASSES = set("ATGC")
START_PRIMER = "TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG"
END_PRIMER = "TATGTGCGCATACATGGATCCTC"
GENERATOR_PARAMS = {
    "num_timesteps": 2,
    "dim": 32,
    "depth": 1,
    "max_epochs": 1,
    "random_state": 0,
}


class _TransformOnlyVAE:
    """Mock VAE model exposing only transform.

    Proves AptaDiff supports generic non-RaptGen models.
    """

    def __init__(self, embed_size: int) -> None:
        self.embed_size = embed_size

    def transform(self, X_idx) -> np.ndarray:  # noqa: N803
        idx = np.asarray(X_idx, dtype=np.float64)
        cols = [
            idx[:, i :: self.embed_size].sum(axis=1) for i in range(self.embed_size)
        ]
        return np.stack(cols, axis=1).astype(np.float32)


def _vae() -> CNNPHMMVAE:
    """Return a VAE representing a pretrained one."""
    torch.manual_seed(0)
    return CNNPHMMVAE(motif_len=SEQ_LEN, embed_size=EMBED_SIZE)


def _trimmer() -> PrimerTrimmer:
    """Return a trimmer configured for the expected SELEX reads."""
    return PrimerTrimmer(
        start_primer=START_PRIMER,
        end_primer=END_PRIMER,
        variable_length=SEQ_LEN,
    )


def _rna_encoder() -> SequenceOneHotEncoder:
    """Return an encoder for sequences, with T sharing U's index."""
    return SequenceOneHotEncoder(
        vocab={"A": 0, "U": 1, "G": 2, "C": 3, "T": 1},
        inverse_vocab={0: "A", 1: "U", 2: "G", 3: "C"},
        handle_unknown="drop",
    )


def _five_class_encoder() -> SequenceOneHotEncoder:
    """Return an encoder whose vocabulary holds five classes instead of four."""
    return SequenceOneHotEncoder(
        vocab={"A": 0, "T": 1, "G": 2, "C": 3, "N": 4},
        inverse_vocab={0: "A", 1: "T", 2: "G", 3: "C", 4: "N"},
        handle_unknown="drop",
    )


def _estimator(**kwargs) -> AptaDiff:
    """Return an AptaDiff instance, with any parameter overridable."""
    if "vae" not in kwargs:
        kwargs["vae"] = _vae()
    if "generator" not in kwargs:
        kwargs["generator"] = AptaDiffGenerator(**GENERATOR_PARAMS)
    defaults = {
        "trimmer": _trimmer(),
        "n_clusters": N_CLUSTERS,
        "gmm_restarts": 3,
        "random_state": 0,
    }
    return AptaDiff(**{**defaults, **kwargs})


def _read(random_region: str) -> str:
    """Wrap a random region in the standard SELEX primers."""
    return START_PRIMER + random_region + END_PRIMER


def _is_valid_output(sequence: str) -> bool:
    """Check if the generated sequence matches the expected test length and
    classes.
    """
    return len(sequence) == SEQ_LEN and set(sequence) <= VALID_CLASSES


@pytest.fixture(scope="module")
def reads() -> pd.DataFrame:
    """Return sample SELEX reads."""
    return load_sample_fastq()


@pytest.fixture(scope="module")
def fitted(reads: pd.DataFrame) -> AptaDiff:
    """Return an AptaDiff fitted on the sample reads."""
    return _estimator().fit(reads)


@pytest.mark.parametrize("n_samples", [1, 4])
def test_fit_and_sample(fitted: AptaDiff, n_samples: int) -> None:
    """Fitting populates the documented attributes and sample generates from them."""
    assert fitted.seq_len_ == SEQ_LEN
    assert fitted.embed_size_ == EMBED_SIZE
    assert fitted.latents_.shape == (VALID_READ_COUNT, EMBED_SIZE)
    assert fitted.sequences_.shape == (VALID_READ_COUNT, 1)
    assert fitted.gmm_centers_.shape == (N_CLUSTERS, EMBED_SIZE)
    assert fitted.encoder_.handle_unknown == "drop"

    generated = fitted.sample(n_samples)
    assert list(generated.columns) == ["sequence"]
    assert len(generated) == n_samples
    assert all(_is_valid_output(sequence) for sequence in generated["sequence"])


def test_reconstruct_matches_input_rows(fitted: AptaDiff, reads: pd.DataFrame) -> None:
    """Reconstructions are indexed by the reads they were generated from."""
    reconstructed = fitted.reconstruct(reads)

    assert list(reconstructed.index) == list(fitted.sequences_.index)
    assert all(_is_valid_output(sequence) for sequence in reconstructed["sequence"])


def test_rows_stay_aligned_across_dropped_reads() -> None:
    """Each latent coordinate corresponds to its own sequence when reads
    are dropped.

    The trimmer and the encoder drop different rows, so the surviving
    sequences and their coordinates must stay paired across both operations.
    """
    kept = ["ACGT" * 10, "TGCA" * 10, "GGCC" * 10]
    frame = pd.DataFrame(
        {
            "sequence": [
                _read(kept[0]),
                "TOOSHORT",
                _read(kept[1]),
                _read("ACGN" + "ACGT" * 9),
                _read(kept[2]),
            ]
        }
    )
    estimator = _estimator(n_clusters=2).fit(frame)

    assert list(estimator.sequences_.iloc[:, 0]) == kept
    for position, sequence in enumerate(kept):
        one_read = pd.DataFrame({"sequence": [_read(sequence)]})
        assert np.allclose(
            estimator.transform(one_read)[0], estimator.latents_[position], atol=1e-5
        )


def test_inverse_transform_accepts_external_latents(fitted: AptaDiff) -> None:
    """Latent coordinates chosen outside the estimator can generate sequences."""
    from_array = fitted.inverse_transform([[0.1, 0.2], [0.3, 0.4]])
    from_centers = fitted.inverse_transform(fitted.gmm_centers_)

    assert len(from_array) == 2
    assert len(from_centers) == N_CLUSTERS
    for generated in (from_array, from_centers):
        assert all(_is_valid_output(sequence) for sequence in generated["sequence"])


def test_score_matches_score_samples_mean(
    fitted: AptaDiff, reads: pd.DataFrame
) -> None:
    """score is the mean of one finite value per encoded sequence."""
    per_sequence = fitted.score_samples(reads)
    overall = fitted.score(reads)

    assert per_sequence.shape == (VALID_READ_COUNT,)
    assert np.isfinite(per_sequence).all()
    assert isinstance(overall, float)
    assert np.isclose(overall, per_sequence.mean())


@pytest.mark.parametrize("embed_size", [2, 4])
def test_embed_size_follows_the_vae(reads: pd.DataFrame, embed_size: int) -> None:
    """embed_size_ is read off the passed model's output, not hardcoded."""
    estimator = _estimator(vae=_TransformOnlyVAE(embed_size), n_clusters=2).fit(reads)

    assert estimator.embed_size_ == embed_size
    assert estimator.latents_.shape == (VALID_READ_COUNT, embed_size)
    assert len(estimator.sample(2)) == 2


def test_rejects_unusable_vae(reads: pd.DataFrame) -> None:
    """A model exposing neither accepted interface raises TypeError."""
    with pytest.raises(TypeError, match="transform.*encoder and h2mu"):
        _estimator(vae=object()).fit(reads)


def test_injected_encoder_governs_output_classes(reads: pd.DataFrame) -> None:
    """The passed encoder decides the classes of generated sequences."""
    estimator = _estimator(encoder=_rna_encoder()).fit(reads)
    generated = estimator.sample(2)["sequence"]

    chars = {char for sequence in generated for char in sequence}
    assert chars <= set("AUGC")
    assert "U" in chars and "T" not in chars
    assert all(len(sequence) == SEQ_LEN for sequence in generated)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"encoder": _five_class_encoder()}, "encoder produces 5 classes"),
        ({"gmm_restarts": 0}, "gmm_restarts must be at least 1"),
    ],
)
def test_fit_guards(reads: pd.DataFrame, kwargs: dict, match: str) -> None:
    """Misconfigured components are refused with a specific message."""
    with pytest.raises(ValueError, match=match):
        _estimator(**kwargs).fit(reads)


def test_trimmer_none_rejects_variable_lengths() -> None:
    """Without a trimmer, input of uneven lengths is refused."""
    frame = pd.DataFrame({"sequence": ["ACGT" * 10, "ACGT", "ACGT" * 5]})

    with pytest.raises(ValueError, match="variable lengths"):
        _estimator(trimmer=None).fit(frame)


def test_rejects_other_input() -> None:
    """Input that is neither a MoleculeLoader nor a DataFrame raises TypeError."""
    with pytest.raises(
        TypeError, match="MoleculeLoader instance or a pandas DataFrame"
    ):
        _estimator().fit([["ACGT"]])


def test_transform_rejects_other_length() -> None:
    """Reads of a length other than the fitted one are refused."""
    frame = pd.DataFrame({"sequence": ["ACGT" * 10, "TGCA" * 10, "GGCC" * 10]})
    estimator = _estimator(trimmer=None, n_clusters=2).fit(frame)

    with pytest.raises(ValueError, match="fitted on length"):
        estimator.transform(pd.DataFrame({"sequence": ["ACGT"]}))


@pytest.mark.parametrize(
    "method",
    [
        "sample",
        "transform",
        "inverse_transform",
        "reconstruct",
        "score",
        "score_samples",
    ],
)
def test_methods_require_fitting(reads: pd.DataFrame, method: str) -> None:
    """Every public method refuses to run before fit."""
    if method == "inverse_transform":
        argument = np.zeros((2, EMBED_SIZE))
    elif method == "sample":
        argument = 1
    else:
        argument = reads

    with pytest.raises(NotFittedError):
        getattr(_estimator(), method)(argument)


def test_fit_leaves_vae_and_global_rng_untouched(reads: pd.DataFrame) -> None:
    """fit uses the VAE forward-only and does not change the global RNG state."""
    vae = _vae()
    weights = {name: value.clone() for name, value in vae.state_dict().items()}
    was_training = vae.training
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()

    _estimator(vae=vae).fit(reads)

    assert all(
        torch.equal(weights[name], value) for name, value in vae.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in vae.parameters())
    assert vae.training == was_training
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_random_state_is_reproducible(reads: pd.DataFrame) -> None:
    """The same seed gives the same mixture and samples, different gives different."""
    first = _estimator().fit(reads)
    second = _estimator().fit(reads)
    other = _estimator(random_state=1).fit(reads)

    assert np.allclose(first.latents_, second.latents_)
    assert np.allclose(first.gmm_centers_, second.gmm_centers_)
    assert list(first.sample(3)["sequence"]) == list(second.sample(3)["sequence"])
    assert not np.allclose(first.gmm_centers_, other.gmm_centers_)


def test_get_params_and_clone() -> None:
    """The estimator follows scikit-learn's constructor conventions."""
    estimator = _estimator()

    assert set(estimator.get_params(deep=False)) == {
        "vae",
        "trimmer",
        "encoder",
        "n_clusters",
        "gmm_restarts",
        "generator",
        "random_state",
    }

    estimator.set_params(n_clusters=5)
    cloned = clone(estimator)

    assert cloned.n_clusters == 5
    assert not hasattr(cloned, "latents_")
