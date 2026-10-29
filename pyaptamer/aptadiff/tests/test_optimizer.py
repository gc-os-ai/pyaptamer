"""Test the AptaDiff Bayesian optimization loop."""

__author__ = ["aditi-dsi"]

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator, AptaDiffOptimizer
from pyaptamer.bayesian_optimization import BayesianOptimizer
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_sample_fastq
from pyaptamer.raptgen import CNNPHMMVAE
from pyaptamer.transformations.transform import PrimerTrimmer

SEQ_LEN = 40
EMBED_SIZE = 2
N_CLUSTERS = 3
BATCH_SIZE = 4
BOUNDS = (-4, 4)
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
PROPOSAL_COLUMNS = ["sequence", "z0", "z1", "rz0", "rz1", "drift"]
PEAK = np.array([1.5, -2.0])


def _activity(Z) -> np.ndarray:  # noqa: N803
    """Return a synthetic binding activity for latent coordinates."""
    Z = np.asarray(Z, dtype=float)
    return -((Z - PEAK) ** 2).sum(axis=1)


def _observations(n_side: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return a grid of measured coordinates inside `BOUNDS` and their activity."""
    axis = np.linspace(-3.0, 3.0, n_side)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, EMBED_SIZE)
    return grid, _activity(grid)


def _vae() -> CNNPHMMVAE:
    """Return a an instance of pretrained VAE."""
    torch.manual_seed(0)
    return CNNPHMMVAE(motif_len=SEQ_LEN, embed_size=EMBED_SIZE)


def _trimmer() -> PrimerTrimmer:
    """Return a trimmer configured for the expected SELEX reads."""
    return PrimerTrimmer(
        start_primer=START_PRIMER,
        end_primer=END_PRIMER,
        variable_length=SEQ_LEN,
    )


def _aptadiff(**kwargs) -> AptaDiff:
    """Return an AptaDiff instance."""
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


def _optimizer(aptadiff: AptaDiff, **kwargs) -> AptaDiffOptimizer:
    """Return an AptaDiffOptimizer instance."""
    defaults = {"bounds": BOUNDS, "batch_size": BATCH_SIZE, "random_state": 0}
    return AptaDiffOptimizer(aptadiff, **{**defaults, **kwargs})


def _is_valid_output(sequence: str) -> bool:
    """Check if a generated sequence matches the expected length and classes."""
    return len(sequence) == SEQ_LEN and set(sequence) <= VALID_CLASSES


def _is_inside(Z: np.ndarray, bounds=BOUNDS) -> bool:  # noqa: N803
    """Check if every coordinate lies inside the search range."""
    low, high = bounds
    return bool((Z >= low).all() and (Z <= high).all())


@pytest.fixture(scope="module")
def reads() -> MoleculeLoader:
    """Return sample SELEX reads."""
    return load_sample_fastq()


@pytest.fixture(scope="module")
def fitted(reads: MoleculeLoader) -> AptaDiff:
    """Return an AptaDiff fitted on the sample reads."""
    return _aptadiff().fit(reads)


def test_synthetic_activity_peaks_where_expected() -> None:
    """Verify that the synthetic lab scoring function assigns
    the highest score to the PEAK coordinate.
    """
    X, y = _observations()

    assert _activity([PEAK])[0] == pytest.approx(0.0)
    assert y.max() < 0
    assert np.linalg.norm(X[np.argmax(y)] - PEAK) <= np.linalg.norm(
        X[np.argmin(y)] - PEAK
    )


def test_initial_design_decodes_the_cluster_centers(fitted: AptaDiff) -> None:
    """The first round decodes one candidate per cluster center."""
    optimizer = _optimizer(fitted)
    first = optimizer.initial_design()

    assert list(first.columns) == PROPOSAL_COLUMNS
    assert len(first) == N_CLUSTERS
    assert all(_is_valid_output(sequence) for sequence in first["sequence"])
    np.testing.assert_allclose(first[["z0", "z1"]].to_numpy(), fitted.gmm_centers_)

    pd.testing.assert_frame_equal(optimizer.proposals_, first)
    np.testing.assert_allclose(optimizer.drift_, first["drift"].to_numpy())
    assert not hasattr(optimizer, "next_points_")


def test_propose_returns_a_batch_inside_bounds(fitted: AptaDiff) -> None:
    """A round proposes `batch_size` candidates within the search range."""
    X, y = _observations()
    optimizer = _optimizer(fitted)
    proposals = optimizer.propose(X, y)

    assert list(proposals.columns) == PROPOSAL_COLUMNS
    assert len(proposals) == BATCH_SIZE
    assert optimizer.next_points_.shape == (BATCH_SIZE, EMBED_SIZE)
    assert _is_inside(optimizer.next_points_)
    assert all(_is_valid_output(sequence) for sequence in proposals["sequence"])
    np.testing.assert_allclose(
        proposals[["z0", "z1"]].to_numpy(), optimizer.next_points_
    )


def test_propose_maximises_the_activity_it_is_given(fitted: AptaDiff) -> None:
    """Binding activities are passed on, with the optimizer set to maximise it."""
    X, y = _observations()
    optimizer = _optimizer(fitted, kappa=1.5)
    optimizer.propose(X, y)

    reference = BayesianOptimizer(
        bounds=BOUNDS,
        batch_size=BATCH_SIZE,
        minimize=False,
        kappa=1.5,
        random_state=0,
    ).suggest(X, y)

    np.testing.assert_allclose(optimizer.next_points_, reference)

    negated = BayesianOptimizer(
        bounds=BOUNDS,
        batch_size=BATCH_SIZE,
        minimize=False,
        kappa=1.5,
        random_state=0,
    ).suggest(X, -y)

    assert not np.allclose(reference, negated)


def test_proposals_move_towards_the_peak(fitted: AptaDiff) -> None:
    """Verify that proposed coordinates move closer to the highest-scoring
    PEAK coordinate.
    """
    X, y = _observations(n_side=5)
    optimizer = _optimizer(fitted, kappa=0.0)
    optimizer.propose(X, y)

    proposed = np.linalg.norm(optimizer.next_points_ - PEAK, axis=1).min()
    measured = np.linalg.norm(X - PEAK, axis=1).mean()

    assert proposed < measured


def test_process_runs_over_several_rounds(fitted: AptaDiff) -> None:
    """Rounds can be chained by scoring each batch and feeding it back in."""
    X, y = _observations()
    optimizer = _optimizer(fitted)
    best_measured = y.max()

    for _ in range(3):
        proposals = optimizer.propose(X, y)

        assert len(proposals) == BATCH_SIZE
        assert _is_inside(optimizer.next_points_)

        X = np.vstack([X, optimizer.next_points_])
        y = np.concatenate([y, _activity(optimizer.next_points_)])

    assert len(X) == 16 + 3 * BATCH_SIZE
    assert y.max() > np.median(y[:16])
    assert y.max() >= best_measured


def test_same_seed_reproduces_a_round(fitted: AptaDiff) -> None:
    """A round is repeatable, down to the sequences it returns."""
    X, y = _observations()

    first = _optimizer(fitted).propose(X, y)
    second = _optimizer(fitted).propose(X, y)

    pd.testing.assert_frame_equal(first, second)


def test_drift_measures_the_round_trip(fitted: AptaDiff) -> None:
    """Verify that the calculated drift is the exact Euclidean distance
    between proposed and re-embedded coordinates.
    """
    optimizer = _optimizer(fitted)
    proposals = optimizer.initial_design()

    proposed = proposals[["z0", "z1"]].to_numpy()
    reembedded = proposals[["rz0", "rz1"]].to_numpy()
    expected = np.linalg.norm(proposed - reembedded, axis=1)

    np.testing.assert_allclose(proposals["drift"].to_numpy(), expected)
    assert np.all(proposals["drift"] >= 0)
    assert np.all(np.isfinite(proposals["drift"]))


def test_drift_leaves_the_trimmer_in_place(fitted: AptaDiff) -> None:
    """The trimmer is only skipped while encoding the generated sequences."""
    trimmer = fitted.trimmer_
    _optimizer(fitted).initial_design()

    assert fitted.trimmer_ is trimmer


def test_encode_sequences_reads_raw_reads(
    fitted: AptaDiff, reads: MoleculeLoader
) -> None:
    """With `encode_sequences`, reads are encoded instead of taken as coordinates."""
    coordinates = fitted.transform(reads)
    activity = _activity(coordinates)

    optimizer = _optimizer(fitted, encode_sequences=True)
    proposals = optimizer.propose(reads, activity)

    assert len(proposals) == BATCH_SIZE
    assert _is_inside(optimizer.next_points_)


def test_encode_sequences_reports_dropped_reads(
    fitted: AptaDiff, reads: MoleculeLoader
) -> None:
    """Check that if PrimerTrimmer drops reads during encoding, a ValueError
    is raised to prevent mismatched scores.
    """
    n_raw = len(reads.to_dataframe())
    optimizer = _optimizer(fitted, encode_sequences=True)

    assert len(fitted.transform(reads)) < n_raw

    with pytest.raises(ValueError, match="same number of rows"):
        optimizer.propose(reads, np.zeros(n_raw))


@pytest.mark.parametrize(
    "kwargs, X, y, match",
    [
        ({}, np.full((3, EMBED_SIZE), 9.0), np.zeros(3), "outside bounds"),
        ({}, np.zeros((3, EMBED_SIZE)), np.zeros(2), "same number of rows"),
        ({}, np.zeros((3, 5)), np.zeros(3), "embed_size_=2"),
        ({}, np.zeros(3), np.zeros(3), "two-dimensional"),
        (
            {"bounds": [(-1, 1)] * 3},
            np.zeros((3, EMBED_SIZE)),
            np.zeros(3),
            "one per",
        ),
        ({"bounds": (1, -1)}, np.zeros((3, EMBED_SIZE)), np.zeros(3), "low < high"),
    ],
)
def test_propose_rejects_bad_inputs(
    fitted: AptaDiff, kwargs: dict, X, y, match: str
) -> None:  # noqa: N803
    """Check if ValueError is raised for incorrect dimensions, mismatched lengths,
    or out-of-bounds coordinates."""
    with pytest.raises(ValueError, match=match):
        _optimizer(fitted, **kwargs).propose(X, y)


def test_out_of_bounds_error_names_the_rows(fitted: AptaDiff) -> None:
    """Check out-of-bounds error messages explicitly name the index of the bad row."""
    X = np.zeros((4, EMBED_SIZE))
    X[2] = 9.0

    with pytest.raises(ValueError, match=r"row\(s\) \[2\]"):
        _optimizer(fitted).propose(X, np.zeros(4))


@pytest.mark.parametrize("method", ["initial_design", "propose"])
def test_methods_require_a_fitted_aptadiff(method: str) -> None:
    """Both rounds need a fitted AptaDiff to decode with."""
    optimizer = _optimizer(_aptadiff())
    args = () if method == "initial_design" else _observations()

    with pytest.raises(NotFittedError):
        getattr(optimizer, method)(*args)


def test_get_params_and_set_params(fitted: AptaDiff) -> None:
    """The optimizer follows scikit-base's constructor conventions."""
    optimizer = _optimizer(fitted)

    assert set(optimizer.get_params(deep=False)) == {
        "aptadiff",
        "bounds",
        "batch_size",
        "kappa",
        "random_state",
        "encode_sequences",
    }

    optimizer.set_params(batch_size=2)

    assert optimizer.batch_size == 2
    assert len(optimizer.propose(*_observations())) == 2
