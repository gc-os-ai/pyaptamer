import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from pyaptamer.visualization import (
    plot_benchmark_results,
    plot_confusion_matrix,
    plot_interaction_map,
    plot_roc_curve,
    plot_training_curves,
)


class TestPlotConfusionMatrix:
    def test_basic(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        ax = plot_confusion_matrix(y_true, y_pred)
        assert ax is not None
        assert ax.get_title() == "Confusion Matrix"

    def test_normalized(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        ax = plot_confusion_matrix(y_true, y_pred, normalize=True)
        assert ax is not None

    def test_custom_labels(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        ax = plot_confusion_matrix(y_true, y_pred, labels=["Neg", "Pos"])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["Neg", "Pos"]


class TestPlotRocCurve:
    def test_basic(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        ax = plot_roc_curve(y_true, y_score)
        assert ax is not None
        assert "AUC" in ax.get_legend().get_texts()[0].get_text()

    def test_perfect_prediction(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.0, 0.0, 1.0, 1.0]
        ax = plot_roc_curve(y_true, y_score)
        assert "1.00" in ax.get_legend().get_texts()[0].get_text()


class TestPlotBenchmarkResults:
    @pytest.fixture()
    def benchmark_df(self):
        index = pd.MultiIndex.from_tuples(
            [("ModelA", "accuracy"), ("ModelA", "f1"), ("ModelB", "accuracy")],
            names=["estimator", "metric"],
        )
        return pd.DataFrame(
            {"train": [0.9, 0.85, 0.88], "test": [0.8, 0.75, 0.82]},
            index=index,
        )

    def test_all_metrics(self, benchmark_df):
        ax = plot_benchmark_results(benchmark_df)
        assert ax is not None

    def test_single_metric(self, benchmark_df):
        ax = plot_benchmark_results(benchmark_df, metric="accuracy")
        assert ax is not None


class TestPlotTrainingCurves:
    def test_train_only(self):
        ax = plot_training_curves([1.0, 0.8, 0.6, 0.4])
        assert ax is not None
        assert len(ax.get_lines()) == 1

    def test_train_and_val(self):
        ax = plot_training_curves([1.0, 0.8, 0.6], [0.9, 0.85, 0.7])
        assert len(ax.get_lines()) == 2


class TestPlotInteractionMap:
    def test_numpy_input(self):
        imap = np.random.rand(10, 20)
        ax = plot_interaction_map(imap)
        assert ax is not None

    def test_with_sequences(self):
        imap = np.random.rand(5, 8)
        ax = plot_interaction_map(imap, aptamer_seq="AGCTA", protein_seq="ACDEFGHI")
        assert ax is not None

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="2D"):
            plot_interaction_map(np.random.rand(5))

    def test_squeeze_batch_dim(self):
        imap = np.random.rand(1, 1, 5, 8)
        ax = plot_interaction_map(imap)
        assert ax is not None
