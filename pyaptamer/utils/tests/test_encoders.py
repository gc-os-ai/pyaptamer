import pandas as pd
import pytest

from pyaptamer.trafos.encode import (
    KMerEncoder,
    PairsToFeaturesTransformer,
    PSeAACEncoder,
)


class TestKMerEncoder:
    def test_output_shape_k4(self):
        # k=4 produces 4+16+64+256 = 340 features
        encoder = KMerEncoder(k=4)
        X = pd.DataFrame({"seq": ["ACGTACGT", "GGGGAAAA"]})
        result = encoder.fit_transform(X)
        assert result.shape == (2, 340)

    def test_output_shape_k2(self):
        encoder = KMerEncoder(k=2)
        X = pd.DataFrame({"seq": ["ACGT"]})
        result = encoder.fit_transform(X)
        assert result.shape == (1, 20)  # 4 + 16 features

    def test_preserves_index(self):
        encoder = KMerEncoder(k=2)
        X = pd.DataFrame({"seq": ["ACGT", "GGAA"]}, index=["a", "b"])
        result = encoder.fit_transform(X)
        assert list(result.index) == ["a", "b"]

    def test_get_test_params(self):
        encoder = KMerEncoder()
        params = encoder.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0


class TestPSeAACEncoder:
    def test_output_shape(self):
        encoder = PSeAACEncoder(lambda_val=5)
        X = pd.DataFrame({"seq": ["ACDEFGHIKLMNPQRSTVWY" * 2]})
        result = encoder.fit_transform(X)
        assert result.shape[0] == 1
        # Default is 7 groups of 3 properties.
        # Shape should be (20 + 5) * 7 = 175
        assert result.shape[1] == 175

    def test_preserves_index(self):
        encoder = PSeAACEncoder(lambda_val=5)
        X = pd.DataFrame({"seq": ["ACDEFGHIKLMNPQRSTVWY" * 2]}, index=["protein1"])
        result = encoder.fit_transform(X)
        assert list(result.index) == ["protein1"]


class TestPairsToFeaturesTransformer:
    def test_with_named_columns(self):
        transformer = PairsToFeaturesTransformer(k=2, lambda_val=5)
        X = pd.DataFrame(
            {
                "aptamer": ["ACGT", "GGAA"],
                "protein": ["ACDEFGHIKLMNPQRSTVWY" * 2] * 2,
            }
        )
        result = transformer.fit_transform(X)
        assert result.shape[0] == 2
        # Kmer (k=2) = 20, PSeAAC = 175 -> 195
        assert result.shape[1] == 195

    def test_custom_column_names(self):
        transformer = PairsToFeaturesTransformer(
            k=2, aptamer_col="dna", protein_col="target", lambda_val=5
        )
        X = pd.DataFrame(
            {
                "dna": ["ACGT"],
                "target": ["ACDEFGHIKLMNPQRSTVWY" * 2],
            }
        )
        result = transformer.fit_transform(X)
        assert result.shape[0] == 1

    def test_missing_columns(self):
        transformer = PairsToFeaturesTransformer(k=2, lambda_val=5)
        X = pd.DataFrame({"wrong_col": ["ACGT"]})
        with pytest.raises(ValueError):
            transformer.fit_transform(X)


class TestDeprecationWarnings:
    def test_generate_kmer_vecs_warning(self):
        from pyaptamer.utils._aptanet_utils import generate_kmer_vecs

        with pytest.warns(DeprecationWarning, match="deprecated"):
            generate_kmer_vecs("ACGT")

    def test_pairs_to_features_warning(self):
        from pyaptamer.utils._aptanet_utils import pairs_to_features

        X = [("ACGT", "ACDEFGHIKLMNPQRSTVWY" * 2)]
        with pytest.warns(DeprecationWarning, match="deprecated"):
            pairs_to_features(X, k=2)
