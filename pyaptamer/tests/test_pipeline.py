import numpy as np
from pyaptamer.aptanet import AptaNetPipeline

def test_fit_returns_self():
    aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

    X = [(aptamer, protein) for _ in range(10)]
    y = np.array([0, 1] * 5, dtype=np.float32)

    pipe = AptaNetPipeline()
    result = pipe.fit(X, y)

    assert result is pipe