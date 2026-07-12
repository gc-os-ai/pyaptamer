"""
Example demonstrating the enhanced Benchmarking API.

This example shows:
1. Reusability: Using the same Benchmarking instance across different datasets
2. String-based metrics: Using sklearn metric names like "accuracy"
3. Estimator naming: Handling multiple instances of the same model class
4. Custom names: Providing custom names for estimators
"""

import numpy as np
from sklearn.model_selection import PredefinedSplit

from pyaptamer.aptanet import AptaNetPipeline
from pyaptamer.benchmarking import Benchmarking

# Sample data
aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

# Example 1: Reusability across different datasets
print("=" * 60)
print("Example 1: Reusability across different datasets")
print("=" * 60)

clf = AptaNetPipeline(k=4)
bench = Benchmarking(estimators=[clf], metrics=["accuracy"])

# First dataset
X1 = [(aptamer_seq, protein_seq) for _ in range(40)]
y1 = np.array([0] * 20 + [1] * 20, dtype=np.float32)
test_fold1 = np.ones(len(y1), dtype=int) * -1
test_fold1[-5:] = 0
cv1 = PredefinedSplit(test_fold1)

print("\nDataset 1 (40 samples):")
results1 = bench.run(X=X1, y=y1, cv=cv1)
print(results1)

# Second dataset (reusing the same bench instance)
X2 = [(aptamer_seq, protein_seq) for _ in range(30)]
y2 = np.array([1] * 15 + [0] * 15, dtype=np.float32)
test_fold2 = np.ones(len(y2), dtype=int) * -1
test_fold2[-3:] = 0
cv2 = PredefinedSplit(test_fold2)

print("\nDataset 2 (30 samples, reusing same Benchmarking instance):")
results2 = bench.run(X=X2, y=y2, cv=cv2)
print(results2)

# Example 2: String-based metrics
print("\n" + "=" * 60)
print("Example 2: String-based metrics (sklearn conventions)")
print("=" * 60)

X = [(aptamer_seq, protein_seq) for _ in range(40)]
y = np.array([0] * 20 + [1] * 20, dtype=np.float32)
test_fold = np.ones(len(y), dtype=int) * -1
test_fold[-5:] = 0
cv = PredefinedSplit(test_fold)

bench_str = Benchmarking(
    estimators=[clf],
    metrics=["accuracy", "f1"]  # String-based metric names
)
results = bench_str.run(X=X, y=y, cv=cv)
print(results)

# Example 3: Handling multiple instances of the same model class
print("\n" + "=" * 60)
print("Example 3: Automatic naming for multiple instances")
print("=" * 60)

clf1 = AptaNetPipeline(k=3)
clf2 = AptaNetPipeline(k=4)
clf3 = AptaNetPipeline(k=5)

bench_multi = Benchmarking(
    estimators=[clf1, clf2, clf3],
    metrics=["accuracy"]
)
results = bench_multi.run(X=X, y=y, cv=cv)
print(results)
print("\nNote: Estimators are automatically named to avoid collisions:")
print(results.index.get_level_values("estimator").unique().tolist())

# Example 4: Custom estimator names
print("\n" + "=" * 60)
print("Example 4: Custom estimator names")
print("=" * 60)

bench_custom = Benchmarking(
    estimators=[
        ("k=3", AptaNetPipeline(k=3)),
        ("k=4", AptaNetPipeline(k=4)),
        ("k=5", AptaNetPipeline(k=5)),
    ],
    metrics=["accuracy"]
)
results = bench_custom.run(X=X, y=y, cv=cv)
print(results)
print("\nCustom names used:")
print(results.index.get_level_values("estimator").unique().tolist())
