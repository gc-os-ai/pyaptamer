# Benchmarking

{class}`~pyaptamer.benchmarking.Benchmarking` fits each estimator in
cross-validation and reports the mean train and test score of every metric.
The estimators get the same folds, so their scores are comparable.

## Compare estimators

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.benchmarking import Benchmarking
from pyaptamer.datasets import load_li2014

X, y = load_li2014(split="test", return_X_y=True)
X = X.to_dataframe()
y = (y["label"] == "positive").astype(int).to_numpy()

bench = Benchmarking(
    estimators=[DummyClassifier(strategy="most_frequent"), AptaMCTSPipeline()],
    metrics=[accuracy_score, f1_score],
    X=X,
    y=y,
    cv=StratifiedKFold(3, shuffle=True, random_state=0),
)
bench.run().round(3)
```

```text
                                 train   test
estimator        metric
DummyClassifier  accuracy_score   0.75  0.750
                 f1_score         0.00  0.000
AptaMCTSPipeline accuracy_score   1.00  0.755
                 f1_score         1.00  0.102
```

`estimators` take `fit` and `predict`. `metrics` are callables
`(y_true, y_pred) -> float`, such as the functions in `sklearn.metrics`.
`cv` is a fold count or any `scikit-learn` splitter, and `None` means five
folds. The result is a `DataFrame` indexed by estimator and metric. The
estimator name is the class name, so pass one estimator per class.

`X` must be a `DataFrame`, not a loader, because the folds are taken by
`scikit-learn`. Call `to_dataframe()` on a loader first. Labels must be
numbers for metrics such as `f1_score`.

The baseline above predicts the majority class and shows what the accuracy
of a classifier is worth. Three of four pairs in this data are negative, so
an accuracy of 0.75 means nothing on its own.

## Fixed train and test split

To score on a given test set, pass a `PredefinedSplit`. Rows marked `-1`
are always in training, rows marked `0` form the test fold.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit

X_train, y_train = load_li2014(split="train", return_X_y=True)
X_test, y_test = load_li2014(split="test", return_X_y=True)

X = pd.concat([X_train.to_dataframe(), X_test.to_dataframe()], ignore_index=True)
labels = pd.concat([y_train["label"], y_test["label"]], ignore_index=True)
y = (labels == "positive").astype(int).to_numpy()
test_fold = np.r_[np.full(len(y_train), -1), np.zeros(len(y_test))]

bench = Benchmarking(
    estimators=[AptaMCTSPipeline()],
    metrics=[accuracy_score, f1_score],
    X=X,
    y=y,
    cv=PredefinedSplit(test_fold),
)
bench.run().round(3)
```

```text
                                 train   test
estimator        metric
AptaMCTSPipeline accuracy_score    1.0  0.816
                 f1_score          1.0  0.415
```
