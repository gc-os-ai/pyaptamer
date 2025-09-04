# --- imports
from sklearn.metrics import accuracy_score, f1_score

# AptaNetPipeline import (cover common locations)
from pyaptamer.aptanet import AptaNetPipeline
from pyaptamer.benchmarking import Benchmarking
from pyaptamer.benchmarking.preprocessors.aptanet import AptaNetPreprocessor
from pyaptamer.datasets import load_csv_dataset

# --- load raw dataset (aptamer/protein/y)
df = load_csv_dataset("train_li2014")

# --- preprocessor
pre = AptaNetPreprocessor()

# --- AptaNet model/pipeline (use defaults; tweak if you like)
clf = AptaNetPipeline()
df = df.rename(columns={"label": "y"})

# 2. Map values
df["y"] = df["y"].map({"negative": 0, "positive": 1})

# --- set up the benchmark (auto split + stratify)
bench = Benchmarking(
    estimators=[clf],
    evaluators=[accuracy_score, f1_score],
    task="classification",
    preprocessor=pre,
    datasets=df,  # can also be [df1, df2] or {"my_ds": df}
    test_size=0.25,
    stratify=True,
    random_state=1337,
)

# --- run
results = bench.run()
