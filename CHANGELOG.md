# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AptaNet Hyperparameter Exposure**: Expose critical training parameters (`n_estimators`, `max_depth`, `optimizer`, `device`, `weight_decay`, `alpha`, `eps`) in `AptaNetClassifier` and `AptaNetRegressor`.
- **Benchmarking Enhancements**: Added `return_raw` parameter to `Benchmarking.run()` to allow extraction of per-fold scores, enabling deeper integration with `sktime`.
- **Benchmarking Labels**: Added `labels` parameter to `Benchmarking` constructor for custom estimator naming.
- **Robust Integration Tests**: Added test suites for hyperparameter propagation and benchmarking API validation.

### Fixed
- **MaskedDataset Initialization**: Fixed a critical bug where `y_masked` was incorrectly initialized from input features instead of target labels.
- **skbase Compliance**: Corrected `GreedyEncoder.get_test_params` to use `@classmethod` decorator, fixing scikit-base contract compliance.
- **AptaTrans Pretraining Pipeline**: Aligned `MaskedDataset` return order with `AptaTransEncoderLightning` batch expectations.
- **Docstring Polish**: Updated AptaNet docstrings to follow `numpydoc` standards and fixed default value inconsistencies.
