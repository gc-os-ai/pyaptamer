import numpy as np
import pytest
import torch
from torch import optim
from pyaptamer.aptanet import AptaNetClassifier, AptaNetRegressor

def test_aptanet_classifier_hyperparameter_propagation():
    """
    Integration test to ensure that exposed hyperparameters are correctly 
    propagated to the underlying selector and skorch model in AptaNetClassifier.
    """
    # Custom hyperparameters
    n_estimators = 50
    max_depth = 5
    lr = 0.001
    weight_decay = 0.01
    optimizer = optim.Adam
    
    clf = AptaNetClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        random_state=42
    )
    
    # Create dummy data
    X = np.random.rand(20, 10).astype(np.float32)
    y = np.random.randint(0, 2, 20).astype(np.float32)
    
    # Fit the classifier to build the internal pipeline
    clf.fit(X, y)
    
    # Access the internal pipeline
    pipeline = clf.pipeline_
    selector = pipeline.named_steps["select"]
    net = pipeline.named_steps["net"]
    
    # Check selector parameters
    assert selector.estimator.n_estimators == n_estimators
    assert selector.estimator.max_depth == max_depth
    assert selector.estimator.random_state == 42
    
    # Check skorch (net) parameters
    assert net.lr == lr
    assert net.optimizer == optimizer
    assert net.optimizer__weight_decay == weight_decay
    # alpha and eps are defaults but we can check they exist
    assert net.optimizer__alpha == 0.9
    assert net.optimizer__eps == 1e-08

def test_aptanet_regressor_hyperparameter_propagation():
    """
    Integration test to ensure that exposed hyperparameters are correctly 
    propagated to the underlying selector and skorch model in AptaNetRegressor.
    """
    # Custom hyperparameters
    n_estimators = 40
    max_depth = 3
    lr = 0.0005
    weight_decay = 0.05
    optimizer = optim.SGD
    
    reg = AptaNetRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        random_state=42
    )
    
    # Create dummy data
    X = np.random.rand(20, 10).astype(np.float32)
    y = np.random.rand(20).astype(np.float32)
    
    # Fit the regressor to build the internal pipeline
    reg.fit(X, y)
    
    # Access the internal pipeline
    pipeline = reg.pipeline_
    selector = pipeline.named_steps["select"]
    net = pipeline.named_steps["net"]
    
    # Check selector parameters
    assert selector.estimator.n_estimators == n_estimators
    assert selector.estimator.max_depth == max_depth
    
    # Check skorch (net) parameters
    assert net.lr == lr
    assert net.optimizer == optimizer
    assert net.optimizer__weight_decay == weight_decay
