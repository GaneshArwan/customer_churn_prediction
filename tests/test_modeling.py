import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train_model import train_model

def test_train_model():
    # Create synthetic classification data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = train_model(X_train, y_train)
    
    # Check if it has a predict method (is a valid sklearn estimator)
    assert hasattr(model, 'predict')
    
    # Check predictions format
    preds = model.predict(X_train[:5])
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
