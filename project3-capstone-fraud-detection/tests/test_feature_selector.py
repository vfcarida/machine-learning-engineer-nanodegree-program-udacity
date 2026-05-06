import pytest
import pandas as pd
import numpy as np
from src.feature_selector import FeatureSelector

@pytest.fixture
def sample_data():
    """Fixture for dummy data."""
    data = pd.DataFrame({
        'feat_missing': [1, np.nan, np.nan, 4, 5],
        'feat_single': [1, 1, 1, 1, 1],
        'feat_normal': [1, 2, 3, 4, 5],
        'feat_collinear': [1.1, 2.1, 3.1, 4.1, 5.1]
    })
    labels = np.array([0, 1, 0, 1, 0])
    return data, labels

def test_identify_missing(sample_data):
    """Test identifying features with missing values."""
    data, labels = sample_data
    fs = FeatureSelector(data, labels)
    fs.identify_missing(missing_threshold=0.3)
    assert 'feat_missing' in fs.ops['missing']
    assert 'feat_normal' not in fs.ops['missing']

def test_identify_single_unique(sample_data):
    """Test identifying features with a single unique value."""
    data, labels = sample_data
    fs = FeatureSelector(data, labels)
    fs.identify_single_unique()
    assert 'feat_single' in fs.ops['single_unique']
    assert 'feat_normal' not in fs.ops['single_unique']

def test_identify_collinear(sample_data):
    """Test identifying collinear features."""
    data, labels = sample_data
    fs = FeatureSelector(data, labels)
    fs.identify_collinear(correlation_threshold=0.9)
    assert 'feat_collinear' in fs.ops['collinear']
