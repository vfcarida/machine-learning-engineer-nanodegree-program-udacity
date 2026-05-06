import pytest
import torch
import numpy as np
from src.model import LSTMClassifier
from src.predict import input_fn, output_fn

def test_input_fn():
    """Test input deserialization."""
    test_data = b"This was an excellent movie!"
    content_type = "text/plain"
    result = input_fn(test_data, content_type)
    assert result == "This was an excellent movie!"

def test_input_fn_error():
    """Test input deserialization with unsupported content type."""
    with pytest.raises(ValueError):
        input_fn(b"{}", "application/json")

def test_output_fn():
    """Test output serialization."""
    assert output_fn(1, "text/plain") == "1"
    assert output_fn(0, "text/plain") == "0"

def test_model_init():
    """Test model initialization."""
    model = LSTMClassifier(embedding_dim=32, hidden_dim=100, vocab_size=5000)
    assert model.embedding.num_embeddings == 5000
    assert model.embedding.embedding_dim == 32
    assert model.lstm.hidden_size == 100
