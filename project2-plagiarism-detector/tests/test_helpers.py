import pytest
import pandas as pd
import numpy as np
import io
from src.helpers import process_file, train_test_dataframe

def test_process_file():
    """Test text preprocessing logic."""
    test_text = "This is a TEST! It has numbers 123 and symbols."
    file_mock = io.StringIO(test_text)
    result = process_file(file_mock)
    # Expected: lowercase, no symbols, collapsed spaces
    assert result == "this is a test it has numbers 123 and symbols"

def test_train_test_dataframe_logic():
    """Test the structure of the dataframe returned by train_test_dataframe."""
    # Create a dummy dataframe
    data = {
        'File': ['f1', 'f2', 'f3'],
        'Task': ['a', 'b', 'c'],
        'Category': [1, 0, 3] # 1,3 are plagiarism, 0 is original
    }
    df = pd.DataFrame(data)
    
    # Run the splitting logic
    # Note: create_datatype uses sampling, so with only 3 records it might be tricky, 
    # but let's check if 'Datatype' column is created.
    result_df = train_test_dataframe(df, random_seed=42)
    
    assert 'Datatype' in result_df.columns
    assert all(val in ['orig', 'train', 'test'] for val in result_df['Datatype'])
