import argparse
import json
import os
import pickle
import sys
import logging
import numpy as np
import torch
from typing import Any, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from model import LSTMClassifier
from utils import review_to_words, convert_and_pad

def model_fn(model_dir: str) -> LSTMClassifier:
    """
    Load the PyTorch model from the `model_dir` directory.
    
    Args:
        model_dir (str): Directory where the model files are stored.
        
    Returns:
        LSTMClassifier: The loaded model in evaluation mode.
    """
    logger.info("Loading model.")

    # Load the parameters used to create the model
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    try:
        with open(model_info_path, 'rb') as f:
            model_info = torch.load(f)
        logger.info(f"Model info: {model_info}")
    except FileNotFoundError:
        logger.error(f"model_info.pth not found in {model_dir}")
        raise

    # Determine the device and construct the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    try:
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))
    except FileNotFoundError:
        logger.error(f"model.pth not found in {model_dir}")
        raise

    # Load the saved word_dict
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    try:
        with open(word_dict_path, 'rb') as f:
            model.word_dict = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"word_dict.pkl not found in {model_dir}")
        raise

    model.to(device).eval()

    logger.info("Done loading model.")
    return model

def input_fn(serialized_input_data: bytes, content_type: str) -> str:
    """
    Deserialize the input data.
    
    Args:
        serialized_input_data (bytes): The raw request body.
        content_type (str): The content type of the request.
        
    Returns:
        str: The decoded input string.
    """
    logger.info(f"Deserializing input data of type: {content_type}")
    if content_type == 'text/plain':
        return serialized_input_data.decode('utf-8')
    raise ValueError(f"Unsupported content type: {content_type}. Expected 'text/plain'.")

def output_fn(prediction_output: int, accept: str) -> str:
    """
    Serialize the prediction output.
    
    Args:
        prediction_output (int): The prediction result (0 or 1).
        accept (str): The requested content type for the response.
        
    Returns:
        str: String representation of the result.
    """
    logger.info(f"Serializing generated output for accept type: {accept}")
    return str(prediction_output)

def predict_fn(input_data: str, model: LSTMClassifier) -> int:
    """
    Perform inference on the input data using the provided model.
    
    Args:
        input_data (str): The raw review text.
        model (LSTMClassifier): The trained sentiment analysis model.
        
    Returns:
        int: Prediction result (1 for positive, 0 for negative).
    """
    logger.info("Inferring sentiment of input data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise RuntimeError("Model word_dict is missing. Prediction cannot proceed.")
    
    # Preprocess the input review
    words = review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict, words)

    # Construct input tensor. Model expects format: [length, review_sequence]
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack).to(device)

    # Ensure model is in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(data)
    
    # Round output to get binary classification
    result = int(np.round(output.cpu().numpy()))

    return result

if __name__ == '__main__':
    # This script is primarily intended for use with SageMaker inference
    # but can be executed directly for basic testing if needed.
    pass
