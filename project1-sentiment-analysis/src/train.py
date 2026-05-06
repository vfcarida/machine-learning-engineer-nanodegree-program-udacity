import argparse
import json
import os
import pickle
import sys
import logging
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from typing import Tuple, Dict, Any

from model import LSTMClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def model_fn(model_dir: str) -> LSTMClassifier:
    """
    Load the PyTorch model from the `model_dir` directory.
    
    Args:
        model_dir (str): Directory where the model files are stored.
        
    Returns:
        LSTMClassifier: The loaded model in evaluation mode.
    """
    logger.info("Loading model.")

    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    logger.info(f"model_info: {model_info}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    logger.info("Done loading model.")
    return model

def _get_train_data_loader(batch_size: int, training_dir: str) -> torch.utils.data.DataLoader:
    """
    Load training data from CSV and return a DataLoader.
    
    Args:
        batch_size (int): Size of batches for training.
        training_dir (str): Directory where train.csv is located.
        
    Returns:
        DataLoader: PyTorch DataLoader for training data.
    """
    logger.info("Loading training data.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def train(model: LSTMClassifier, 
          train_loader: torch.utils.data.DataLoader, 
          epochs: int, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          device: torch.device) -> None:
    """
    Standard training loop for the PyTorch model.
    
    Args:
        model (LSTMClassifier): The PyTorch model to train.
        train_loader (DataLoader): DataLoader providing training batches.
        epochs (int): Number of training epochs.
        optimizer (Optimizer): The optimizer for weight updates.
        loss_fn (Module): The loss function.
        device (device): Target device (CPU or GPU).
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch: {epoch}, BCELoss: {avg_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'localhost'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '.'))
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', 0)))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}.")

    torch.manual_seed(args.seed)

    # Load the training data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    word_dict_path = os.path.join(args.data_dir, "word_dict.pkl")
    try:
        with open(word_dict_path, "rb") as f:
            model.word_dict = pickle.load(f)
    except FileNotFoundError:
        logger.warning("word_dict.pkl not found in data-dir. Ensure it is provided if needed for inference.")

    logger.info(f"Model initialized with embedding_dim {args.embedding_dim}, "
                f"hidden_dim {args.hidden_dim}, vocab_size {args.vocab_size}.")

    # Train the model
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

    # Save the word_dict
    if model.word_dict:
        word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
        with open(word_dict_path, 'wb') as f:
            pickle.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    
    logger.info("Training complete. Model saved.")
