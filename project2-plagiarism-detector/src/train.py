import argparse
import os
import logging
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def model_fn(model_dir: str) -> Any:
    """
    Load the trained model from the model_dir.
    
    Args:
        model_dir (str): Directory where the model is saved.
        
    Returns:
        Any: The loaded sklearn model.
    """
    logger.info("Loading model.")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    logger.info("Done loading model.")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '.'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '.'))

    args = parser.parse_args()

    # Read in CSV training file
    training_path = os.path.join(args.data_dir, "train.csv")
    logger.info(f"Reading training data from: {training_path}")
    
    try:
        train_data = pd.read_csv(training_path, header=None)
    except Exception as e:
        logger.error(f"Failed to read training data: {e}")
        raise

    # Labels are in the first column, features in the rest
    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]

    logger.info(f"Training LinearSVC model with {len(train_data)} samples.")
    model = LinearSVC()
    model.fit(train_x, train_y)

    # Save the trained model
    model_save_path = os.path.join(args.model_dir, "model.joblib")
    logger.info(f"Saving model to: {model_save_path}")
    joblib.dump(model, model_save_path)
    logger.info("Model saved successfully.")