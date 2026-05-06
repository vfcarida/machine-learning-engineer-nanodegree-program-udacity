# Sentiment Analysis Web App (Data Science, Machine Learning)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=flat&logo=amazon-aws&logoColor=white)

This project implements a sentiment analysis application using **PyTorch** and **AWS SageMaker**. It uses an LSTM-based Recurrent Neural Network (RNN) to determine whether a movie review is positive or negative.

## 🏗️ Architecture

The application is deployed as a serverless web app:
1. **Frontend**: A simple HTML/Javascript interface.
2. **API Gateway**: Handles HTTP requests from the frontend.
3. **AWS Lambda**: Processes the request and calls the SageMaker endpoint.
4. **SageMaker Endpoint**: Hosts the trained PyTorch model for real-time inference.

## 📂 Project Structure

- `notebooks/sentiment_analysis.ipynb`: The main workflow for data collection, preprocessing, training, and deployment.
- `src/model.py`: Definition of the `LSTMClassifier`.
- `src/train.py`: Training script for SageMaker training jobs.
- `src/predict.py`: Inference script for the SageMaker endpoint.
- `src/utils.py`: Text cleaning and tokenization utilities.
- `web/index.html`: Web interface for testing predictions.

## 🚀 How to Run

1. **Training**: Run the cells in the notebook to train the model on the IMDB dataset via SageMaker.
2. **Deployment**: Deploy the model to an endpoint.
3. **Web App**: 
   - Update the `action` URL in `web/index.html` with your deployed API Gateway endpoint.
   - Open `index.html` in your browser.

## 🧪 Testing

To run local unit tests for the inference logic:
```bash
pytest tests/test_predict.py
```
