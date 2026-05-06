# Plagiarism Detector

![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=flat&logo=amazon-aws&logoColor=white)

A machine learning system built with **Scikit-learn** and **AWS SageMaker** to detect plagiarism in text files.

## 📊 Features & Logic

The system uses custom feature engineering to compare answer texts with source texts:
- **Containment**: A measure of n-gram overlap between texts.
- **Longest Common Subsequence (LCS)**: Finds the longest sequence of words common to both texts, normalized by length.

## 📂 Project Structure

- `notebooks/01_data_exploration.ipynb`: Initial dataset analysis.
- `notebooks/02_feature_engineering.ipynb`: Implementation of containment and LCS extraction.
- `notebooks/03_training_model.ipynb`: Training and evaluating the LinearSVC model on SageMaker.
- `src/helpers.py`: Data labeling and text preprocessing utilities.
- `src/train.py`: SageMaker training script using `LinearSVC`.
- `tests/test_plagiarism.py`: Validation tests for feature engineering logic.

## 🚀 Usage

1. Follow the notebooks in order (01 -> 02 -> 03).
2. The final notebook will output a trained model stored in S3.
3. Use the SageMaker SDK to deploy the model for inference.

## 🧪 Testing

Run local tests for preprocessing and labeling:
```bash
pytest tests/test_helpers.py
```