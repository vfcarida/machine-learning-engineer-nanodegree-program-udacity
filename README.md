# Machine Learning Engineer Nanodegree — Udacity Portfolio

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Frameworks](https://img.shields.io/badge/frameworks-PyTorch%20%7C%20Scikit--Learn%20%7C%20LightGBM-orange.svg)

Professional overhaul of projects completed during the **Udacity Machine Learning Engineer Nanodegree**. This repository showcases end-to-end ML pipelines, from data exploration and feature engineering to model deployment on AWS SageMaker.

---

## 🚀 Projects Overview

| Project | Description | Key Technologies |
| :--- | :--- | :--- |
| **[Sentiment Analysis](./project1-sentiment-analysis)** | Deployment of an LSTM model using PyTorch to classify movie reviews. Includes a web interface. | PyTorch, AWS SageMaker, Lambda, API Gateway |
| **[Plagiarism Detector](./project2-plagiarism-detector)** | Binary classification system to detect plagiarism using containment and LCS features. | Scikit-learn, AWS SageMaker, Feature Engineering |
| **[Capstone: Fraud Detection](./project3-capstone-fraud-detection)** | High-performance feature selection and fraud detection on credit card transaction data. | LightGBM, Pandas, Feature Selection, Imbalanced Data |

---

## 🛠️ Architecture & Workflow

The repository follows a standardized structure for production-ready ML code:

- **`notebooks/`**: Exploratory Data Analysis (EDA) and iterative development.
- **`src/`**: Modularized Python scripts for training, inference, and utilities.
- **`tests/`**: Unit tests to ensure robustness and reproducibility.
- **`web/`**: (Project 1) Frontend components for real-world application testing.

---

## 📋 Prerequisites

To replicate these environments locally, ensure you have:
- Python 3.8 or higher
- [Git LFS](https://git-lfs.github.com/) installed (for large datasets)
- An AWS account (for SageMaker deployment examples)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vfcarida/machine-learning-engineer-nanodegree-program-udacity.git
   cd machine-learning-engineer-nanodegree-program-udacity
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install NLTK data** (required for Project 1 & 2):
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

---

## 📂 Project Structure

```text
.
├── .gitattributes             # Git LFS configuration
├── .gitignore                # Standard Python/Jupyter ignores
├── LICENSE                   # MIT License
├── requirements.txt          # Consolidated dependencies
├── CONTRIBUTING.md           # Contribution guidelines
│
├── project1-sentiment-analysis/
│   ├── notebooks/            # EDA & Workflow notebooks
│   ├── src/                  # LSTM model & SageMaker scripts
│   ├── web/                  # HTML interface
│   └── tests/                # Pytest suite
│
├── project2-plagiarism-detector/
│   ├── notebooks/            # Feature engineering logic
│   ├── src/                  # Sklearn train & helpers
│   └── tests/                # Feature extraction tests
│
└── project3-capstone-fraud-detection/
    ├── notebooks/            # Fraud analysis
    ├── src/                  # Custom FeatureSelector class
    └── tests/                # Class behavior tests
```

---

## 🤝 Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to improve the code or documentation.

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
