# Capstone: Credit Card Fraud Detection

![LightGBM](https://img.shields.io/badge/LightGBM-%23008080.svg?style=flat)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)

This capstone project focuses on building a high-performance model to detect fraudulent credit card transactions. It emphasizes rigorous **feature selection** and handling highly **imbalanced datasets**.

## 🧠 Key Component: FeatureSelector

The project includes a custom `FeatureSelector` class (`src/feature_selector.py`) that implements five automated methods for identifying redundant or low-value features:
1. **Missing Values**: Identifies features exceeding a missing percentage threshold.
2. **Single Unique Value**: Finds features that provide no information variance.
3. **Collinearity**: Removes highly correlated features using Pearson correlation.
4. **Zero Importance**: Uses a LightGBM model to find features with zero gain.
5. **Low Cumulative Importance**: Retains only the top features required to reach a specific cumulative importance threshold (e.g., 95%).

## 📂 Project Structure

- `notebooks/fraud_detection.ipynb`: The complete analysis pipeline, from EDA to model evaluation.
- `src/feature_selector.py`: The core utility class for automated feature selection.
- `docs/credit_card_fraud_detection.pdf`: Detailed technical report of the methodology and results.
- `tests/test_feature_selector.py`: Unit tests for the `FeatureSelector` methods.

## 🚀 Usage

```python
from src.feature_selector import FeatureSelector

# Initialize with data and labels
fs = FeatureSelector(data=X, labels=y)

# Identify collinear features
fs.identify_collinear(correlation_threshold=0.9)

# Remove identified features
X_filtered = fs.remove(methods='all')
```

## 🧪 Testing

Run unit tests for the selection logic:
```bash
pytest tests/test_feature_selector.py
```
