# Credit Card Fraud Detection

This project applies machine learning techniques to detect fraudulent credit card transactions.
The models are trained and evaluated in Google Colab using the [Credit Card Fraud dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).

---

## Project Overview

* **Goal:** Detect fraudulent credit card transactions
* **Dataset:** Kaggle (≈ 893k transactions, \~9% fraud)
* **Techniques:**

  * Logistic Regression (supervised)
  * Isolation Forest (unsupervised anomaly detection)
* **Skills Practiced:**

  * Handling imbalanced datasets
  * Model evaluation (ROC-AUC, confusion matrix, precision/recall)
  * Working with real-world transaction data

---

## Getting Started

### 1. Run in Google Colab

You can open and run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118vPUxDSKkB8ygt09S_PkVJnHgB8W9jJ#scrollTo=WIgS9LBg39C1)

Steps:

1. Open the Colab notebook.
2. Upload the dataset (`card_transdata.csv`) when prompted.
3. Run all cells to train and evaluate the models.

---

### 2. Dataset

The dataset is not included in this repository because it is too large for GitHub.

* Download the full dataset here: [Kaggle Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).
* Save it as `card_transdata.csv` in the project folder before running.

For demonstration or lightweight testing, you can also create a smaller sample dataset:

```python
import pandas as pd

df = pd.read_csv("card_transdata.csv")
df_sample = df.sample(n=10000, random_state=42)
df_sample.to_csv("card_transdata_sample.csv", index=False)
```

---

## Model Results

* **Logistic Regression**

  * Handles imbalanced data using class weighting.
  * Outputs fraud probability for each transaction.

* **Isolation Forest**

  * Detects anomalies without requiring fraud labels.
  * Simpler but typically less accurate than Logistic Regression.

**Metrics used**:

* ROC-AUC
* Precision, Recall, F1-score
* Confusion Matrix

---

## Project Structure

```
├── creditcardfrauddetection.py   # Training & evaluation script
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── (dataset not included – download from Kaggle)
```

---
