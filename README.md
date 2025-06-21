# Real or Fake Job Classifier

A machine learning project to detect fraudulent job postings using classification algorithms. Built as part of the Python Programming for Data Science course at SPIT.

## 🔍 Overview

Online job platforms are often exploited by scammers posting fake jobs to phish for personal information. This project aims to build a robust classification model that can detect such fraudulent postings using metadata like job location, salary, company profile, etc.

## 📊 Dataset

- **Source:** Hugging Face Datasets  
  https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction
- **Size:** ~17,000 job postings
- **Target Column:** `fraudulent` (0 = Genuine, 1 = Fraudulent)
- **Challenge:** Highly imbalanced (~5% fraudulent)

## ⚙️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn
- Google Colab

## 🛠️ Key Features

- **Data Preprocessing:**
  - Handled missing values, outliers, and inconsistent salary data
  - Dropped irrelevant or duplicate columns

- **Feature Engineering:**
  - Applied **PCA** for dimensionality reduction
  - Used **L1 regularization** and **RFE** for feature selection

- **Class Imbalance Handling:**
  - Used `class_weight='balanced'` and `compute_sample_weight` for fair training

- **Models Implemented:**
  - Logistic Regression (L1 penalty)
  - Random Forest
  - Gradient Boosting

- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC

## 🧪 Results

| Model             | Accuracy | Recall (Fraud) | F1-Score (Fraud) |
|------------------|----------|----------------|------------------|
| Logistic Regression | 80.4%    | 0.62           | 0.23             |
| Random Forest       | 80.1%    | 0.62           | 0.23             |
| Gradient Boosting   | 80.1%    | 0.62           | 0.23             |

- Gradient Boosting gave the **best recall** for fraudulent jobs.
- Logistic Regression provided the **most interpretable model**.

## 📌 Future Work

- Improve precision for fraud class with oversampling or ensemble stacking
- Deploy model via a web app (FastAPI backend + React frontend)
- Collect more labeled data to improve generalization


