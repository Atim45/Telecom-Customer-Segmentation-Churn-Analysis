# Telecom Customer Segmentation and Churn Prediction

A complete end-to-end Machine Learning project that analyzes customer behaviour, segments telecom customers using K-Means clustering, and predicts customer churn using multiple machine learning models.

---

# Table of Contents

* Overview
* Features
* Project Structure
* Tech Stack
* Installation
* Usage
* Machine Learning Pipeline
* Results
* Key Learnings

---

# Overview

Customer churn is one of the biggest challenges for telecom companies. Losing existing customers directly impacts revenue, making early churn prediction an important business problem.

This project follows a complete machine learning workflow—from data exploration and preprocessing to customer segmentation and churn prediction. Multiple machine learning approaches are implemented and compared to identify the best-performing model.

---

# Features

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Feature Engineering
* Customer Segmentation using K-Means Clustering
* Random Forest Baseline Model
* Random Forest with SMOTE
* Hyperparameter Tuned Random Forest
* Feature Selected Random Forest
* XGBoost Classifier
* Model Comparison using Accuracy, Precision, Recall and F1 Score
* Business Insights and Recommendations

---
## 📂 Project Structure

```text
Telecom-Customer-Segmentation-and-Churn-Prediction
│
├── data
│   ├── raw
│   │   └── telecom_churn.csv
│   ├── processed
│   └── sample
│
├── docs
│   └── plots
│       ├── plot_churn_overview.png
│       ├── plot_confusion_matrix.png
│       ├── plot_correlation.png
│       ├── plot_demographics.png
│       ├── plot_elbow.png
│       ├── plot_pca_clusters.png
│       ├── plot_segment_churn.png
│       ├── plot_state_churn.png
│       ├── plot_tenure_age.png
│       └── plot_usage_behavior.png
│
├── notebooks
│   └── telecom_customer_analysis.ipynb
│
├── outputs
│   ├── metrics
│   ├── predictions
│   └── reports
│
├── src
│   ├── analytics
│   ├── eda
│   ├── evaluation
│   ├── models
│   ├── preprocessing
│   └── segmentation
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
# Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost

---

# Installation

Clone the repository

```bash
git clone https://github.com/Atim45/Telecom-Customer-Segmentation-and-Churn-Prediction.git
```

Move into the project directory

```bash
cd Telecom-Customer-Segmentation-and-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

# Machine Learning Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Customer Segmentation (K-Means)
    │
    ▼
Train-Test Split
    │
    ▼
Machine Learning Models

• Baseline Random Forest
• Random Forest + SMOTE
• Hyperparameter Tuned Random Forest
• Feature Selected Random Forest
• XGBoost

    │
    ▼
Model Evaluation & Comparison
    │
    ▼
Business Insights
```

---

# Results

The following models were implemented and compared:

| Model                          | Accuracy | Precision | Recall | F1 Score |
| ------------------------------ | -------- | --------- | ------ | -------- |
| Baseline Random Forest         | 79.95%   | 000       | 00     | 00       |
| Random Forest + SMOTE          | 66.44%   | 0.201     | 0.226  | 0.213    |
| Tuned Random Forest            | 59.35%   | 0.202     | 0.347  | 0.255    |
| Feature Selected Random Forest | 79.94%   | 000       | 0.0    | 0.00     |
| XGBoost                        | 79.95%   | 0.00      | 0.00   | 0.00     |

The Tuned Random Forest achieved the best performance for churn prediction based on F1 Score (0.255), making it the most effective model for identifying churned customers in this imbalanced dataset.
---

# Key Learnings

Through this project, I gained practical experience with:

* Performing Exploratory Data Analysis on a real-world style telecom dataset.
* Building an end-to-end machine learning pipeline.
* Engineering meaningful features for customer behaviour analysis.
* Customer segmentation using K-Means clustering.
* Handling imbalanced datasets using SMOTE.
* Hyperparameter tuning of Random Forest models.
* Feature selection using feature importance.
* Comparing multiple machine learning models using classification metrics.
* Organizing a machine learning project into a modular and maintainable codebase.
