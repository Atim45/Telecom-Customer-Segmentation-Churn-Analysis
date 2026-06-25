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

# Project Structure

```text
Telecom-Customer-Segmentation-and-Churn-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── eda/
│
├── preprocessing/
│
├── segmentation/
│
├── models/
│
├── evaluation/
│
├── analytics/
│
├── outputs/
│
├── docs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

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
| Baseline Random Forest         | XX       | XX        | XX     | XX       |
| Random Forest + SMOTE          | XX       | XX        | XX     | XX       |
| Tuned Random Forest            | XX       | XX        | XX     | XX       |
| Feature Selected Random Forest | XX       | XX        | XX     | XX       |
| XGBoost                        | XX       | XX        | XX     | XX       |

The best-performing model was selected based on the evaluation metrics.

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
