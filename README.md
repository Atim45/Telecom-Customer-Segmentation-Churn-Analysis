# Telecom Customer Segmentation & Churn Prediction

An end-to-end Data Analytics and Machine Learning project that analyzes telecom customer behavior, performs customer segmentation using K-Means clustering, and predicts customer churn using multiple machine learning models.

The project follows a modular Python architecture with separate components for data preprocessing, exploratory data analysis (EDA), customer segmentation, model training, evaluation, and business insights.

---

# Dataset

- **Dataset Name:** `telecom_churn.csv`
- **Domain:** Telecommunications
- **Total Records:** **243,553 customers**
- **Total Features:** **17** (including engineered features)
- **Target Variable:** **churn**
  - `0` → No Churn
  - `1` → Churn
- **Problem Type:** Binary Classification

The dataset contains customer demographic information, telecom usage behavior, registration details, salary, communication activity, and churn status. During preprocessing, additional features such as **tenure_months**, **usage_score**, and **age_group** were engineered to improve customer analysis and model performance.

---

# Features

- Data Cleaning & Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Customer Segmentation using K-Means Clustering
- Churn Prediction using Multiple ML Models
- Feature Importance Analysis
- Model Evaluation & Comparison
- Business Insights & Recommendations
- Modular Python Project Structure

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

---

# Project Structure

```text
Telecom-Customer-Segmentation-and-Churn-Prediction
│
├── data
│   └── raw
│       └── telecom_churn.csv
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

---

#  How to Run

### Clone the repository

```bash
git clone https://github.com/Atim45/Telecom-Customer-Segmentation-Churn-Analysis.git
```

### Navigate into the project

```bash
cd Telecom-Customer-Segmentation-Churn-Analysis
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

# Machine Learning Models

The following machine learning models were implemented and compared:

- Baseline Random Forest
- Random Forest + SMOTE
- Tuned Random Forest
- Feature Selected Random Forest
- XGBoost Classifier

---

# Results

| Model | Accuracy | Precision | Recall | F1 Score |
|:-----------------------------|:--------:|:---------:|:------:|:--------:|
| Baseline Random Forest | **79.95%** | 0.000 | 0.000 | 0.000 |
| Random Forest + SMOTE | **66.44%** | 0.201 | 0.226 | 0.213 |
| Tuned Random Forest ⭐ | **59.35%** | **0.202** | **0.347** | **0.255** |
| Feature Selected Random Forest | **79.94%** | 0.000 | 0.000 | 0.000 |
| XGBoost | **79.95%** | 0.000 | 0.000 | 0.000 |

**Best Performing Model:** **Tuned Random Forest** (based on F1 Score)

---

#  Key Findings

- The dataset is highly imbalanced (~80% Non-Churn vs ~20% Churn), making **Accuracy alone an unreliable evaluation metric.**
- Hyperparameter tuning significantly improved churn detection compared to the baseline model.
- Applying **SMOTE** improved the model's ability to identify churned customers by addressing class imbalance.
- **K-Means Clustering (4 customer segments)** revealed distinct customer groups that can support targeted retention strategies.
- Engineered features such as **tenure_months**, **usage_score**, and **age_group** improved customer profiling and business understanding.
- Customers with **lower tenure** and **lower engagement** represent higher retention priority groups.

---

#  Business Recommendations

- Prioritize retention efforts for newly registered customers.
- Monitor customers with low engagement using usage-based metrics.
- Design personalized offers based on customer segments.
- Implement proactive retention campaigns for customers with higher churn probability.

---

# Learning Outcomes

Through this project, I gained hands-on experience in:

- Data Cleaning & Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Customer Segmentation using K-Means
- Handling Imbalanced Data using SMOTE
- Hyperparameter Tuning
- Feature Selection
- Machine Learning Model Comparison
- Model Evaluation using Precision, Recall, F1 Score & ROC-AUC
- Writing modular and maintainable Python code

---

#  Author

**Atiksh Sharma**
