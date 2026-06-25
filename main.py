from sklearn.model_selection import train_test_split

# -----------------------
# EDA
# -----------------------

from eda.data_loader import load_dataset
from eda.statistics import dataset_summary
from eda.visualizations import (
    plot_churn_distribution,
    plot_numerical_distribution,
    boxplot_vs_churn
)
from eda.correlation import correlation_heatmap

# -----------------------
# Preprocessing
# -----------------------

from preprocessing.cleaning import clean_data
from preprocessing.feature_engineering import feature_engineering
from preprocessing.encoding import encode_features

# -----------------------
# Segmentation
# -----------------------

from segmentation.kmeans import perform_kmeans
from segmentation.cluster_analysis import cluster_summary

# -----------------------
# Machine Learning Models
# -----------------------

from models.baseline_rf import baseline_random_forest
from models.smote_rf import smote_random_forest
from models.tuned_rf import tuned_random_forest
from models.feature_selection import feature_selected_rf
from models.xgboost_model import xgboost_model
from models.comparison import compare_models

# -----------------------
# Evaluation
# -----------------------

from evaluation.metrics import evaluate_model

# -----------------------
# Business Insights
# -----------------------

from analytics.business_insights import business_insights
from analytics.recommendations import recommendations


def main():

    print("=" * 70)
    print("Telecom Customer Segmentation & Churn Prediction")
    print("=" * 70)

    # -----------------------
    # Load Dataset
    # -----------------------

    df = load_dataset("data/raw/telecom_churn.csv")

    # -----------------------
    # Data Preprocessing
    # -----------------------

    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)

    # -----------------------
    # Exploratory Data Analysis
    # -----------------------

    dataset_summary(df)

    plot_churn_distribution(df)
    plot_numerical_distribution(df, "age")
    boxplot_vs_churn(df, "tenure_months")
    correlation_heatmap(df)

    # -----------------------
    # Customer Segmentation
    # -----------------------

    cluster_features = [
        "age",
        "tenure_months",
        "usage_score",
        "estimated_salary"
    ]

    df, kmeans_model = perform_kmeans(
        df,
        cluster_features
    )

    cluster_summary(df)

    # -----------------------
    # Prepare Data for ML
    # -----------------------

    X = df.drop(
        columns=[
            "churn",
            "date_of_registration",
            "age_group"
        ]
    )

    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------
    # Train Models
    # -----------------------

    results = []

    baseline_model, pred, metrics = baseline_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )
    metrics["Model"] = "Baseline RF"
    results.append(metrics)

    smote_model, pred, metrics = smote_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )
    metrics["Model"] = "RF + SMOTE"
    results.append(metrics)

    tuned_model, pred, metrics = tuned_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )
    metrics["Model"] = "Tuned RF"
    results.append(metrics)

    feature_model, pred, metrics, importance, top_features = feature_selected_rf(
        X_train,
        X_test,
        y_train,
        y_test
    )
    metrics["Model"] = "Feature Selected RF"
    results.append(metrics)

    xgb_model, pred, metrics = xgboost_model(
        X_train,
        X_test,
        y_train,
        y_test
    )
    metrics["Model"] = "XGBoost"
    results.append(metrics)

    # -----------------------
    # Model Comparison
    # -----------------------

    comparison = compare_models(results)

    print("\n")
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)

    print(comparison)

    # -----------------------
    # Final Evaluation
    # -----------------------

    print("\nBest Model Evaluation\n")

    # For now evaluate XGBoost (last trained model)
    evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    # -----------------------
    # Business Insights
    # -----------------------

    business_insights(df)
    recommendations()

    print("\n")
    print("=" * 70)
    print("Project Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()