import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
def feature_selected_rf(X_train, X_test, y_train, y_test, top_n=7):

    base = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    base.fit(X_train, y_train)
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": base.feature_importances_
    })
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )
    top_features = importance.head(top_n)["Feature"].tolist()
    X_train_fs = X_train[top_features]
    X_test_fs = X_test[top_features]
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(X_train_fs, y_train)
    prediction = model.predict(X_test_fs)
    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1 Score": f1_score(y_test, prediction)
    }
    return (
        model,
        prediction,
        metrics,
        importance,
        top_features
    )