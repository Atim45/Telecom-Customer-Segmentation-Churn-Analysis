from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def smote_random_forest(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train_smote,
        y_train_smote
    )
    prediction = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1 Score": f1_score(y_test, prediction)
    }

    return model, prediction, metrics