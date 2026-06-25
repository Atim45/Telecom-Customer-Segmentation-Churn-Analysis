from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    print("=" * 70)
    print(classification_report(y_test, prediction))
    print("=" * 70)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, prediction))
    try:
        probability = model.predict_proba(X_test)[:, 1]
        print("\nROC-AUC Score")
        print(roc_auc_score(y_test, probability))
    except Exception:
        pass