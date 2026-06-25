import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc
)
def plot_roc(model, X_test, y_test, title):
    probability = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )
    plt.figure(figsize=(6,5))
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc(fpr,tpr):.3f}"
    )
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()