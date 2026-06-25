import matplotlib.pyplot as plt
import pandas as pd

def feature_importance(model, columns):

    importance = pd.DataFrame({
        "Feature":columns,
        "Importance":model.feature_importances_
    })
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )
    plt.figure(figsize=(8,5))
    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return importance