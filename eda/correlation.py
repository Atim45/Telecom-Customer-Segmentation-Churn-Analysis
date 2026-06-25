import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df):

    plt.figure(figsize=(10,8))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()