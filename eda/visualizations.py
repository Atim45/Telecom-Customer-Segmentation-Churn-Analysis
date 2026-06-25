import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(
        x="churn",
        data=df
    )
    plt.title("Customer Churn Distribution")
    plt.xticks([0,1],["No Churn","Churned"])
    plt.tight_layout()
    plt.show()


def plot_numerical_distribution(df, column):
    plt.figure(figsize=(7,4))
    sns.histplot(
        df[column],
        kde=True
    )
    plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.show()


def boxplot_vs_churn(df, feature):
    plt.figure(figsize=(6,4))
    sns.boxplot(
        x="churn",
        y=feature,
        data=df
    )
    plt.xticks([0,1],["No Churn","Churned"])
    plt.title(f"{feature} vs Churn")
    plt.tight_layout()
    plt.show()