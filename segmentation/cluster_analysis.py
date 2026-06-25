def cluster_summary(df):
    print("=" * 60)
    print("Cluster Summary")
    print("=" * 60)
    summary = df.groupby("cluster").mean(numeric_only=True)
    print(summary)
    return summary