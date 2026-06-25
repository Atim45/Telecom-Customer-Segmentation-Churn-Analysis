from sklearn.cluster import KMeans

def perform_kmeans(df, features, n_clusters=4):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    df = df.copy()
    df["cluster"] = model.fit_predict(df[features])
    return df, model