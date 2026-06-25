from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    df = df.copy()
    encoder = LabelEncoder()
    categorical = df.select_dtypes(include="object").columns
    for column in categorical:
        df[column] = encoder.fit_transform(df[column])
    return df