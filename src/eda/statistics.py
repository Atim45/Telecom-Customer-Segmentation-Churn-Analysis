def dataset_summary(df):

    print("=" * 50)
    print("Dataset Shape")
    print("=" * 50)

    print(df.shape)

    print("\n")

    print("=" * 50)
    print("Data Types")
    print("=" * 50)

    print(df.dtypes)

    print("\n")

    print("=" * 50)
    print("Missing Values")
    print("=" * 50)

    print(df.isnull().sum())

    print("\n")

    print("=" * 50)
    print("Statistical Summary")
    print("=" * 50)

    print(df.describe().round(2))