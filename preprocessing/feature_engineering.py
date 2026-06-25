import pandas as pd


def feature_engineering(df):

    df = df.copy()

    # Remove negative values
    df["calls_made"] = df["calls_made"].clip(lower=0)
    df["sms_sent"] = df["sms_sent"].clip(lower=0)
    df["data_used"] = df["data_used"].clip(lower=0)

    # Parse registration date and compute tenure
    df["date_of_registration"] = pd.to_datetime(df["date_of_registration"])

    REFERENCE_DATE = pd.Timestamp("2024-01-01")

    df["tenure_months"] = (
        (REFERENCE_DATE - df["date_of_registration"]).dt.days / 30
    ).astype(int)

    # Usage score
    df["usage_score"] = (
        (df["calls_made"] / df["calls_made"].max())
        + (df["sms_sent"] / df["sms_sent"].max())
        + (df["data_used"] / df["data_used"].max())
    ).round(4)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[18, 30, 45, 60, 75],
        labels=["18-30", "31-45", "46-60", "61-75"]
    )
    return df