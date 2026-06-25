def business_insights(df):

    print("="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    churn = df.groupby("churn").mean(numeric_only=True)
    print(churn)
    print()
    print("Highest Risk Segment")
    print(df.groupby("cluster")["churn"].mean())