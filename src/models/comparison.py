import pandas as pd

def compare_models(results):
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        by="F1 Score",
        ascending=False
    )
    return comparison