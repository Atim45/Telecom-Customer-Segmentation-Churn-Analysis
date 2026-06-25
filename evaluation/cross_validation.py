from sklearn.model_selection import cross_val_score

def perform_cross_validation(model, X, y):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="f1"
    )
    print("="*50)
    print("Cross Validation")
    print("="*50)
    print(scores)
    print()
    print("Average F1 Score")
    print(scores.mean())