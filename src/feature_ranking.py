import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def rank_features(X, y):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    return importance
