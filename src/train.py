
from .evaluation import evaluate
from .config import TARGET, N_REPEATS, N_SPLITS, RANDOM_STATE

import numpy as np
import pandas as pd
from time import time

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold


def run_experiment(
    df,
    models,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS
    ):

    if 'time' in df.columns:
        X = df.drop(columns=[TARGET, "time"], errors="ignore")
    else:
        X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    results = {name: [] for name in models}

    # Accumulate validation probabilities for each observation.
    probability_sum = {
        name: np.zeros(len(X)) for name in models
    }

    prediction_counts = np.zeros(len(X), dtype=int)

    # Collect Random Forest feature importance across folds.
    rf_importances = []

    total_folds = n_splits * n_repeats
    start_time = time()

    print(f"Starting cross-validation: {total_folds} folds")
    
    
    # Cross-validation loop
    for fold, (train_idx, validation_idx) in enumerate(
    cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        for name, original_model in models.items():

            model = clone(original_model)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_validation)
            y_prob = model.predict_proba(X_validation)[:, 1]

            results[name].append(
                evaluate(y_validation, y_pred, y_prob)
            )

            probability_sum[name][validation_idx] += y_prob

            if name == "random_forest":
                rf_importances.append(
                    pd.Series(
                        model.feature_importances_,
                        index=X.columns
                    )
                )

        # Update validation counts once per fold.
        prediction_counts[validation_idx] += 1

        elapsed = time() - start_time

        print(
            f"Completed fold {fold}/{total_folds} "
            f"| Elapsed: {elapsed:.1f}s",
            flush=True
        )
    
    # Check that every observation received validation predictions.
    if np.any(prediction_counts == 0):
        raise ValueError(
            "Some observations received no validation predictions."
        )

    # Average probabilities AFTER all CV folds are complete.
    outputs = {}

    for name, original_model in models.items():

        mean_probabilities = (
            probability_sum[name] / prediction_counts
        )

        mean_predictions = (
            mean_probabilities >= 0.5
        ).astype(int)

        # Fit a final model on the complete dataset for interpretation.
        final_model = clone(original_model)
        final_model.fit(X, y)

        outputs[name] = {
            "model": final_model,
            "X": X,
            "y_true": y.to_numpy(),
            "y_prob": mean_probabilities,
            "y_pred": mean_predictions
        }

    # Aggregate Random Forest feature importance across CV folds.
    if rf_importances:
        importance_df = pd.DataFrame(rf_importances)

        outputs["random_forest"]["feature_importance_mean"] = (
            importance_df.mean()
            .sort_values(ascending=False)
        )

        outputs["random_forest"]["feature_importance_std"] = (
            importance_df.std(ddof=1)
            .reindex(
                outputs["random_forest"]["feature_importance_mean"].index
            )
        )

    return results, outputs
