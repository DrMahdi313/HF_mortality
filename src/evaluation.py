import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def evaluate(y_true, y_pred, y_prob):
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

def aggregate_results(results):
    aggregated = {}
    for model_name, runs in results.items():
        aggregated[model_name] = {
            metric: np.mean([run[metric] for run in runs])
            for metric in runs[0]
        }
    return aggregated
