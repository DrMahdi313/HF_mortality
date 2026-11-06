
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label="Random Forest")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(importances, save_path, importance_std=None):
    sorted_importances = importances.sort_values(ascending=True)

    if importance_std is not None:
        sorted_std = importance_std.reindex(sorted_importances.index)
        xerr = sorted_std
    else:
        xerr = None

    plt.figure(figsize=(8, 6))
    sorted_importances.plot(
        kind="barh",
        xerr=xerr,
        capsize=3
    )

    plt.xlabel("Mean Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curve(y_true, y_prob, save_path):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10
    )

    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
