import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importances, save_path):
    importances.sort_values().plot(kind="barh")
    plt.title("Feature Importance")
    plt.savefig(save_path)
    plt.close()