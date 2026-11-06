from src.data import load_data
from src.models import get_models
from src.train import run_experiment
from src.evaluation import aggregate_results
from src.utils import save_json
from src.visualization import (
    plot_calibration_curve, 
    plot_roc_curve, 
    plot_pr_curve, 
    plot_feature_importance
)
from src.config import DATA_PATH, TARGET
from src.interpretation import compute_shap

import matplotlib.pyplot as plt
import shap
import os

df = load_data(DATA_PATH)
models = get_models()

results, outputs = run_experiment(df, models,columns=[TARGET + 'time'])

aggregated = aggregate_results(results)

save_json(aggregated, "results/metrics/full_features.json")

# ensure figure dir exists
os.makedirs("results/figures", exist_ok=True)


# generate plots (use Random Forest as main model)
rf_output = outputs["random_forest"]

model = rf_output["model"]
X = rf_output['X']
y_true = rf_output["y_true"]
y_prob = rf_output["y_prob"]
y_pred = rf_output["y_pred"]

# Evaluation plots using cross-validated predictions
plot_roc_curve(
    y_true,
    y_prob,
    "results/figures/roc_rf.png"
)

plot_pr_curve(
    y_true,
    y_prob,
    "results/figures/pr_rf.png"
)

plot_calibration_curve(
    y_true,
    y_prob,
    "results/figures/calibration_rf.png"
)


# Feature importance
importances = rf_output["feature_importance_mean"]
importance_std = rf_output["feature_importance_std"]

plot_feature_importance(
    importances,
    "results/figures/feature_importance_rf.png",
    importance_std=importance_std
)

# SHAP values
# SHAP interpretation
shap_values = compute_shap(model, X)

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.savefig("results/figures/shap_summary.png")
plt.close()