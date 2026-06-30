from src.data import load_data
from src.models import get_models
from src.train import run_experiment
from src.evaluation import aggregate_results
from src.utils import save_json
from src.visualization import plot_calibration_curve, plot_roc_curve, plot_pr_curve, plot_feature_importance
from src.feature_ranking import rank_features
from src.config import DATA_PATH, N_RUNS, TARGET
from src.interpretation import compute_shap
import matplotlib.pyplot as plt

import shap
import os

df = load_data(DATA_PATH)
models = get_models()

results, outputs = run_experiment(df, models, n_runs=N_RUNS)
aggregated = aggregate_results(results)

print(aggregated)
save_json(aggregated, "results/metrics/full_features.json")

# ensure figure dir exists
os.makedirs("results/figures", exist_ok=True)

# generate plots (use Random Forest as main model)
rf_output = outputs["random_forest"]

model = rf_output["model"]
X_test = rf_output["X_test"]
y_test = rf_output["y_test"]
y_prob = rf_output["y_prob"]

plot_roc_curve(y_test, y_prob, "results/figures/roc_rf.png")
plot_pr_curve(y_test, y_prob, "results/figures/pr_rf.png")
plot_calibration_curve(y_test, y_prob, "results/figures/calibration_rf.png")

# feature importance
X = X_test
y = df[TARGET]

importances = rank_features(X, y)

plot_feature_importance(importances, "results/figures/feature_importance.png")

# SHAP values
shap_values = compute_shap(model, X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig("results/figures/shap_summary.png")
plt.close()