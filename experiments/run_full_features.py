from src.data import load_data
from src.models import get_models
from src.train import run_experiment
from src.evaluation import aggregate_results
from src.utils import save_json
from src.visualization import plot_roc_curve, plot_pr_curve, plot_feature_importance
from src.feature_ranking import rank_features
from src.config import DATA_PATH, N_RUNS, TARGET

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
y_test, y_prob, model = outputs["random_forest"]

plot_roc_curve(y_test, y_prob, "results/figures/roc_rf.png")
plot_pr_curve(y_test, y_prob, "results/figures/pr_rf.png")

# feature importance
X = df.drop(columns=[TARGET])
y = df[TARGET]
importances = rank_features(X, y)

plot_feature_importance(importances, "results/figures/feature_importance.png")