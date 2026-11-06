
from src.data import load_data
from src.config import DATA_PATH, TARGET
from src.models import get_models
from src.evaluation import aggregate_results
from src.utils import save_json
from src.train import run_experiment

import pandas as pd


# Load dataset
df = load_data(DATA_PATH)
models = get_models()

# Admission-time predictors only
features = [
    col for col in df.columns
    if col not in [TARGET, "time"]
]

if not features:
    raise ValueError("No predictor features found.")

# Reduced CV configuration for ablation
ABLATION_SPLITS = 5
ABLATION_REPEATS = 3

# Metrics to compare
METRICS = ["mcc", "roc_auc", "pr_auc"]

results_summary = {}
comparison_rows = []


# --------------------------------------------------
# 1. Baseline: all features
# --------------------------------------------------

print("Running baseline experiment...", flush=True)

baseline_results, _ = run_experiment(
    df,
    models,
    n_splits=ABLATION_SPLITS,
    n_repeats=ABLATION_REPEATS
)

results_summary["baseline"] = aggregate_results(baseline_results)


# --------------------------------------------------
# 2. Ablation: remove one feature at a time
# --------------------------------------------------

for i, feature in enumerate(features, start=1):

    print(
        f"Ablation {i}/{len(features)}: removing {feature}",
        flush=True
    )

    subset = [f for f in features if f != feature]
    df_sub = df[subset + [TARGET, "time"]]

    ablation_results, _ = run_experiment(
        df_sub,
        models,
        n_splits=ABLATION_SPLITS,
        n_repeats=ABLATION_REPEATS
    )

    results_summary[f"without_{feature}"] = (
        aggregate_results(ablation_results)
    )

    # Paired fold-level comparison
    for model_name in models:

        for metric in METRICS:

            baseline_values = [
                fold_result[metric]
                for fold_result in baseline_results[model_name]
            ]

            ablation_values = [
                fold_result[metric]
                for fold_result in ablation_results[model_name]
            ]

            differences = [
                ablation - baseline
                for ablation, baseline
                in zip(ablation_values, baseline_values)
            ]

            comparison_rows.append({
                "removed_feature": feature,
                "model": model_name,
                "metric": metric,
                "baseline_mean": float(
                    pd.Series(baseline_values).mean()
                ),
                "without_feature_mean": float(
                    pd.Series(ablation_values).mean()
                ),
                "mean_difference": float(
                    pd.Series(differences).mean()
                ),
                "std_difference": float(
                    pd.Series(differences).std(ddof=1)
                )
            })


# --------------------------------------------------
# 3. Save results
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison_rows)

save_json(
    {
        "aggregated_results": results_summary,
        "paired_comparisons": comparison_rows
    },
    "results/metrics/ablation.json"
)

comparison_df.to_csv(
    "results/metrics/ablation_comparison.csv",
    index=False
)

print("\nAblation experiments completed.", flush=True)

print("\nFeature-removal comparison:")
print(
    comparison_df[
        comparison_df["metric"] == "mcc"]
        .sort_values(
            ["model", "mean_difference"]
            ).to_string(index=False)
    )
