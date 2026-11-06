import json
import pandas as pd

with open("results/metrics/full_features.json") as f:
    full = json.load(f)

with open("results/top2_features.json") as f:
    top2 = json.load(f)

rows = []

for dataset_name, dataset_results in [
    ("full_features", full),
    ("top2_features", top2)
]:
    for model_name, metrics in dataset_results.items():
        row = {
            "dataset": dataset_name,
            "model": model_name
        }

        for metric_name, values in metrics.items():
            row[f"{metric_name}_mean"] = values["mean"]
            row[f"{metric_name}_std"] = values["std"]

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)

print(df.to_string(index=False))