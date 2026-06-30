from src.reporting import generate_metrics_table
import json

with open("results/metrics/full_features.json") as f:
    full = json.load(f)

with open("results/top2_features.json") as f:
    top2 = json.load(f)

combined = {
    "full_features": full,
    "top2_features": top2
}

generate_metrics_table(combined, "results/summary.csv")