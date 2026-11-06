import pandas as pd


def generate_metrics_table(metrics_dict, save_path):
    rows = {}

    for model_name, metrics in metrics_dict.items():
        rows[model_name] = {}

        for metric_name, values in metrics.items():
            rows[model_name][f"{metric_name}_mean"] = values["mean"]
            rows[model_name][f"{metric_name}_std"] = values["std"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"

    df.to_csv(save_path)

    return df