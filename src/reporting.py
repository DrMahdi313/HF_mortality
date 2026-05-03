import pandas as pd


def generate_metrics_table(metrics_dict, save_path):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(save_path)
    return df