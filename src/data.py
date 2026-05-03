import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TARGET, TEST_SIZE

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def split_data(df, random_state=None):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state
    )