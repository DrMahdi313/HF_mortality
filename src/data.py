import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TARGET, TEST_SIZE

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def split_data(df, random_state=None):
    X = df.drop(columns=[TARGET, 'time']) # The 'time' feature is a data from feature. 
                                          # As stated in the main article, it's the follow up period measured in days. 
                                          # So, we need to drop that so we as to prevent target leakage.
    y = df[TARGET]
    

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state
    )