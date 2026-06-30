from src.data import load_data
from src.config import DATA_PATH, TARGET
from src.models import get_models
from src.train import run_experiment

models = get_models()
df = load_data(DATA_PATH)

features = list(df.columns)
features.remove(TARGET)

for feature in features:
    subset = [f for f in features if f != feature]

    df_sub = df[subset + [TARGET]]

    results, _ = run_experiment(df_sub, models)