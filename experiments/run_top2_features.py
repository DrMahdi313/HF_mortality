from src.data import load_data
from src.models import get_models
from src.train import run_experiment
from src.evaluation import aggregate_results
from src.utils import save_json
from src.config import DATA_PATH, N_RUNS, TOP_FEATURES, TARGET

df = load_data(DATA_PATH)

df_small = df[TOP_FEATURES + [TARGET]]

models = get_models()

results, _ = run_experiment(df_small, models, n_runs=N_RUNS)
aggregated = aggregate_results(results)

print(aggregated)
save_json(aggregated, "results/top2_features.json")
