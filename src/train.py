from .data import split_data
from .evaluation import evaluate

def run_experiment(df, models, n_runs=100):
    results = {name: [] for name in models.keys()}
    last_run_outputs = {}

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = split_data(df, random_state=i)

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_prob)
            results[name].append(metrics)

            # store LAST run for plotting
            last_run_outputs[name] = {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "y_prob": y_prob
                }

    return results, last_run_outputs

