
from src.data import load_data, split_data
from src.config import TARGET
from src.models import get_models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/heart_failure_clinical_records_dataset.csv"


def test_load_data():
    df = load_data(DATA_PATH)

    assert not df.empty
    assert TARGET in df.columns
    assert "time" in df.columns


def test_split_data_excludes_target_and_time():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df, random_state=42
    )

    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns
    assert "time" not in X_train.columns
    assert "time" not in X_test.columns


def test_split_data_preserves_row_counts():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df, random_state=42
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(df)


def test_split_data_is_reproducible():
    df = load_data(DATA_PATH)

    first = split_data(df, random_state=42)
    second = split_data(df, random_state=42)

    for first_part, second_part in zip(first, second):
        assert first_part.equals(second_part)



def test_svm_and_mlp_use_scaling_pipelines():
    models = get_models()

    for name in ["svm_rbf", "mlp"]:
        assert isinstance(models[name], Pipeline)
        assert isinstance(
            models[name].named_steps["scaler"],
            StandardScaler
        )


def test_all_models_are_created():
    models = get_models()

    expected = {
        "random_forest",
        "gradient_boosting",
        "svm_rbf",
        "xgboost",
        "mlp"
    }

    assert set(models.keys()) == expected

import pandas as pd
from sklearn.dummy import DummyClassifier
from src.train import run_experiment


def test_run_experiment_excludes_target_and_time():
    df = pd.DataFrame({
        "age": range(40),
        "feature": [i % 3 for i in range(40)],
        "time": [1000 + i for i in range(40)],
        "DEATH_EVENT": [i % 2 for i in range(40)]
    })

    models = {
        "dummy": DummyClassifier(strategy="prior")
    }

    results, outputs = run_experiment(
        df,
        models,
        n_splits=5,
        n_repeats=2
    )

    output = outputs["dummy"]

    assert "DEATH_EVENT" not in output["X"].columns
    assert "time" not in output["X"].columns
    assert list(output["X"].columns) == ["age", "feature"]

    assert len(results["dummy"]) == 10
    assert len(output["y_true"]) == len(df)
    assert len(output["y_prob"]) == len(df)
    assert len(output["y_pred"]) == len(df)