
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
