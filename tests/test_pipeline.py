from src.data import load_data


def test_load_data():
    df = load_data("data/raw/heart_failure_clinical_records_dataset.csv")
    assert df.shape[0] > 0