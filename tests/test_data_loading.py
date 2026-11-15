from pathlib import Path

from src.data.synthetic_fallback import make_synthetic_fraud_data, make_synthetic_loan_data


def test_synthetic_fraud_generator():
    df = make_synthetic_fraud_data(n_samples=500, fraud_ratio=0.02)
    assert len(df) == 500
    assert "is_fraud" in df.columns
    assert df["is_fraud"].sum() == 10


def test_synthetic_loan_generator():
    df = make_synthetic_loan_data(n_samples=500, default_ratio=0.10)
    assert len(df) == 500
    assert "Risk_Flag" in df.columns
    assert df["Risk_Flag"].sum() > 0


def test_fixtures_exist():
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    assert (fixture_dir / "fraud_sample.csv").exists()
    assert (fixture_dir / "loan_sample.csv").exists()
