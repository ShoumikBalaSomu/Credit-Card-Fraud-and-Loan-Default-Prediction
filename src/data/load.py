from pathlib import Path
from typing import Optional, Tuple

import kagglehub
import pandas as pd

from src.data.synthetic_fallback import make_synthetic_fraud_data, make_synthetic_loan_data
from src.utils.logging import setup_logger

logger = setup_logger("load_data")


def load_fraud_raw(
    max_train_rows: Optional[int] = 100000, max_test_rows: Optional[int] = 40000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load credit card fraud train and test sets chronologically."""
    cache_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "kartik2112"
        / "fraud-detection"
        / "versions"
        / "1"
    )
    train_file = cache_dir / "fraudTrain.csv"
    test_file = cache_dir / "fraudTest.csv"

    if not train_file.exists():
        try:
            p = kagglehub.dataset_download("kartik2112/fraud-detection")
            train_file = Path(p) / "fraudTrain.csv"
            test_file = Path(p) / "fraudTest.csv"
        except Exception:
            pass

    if train_file.exists() and test_file.exists():
        logger.info(f"Loading real fraud data from {cache_dir}...")
        # Chronological read
        df_train = pd.read_csv(train_file, index_col=0, nrows=max_train_rows)
        df_test = pd.read_csv(test_file, index_col=0, nrows=max_test_rows)
        return df_train, df_test
    else:
        logger.warning("Real fraud dataset not accessible; using synthetic fallback.")
        df_train = make_synthetic_fraud_data(n_samples=max_train_rows or 20000, random_state=42)
        df_test = make_synthetic_fraud_data(n_samples=max_test_rows or 10000, random_state=99)
        return df_train, df_test


def load_loan_raw(max_rows: Optional[int] = 60000) -> pd.DataFrame:
    """Load customer loan default risk dataset."""
    cache_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "subhamjain"
        / "loan-prediction-based-on-customer-behavior"
        / "versions"
        / "1"
    )
    file_path = cache_dir / "Training Data.csv"

    if not file_path.exists():
        try:
            p = kagglehub.dataset_download("subhamjain/loan-prediction-based-on-customer-behavior")
            file_path = Path(p) / "Training Data.csv"
        except Exception:
            pass

    if file_path.exists():
        logger.info(f"Loading real loan data from {file_path}...")
        df = pd.read_csv(file_path, nrows=max_rows)
        return df
    else:
        logger.warning("Real loan dataset not accessible; using synthetic fallback.")
        return make_synthetic_loan_data(n_samples=max_rows or 20000, random_state=42)
