from pathlib import Path

import kagglehub

from src.utils.logging import setup_logger

logger = setup_logger("download_data")


def download_all_datasets() -> dict:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Kartik Fraud Dataset
    logger.info("Fetching Kartik fraud dataset from Kaggle...")
    try:
        fraud_cache = kagglehub.dataset_download("kartik2112/fraud-detection")
        logger.info(f"Fraud dataset path: {fraud_cache}")
        paths["fraud"] = fraud_cache
    except Exception as e:
        logger.warning(f"Kagglehub download failed for fraud dataset ({e}).")
        paths["fraud"] = None

    # 2. Loan Default Dataset
    logger.info("Fetching Loan Default dataset from Kaggle...")
    try:
        loan_cache = kagglehub.dataset_download(
            "subhamjain/loan-prediction-based-on-customer-behavior"
        )
        logger.info(f"Loan dataset path: {loan_cache}")
        paths["loan"] = loan_cache
    except Exception as e:
        logger.warning(f"Kagglehub download failed for loan dataset ({e}).")
        paths["loan"] = None

    return paths


if __name__ == "__main__":
    download_all_datasets()
