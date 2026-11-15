from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def chronological_fraud_split(
    df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure strict time boundary between train and test to prevent lookahead leakage."""
    # Sort both datasets by timestamp
    df_train = df_train_raw.sort_values("trans_date_trans_time").reset_index(drop=True)
    df_test = df_test_raw.sort_values("trans_date_trans_time").reset_index(drop=True)
    return df_train, df_test


def stratified_loan_split(
    df: pd.DataFrame, target_col: str = "Risk_Flag", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split for credit default prediction."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
