from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_fraud_df
from src.features.engineering import build_fraud_feature_pipeline


def test_no_future_leakage_in_scaling():
    prep, _ = build_fraud_feature_pipeline()
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "fraud_sample.csv"
    df_raw = pd.read_csv(fixture_path)
    df = preprocess_fraud_df(df_raw)
    df_train = df.iloc[:100]
    df_test = df.iloc[100:]
    X_train_proc = prep.fit_transform(df_train)
    scaler = prep.named_transformers_["num"]
    train_mean = scaler.mean_.copy()
    X_test_proc = prep.transform(df_test)
    assert np.allclose(scaler.mean_, train_mean)
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
