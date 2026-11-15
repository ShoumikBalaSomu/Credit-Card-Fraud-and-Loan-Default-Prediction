import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_fraud_df, preprocess_loan_df
from src.features.engineering import build_fraud_feature_pipeline, build_loan_feature_pipeline


def test_extreme_fraud_transaction_values():
    df_extreme = pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2020-01-01 00:00:01",
                "amt": 0.0,  # Zero dollar transaction
                "category": "grocery_pos",
                "lat": 90.0,  # Extreme latitude
                "long": 180.0,
                "merch_lat": -90.0,
                "merch_long": -180.0,
                "dob": "1920-01-01",  # 100-year-old cardholder
                "gender": "F",
                "city_pop": 0,
            },
            {
                "trans_date_trans_time": "2020-12-31 23:59:59",
                "amt": 500000.0,  # Ultra-high transaction amount
                "category": "misc_net",
                "lat": 0.0,
                "long": 0.0,
                "merch_lat": 0.0,
                "merch_long": 0.0,
                "dob": "2002-05-15",
                "gender": "M",
                "city_pop": 10000000,
            },
        ]
    )

    df_proc = preprocess_fraud_df(df_extreme)
    assert not df_proc.isnull().values.any()
    assert df_proc["amt_log"].iloc[0] == 0.0
    assert df_proc["amt_log"].iloc[1] > 10.0
    assert df_proc["age"].iloc[0] >= 100.0

    prep, _ = build_fraud_feature_pipeline()
    X = prep.fit_transform(df_proc)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_thin_file_loan_applicant():
    df_thin = pd.DataFrame(
        [
            {
                "Income": 0,
                "Age": 18,
                "Experience": 0,
                "Married/Single": "single",
                "House_Ownership": "norent_noown",
                "Car_Ownership": "no",
                "CURRENT_JOB_YRS": 0,
                "CURRENT_HOUSE_YRS": 0,
            }
        ]
    )

    df_proc = preprocess_loan_df(df_thin)
    assert not df_proc.isnull().values.any()
    assert df_proc["income_log"].iloc[0] == 0.0
    assert df_proc["exp_to_age_ratio"].iloc[0] == 0.0

    prep, _ = build_loan_feature_pipeline()
    X = prep.fit_transform(df_proc)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
