import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_fraud_df
from src.features.engineering import build_fraud_feature_pipeline


def test_haversine_distance_zero_when_identical():
    df = pd.DataFrame(
        [
            {
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 40.7128,
                "merch_long": -74.0060,
                "amt": 100.0,
            }
        ]
    )
    df_proc = preprocess_fraud_df(df)
    assert np.isclose(df_proc["dist_to_merch_km"].iloc[0], 0.0, atol=1e-3)


def test_feature_pipeline_output_dimensions():
    df_fraud = pd.DataFrame(
        [
            {
                "amt": 50.0,
                "category": "grocery_pos",
                "gender": "M",
                "lat": 40.0,
                "long": -73.0,
                "merch_lat": 40.1,
                "merch_long": -73.1,
                "city_pop": 50000,
                "dob": "1990-01-01",
                "trans_date_trans_time": "2020-05-01 14:00:00",
            }
        ]
    )
    df_fraud_proc = preprocess_fraud_df(df_fraud)
    prep_fraud, cols_fraud = build_fraud_feature_pipeline()
    X_f = prep_fraud.fit_transform(df_fraud_proc)
    assert X_f.shape[0] == 1
    assert X_f.shape[1] >= len(cols_fraud)
