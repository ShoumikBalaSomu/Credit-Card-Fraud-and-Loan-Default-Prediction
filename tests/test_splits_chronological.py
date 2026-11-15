import pandas as pd

from src.data.splits import chronological_fraud_split


def test_chronological_split_ordering():
    dates_train = pd.date_range("2019-01-01", periods=5, freq="D")
    dates_test = pd.date_range("2019-01-06", periods=5, freq="D")
    df1 = pd.DataFrame({"trans_date_trans_time": dates_train, "val": 1})
    df2 = pd.DataFrame({"trans_date_trans_time": dates_test, "val": 2})
    tr, te = chronological_fraud_split(df1, df2)
    assert tr["trans_date_trans_time"].max() < te["trans_date_trans_time"].min()
