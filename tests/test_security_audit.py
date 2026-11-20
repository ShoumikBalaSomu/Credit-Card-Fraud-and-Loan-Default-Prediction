"""
Unit tests for Security, Leakage & Robustness Audit Suite using standard unittest.
"""

import unittest
import pandas as pd
import numpy as np
from src.security_audit import (
    PIIComplianceAuditor,
    TemporalLeakageAuditor,
    TrackBContaminationAuditor,
    AdversarialRobustnessSuite,
    assert_pci_gdpr_compliance,
    assert_temporal_split_integrity,
    assert_track_b_zero_overlap,
)


class TestSecurityAuditSuite(unittest.TestCase):

    def test_pii_audit_detection(self):
        bad_features = ["amt", "category", "cc_num", "first", "last"]
        result = PIIComplianceAuditor.audit_feature_columns(bad_features)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.violations), 3)

        good_features = ["amt", "category", "cardholder_token", "age_years", "distance_km"]
        clean_result = PIIComplianceAuditor.audit_feature_columns(good_features)
        self.assertTrue(clean_result.passed)
        self.assertEqual(len(clean_result.violations), 0)

    def test_pii_tokenization(self):
        s = pd.Series([1234567890123456, 9876543210987654, np.nan])
        tokens = PIIComplianceAuditor.tokenize_pan(s, salt="test_salt")
        self.assertTrue(tokens.iloc[0].startswith("tok_"))
        self.assertTrue(tokens.iloc[1].startswith("tok_"))
        self.assertEqual(tokens.iloc[2], "TOKEN_NA")
        self.assertNotEqual(tokens.iloc[0], tokens.iloc[1])

    def test_temporal_leakage_audit(self):
        df_train = pd.DataFrame({"trans_date_trans_time": ["2019-01-01", "2019-06-01"]})
        df_val = pd.DataFrame({"trans_date_trans_time": ["2019-06-02", "2019-12-01"]})
        df_test = pd.DataFrame({"trans_date_trans_time": ["2019-12-02", "2020-05-01"]})

        res = TemporalLeakageAuditor.audit_splits(df_train, df_val, df_test)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.violations), 0)

        df_val_leaky = pd.DataFrame({"trans_date_trans_time": ["2019-05-15", "2019-12-01"]})
        res_leaky = TemporalLeakageAuditor.audit_splits(df_train, df_val_leaky, df_test)
        self.assertFalse(res_leaky.passed)
        self.assertGreater(len(res_leaky.violations), 0)

    def test_track_b_contamination_audit(self):
        cutoff = pd.Timestamp("2020-03-01 00:00:00")
        feat_max = pd.Timestamp("2020-02-28 23:59:59")
        targ_min = pd.Timestamp("2020-03-01 00:00:01")
        targ_max = pd.Timestamp("2020-06-20 23:59:59")

        res = TrackBContaminationAuditor.audit_windows(cutoff, feat_max, targ_min, targ_max)
        self.assertTrue(res.passed)

        feat_max_bad = pd.Timestamp("2020-03-05 00:00:00")
        res_bad = TrackBContaminationAuditor.audit_windows(cutoff, feat_max_bad, targ_min, targ_max)
        self.assertFalse(res_bad.passed)

    def test_adversarial_micro_splitting(self):
        sample = pd.Series({
            "amt": 800.0,
            "trans_date_trans_time": "2020-01-01 10:00:00",
            "unix_time": 1577872800,
            "category": "grocery_pos",
        })
        splits = AdversarialRobustnessSuite.simulate_micro_splitting(sample, n_splits=4, delta_seconds=60)
        self.assertEqual(len(splits), 4)
        self.assertAlmostEqual(splits["amt"].sum(), 800.0)
        self.assertAlmostEqual(splits["amt"].iloc[0], 200.0)
        self.assertEqual(splits["unix_time"].iloc[1], 1577872800 + 60)

    def test_coordinate_perturbation(self):
        df = pd.DataFrame({
            "merch_lat": [40.7128, 34.0522],
            "merch_long": [-74.0060, -118.2437],
        })
        perturbed = AdversarialRobustnessSuite.perturb_coordinates(df, sigma_deg=0.02, seed=42)
        self.assertFalse(np.allclose(df["merch_lat"], perturbed["merch_lat"]))
        self.assertTrue(np.all(np.abs(df["merch_lat"] - perturbed["merch_lat"]) < 0.1))

    def test_cost_matrix_calculation(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.2, 0.8, 0.1])
        amounts = np.array([100.0, 500.0, 50.0, 20.0])

        loss_dict = AdversarialRobustnessSuite.calculate_track_a_financial_loss(
            y_true, y_prob, amounts, threshold=0.5
        )
        self.assertEqual(loss_dict["tp_count"], 1)
        self.assertEqual(loss_dict["fn_count"], 1)
        self.assertEqual(loss_dict["fp_count"], 1)
        self.assertEqual(loss_dict["tn_count"], 1)
        self.assertEqual(loss_dict["cost_tp"], 5.0)
        self.assertEqual(loss_dict["cost_fp"], 15.0)
        self.assertEqual(loss_dict["cost_fn"], 525.0)
        self.assertEqual(loss_dict["total_financial_loss"], 545.0)
        self.assertEqual(loss_dict["baseline_approve_all_loss"], 650.0)
        self.assertEqual(loss_dict["net_dollar_savings"], 105.0)


if __name__ == "__main__":
    unittest.main()
