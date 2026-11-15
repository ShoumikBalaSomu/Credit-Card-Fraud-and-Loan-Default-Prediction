import numpy as np
import pandas as pd

from src.evaluate.fairness import (
    audit_subgroup_fairness,
    compute_disparate_impact_ratio,
)


def test_fairness_equal_treatment():
    # If model treats groups identically, disparate impact should be ~1.0
    np.random.seed(42)
    n = 1000
    sensitive_group = np.random.binomial(1, 0.5, size=n)
    # Decisions independent of group
    y_pred = np.random.binomial(1, 0.15, size=n)

    di = compute_disparate_impact_ratio(y_pred, sensitive_group, positive_is_adverse=True)
    assert 0.90 <= di <= 1.0, f"Expected near 1.0 disparate impact, got {di}"


def test_fairness_flags_discriminatory_model():
    # Simulating a biased model where group 1 is denied 3x more often
    n = 1000
    sensitive_group = np.random.binomial(1, 0.5, size=n)
    y_pred = np.zeros(n, dtype=int)
    # Reject 10% of group 0, but 60% of group 1
    y_pred[sensitive_group == 0] = np.random.binomial(1, 0.10, size=np.sum(sensitive_group == 0))
    y_pred[sensitive_group == 1] = np.random.binomial(1, 0.60, size=np.sum(sensitive_group == 1))

    di = compute_disparate_impact_ratio(y_pred, sensitive_group, positive_is_adverse=True)
    # Group 1 approval is 40%, Group 0 approval is 90% -> 0.40 / 0.90 = 0.44 (< 0.80)
    assert di < 0.80, f"Biased decisions must fail 80% four-fifths rule, got {di}"


def test_audit_subgroup_fairness_full_report():
    df = pd.DataFrame(
        {
            "age_bracket": ["young", "young", "old", "old", "young", "old"],
        }
    )
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])

    report = audit_subgroup_fairness(df, "age_bracket", y_true, y_pred, positive_is_adverse=True)
    assert report["sensitive_column"] == "age_bracket"
    assert "disparate_impact_ratio" in report
    assert "passes_four_fifths_rule" in report
    assert "group_metrics" in report
    assert "young" in report["group_metrics"]
    assert "old" in report["group_metrics"]
