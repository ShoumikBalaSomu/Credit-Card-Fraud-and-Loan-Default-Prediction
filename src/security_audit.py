"""
Security, Leakage & Robustness Audit Suite for Financial Risk ML (CSE431).

Provides rigorous programmatic audit checks, PCI-DSS / GDPR compliance assertions,
temporal leakage verifications, Track B target contamination guards, and an
adversarial robustness evaluation suite.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ==============================================================================
# 1. PII RETENTION & COMPLIANCE AUDITOR (PCI-DSS & GDPR)
# ==============================================================================

@dataclass
class PIIAuditResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED (COMPLIANT)" if self.passed else "FAILED (NON-COMPLIANT)"
        msg = [f"[PII/PCI-DSS Audit] Status: {status}"]
        if self.violations:
            msg.append("  CRITICAL Violations:")
            for v in self.violations:
                msg.append(f"    - {v}")
        if self.warnings:
            msg.append("  Warnings:")
            for w in self.warnings:
                msg.append(f"    - {w}")
        return "\n".join(msg)


class PIIComplianceAuditor:
    """
    Audits feature schemas and datasets against:
    - PCI-DSS Requirement 3: Protecting stored cardholder data (PAN encryption/tokenization).
    - GDPR Article 5(1)(c): Data minimization (prohibiting unneeded direct identifiers like name/street).
    """

    STRICT_PROHIBITED_COLUMNS = {
        "first": "GDPR Direct PII: First Name",
        "last": "GDPR Direct PII: Last Name",
        "street": "GDPR Direct PII: Physical Residential Address",
        "trans_num": "Transaction UUID / Non-generalizable identifier",
        "Unnamed: 0": "Arbitrary index column artifact",
        "dob": "GDPR Direct PII: Raw Date of Birth (must be converted to age in years)",
        "cc_num": "PCI-DSS Req 3.4: Cleartext Primary Account Number (PAN)",
    }

    PAN_REGEX = re.compile(r"^\d{13,19}$")

    @classmethod
    def audit_feature_columns(cls, feature_columns: List[str]) -> PIIAuditResult:
        violations = []
        warnings = []

        for col in feature_columns:
            col_lower = col.lower().strip()
            if col_lower in cls.STRICT_PROHIBITED_COLUMNS:
                violations.append(
                    f"Forbidden column detected: '{col}' -> {cls.STRICT_PROHIBITED_COLUMNS[col_lower]}"
                )
            elif "cc_num" in col_lower or "card_num" in col_lower or "pan" in col_lower:
                violations.append(
                    f"Possible cleartext PAN column: '{col}'. Must use tokenized/hashed representation."
                )
            elif col_lower in ("first_name", "last_name", "full_name", "customer_name"):
                violations.append(f"Direct PII name column detected: '{col}'.")
            elif col_lower in ("ssn", "social_security", "national_id"):
                violations.append(f"Critical PII identity column detected: '{col}'.")

        passed = len(violations) == 0
        return PIIAuditResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            sanitized_features=[c for c in feature_columns if c.lower() not in cls.STRICT_PROHIBITED_COLUMNS],
        )

    @classmethod
    def audit_dataframe_values(cls, df: pd.DataFrame, sample_size: int = 1000) -> PIIAuditResult:
        col_audit = cls.audit_feature_columns(df.columns.tolist())
        violations = list(col_audit.violations)
        warnings = list(col_audit.warnings)

        sample = df.head(sample_size)
        for col in sample.columns:
            if sample[col].dtype == object or pd.api.types.is_string_dtype(sample[col]):
                non_null = sample[col].dropna().astype(str).str.strip()
                if non_null.empty:
                    continue
                pan_matches = non_null.apply(lambda val: bool(cls.PAN_REGEX.match(val))).sum()
                if pan_matches > 0 and pan_matches / len(non_null) > 0.5:
                    violations.append(
                        f"Column '{col}' appears to contain raw PAN numbers ({pan_matches}/{len(non_null)} match 13-19 digit format)."
                    )
            elif pd.api.types.is_integer_dtype(sample[col]):
                max_val = sample[col].max()
                min_val = sample[col].min()
                if min_val > 10**12 and max_val < 10**20:
                    if col not in ("unix_time", "trans_date_trans_time"):
                        warnings.append(
                            f"Column '{col}' has integer values in the range of credit card numbers ({min_val:.0f}..{max_val:.0f}). Verify it is not a raw PAN."
                        )

        passed = len(violations) == 0
        return PIIAuditResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            sanitized_features=col_audit.sanitized_features,
            metadata={"sample_size": sample_size, "total_rows": len(df)},
        )

    @staticmethod
    def tokenize_pan(cc_series: pd.Series, salt: str = "cse431_fraud_salt_v1") -> pd.Series:
        def hash_val(val: Any) -> str:
            if pd.isna(val):
                return "TOKEN_NA"
            token = hashlib.sha256(f"{salt}_{val}".encode("utf-8")).hexdigest()[:16]
            return f"tok_{token}"

        return cc_series.apply(hash_val)

    @staticmethod
    def derive_age(dob_series: pd.Series, trans_date_series: pd.Series) -> pd.Series:
        dob_dt = pd.to_datetime(dob_series)
        trans_dt = pd.to_datetime(trans_date_series)
        age_years = (trans_dt - dob_dt).dt.total_seconds() / (365.25 * 86400.0)
        return age_years.astype(np.float32)


# ==============================================================================
# 2. TEMPORAL LEAKAGE AUDITOR
# ==============================================================================

@dataclass
class TemporalAuditResult:
    passed: bool
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED (NO LEAKAGE)" if self.passed else "FAILED (LEAKAGE DETECTED)"
        msg = [f"[Temporal Leakage Audit] Status: {status}"]
        msg.append(f"  Train Window: {self.train_range[0]} -> {self.train_range[1]}")
        if self.val_range:
            msg.append(f"  Val Window:   {self.val_range[0]} -> {self.val_range[1]}")
        if self.test_range:
            msg.append(f"  Test Window:  {self.test_range[0]} -> {self.test_range[1]}")
        if self.violations:
            msg.append("  Violations:")
            for v in self.violations:
                msg.append(f"    - {v}")
        return "\n".join(msg)


class TemporalLeakageAuditor:
    @classmethod
    def audit_splits(
        cls,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
        time_col: str = "trans_date_trans_time",
    ) -> TemporalAuditResult:
        violations = []
        train_times = pd.to_datetime(df_train[time_col])
        tr_min, tr_max = train_times.min(), train_times.max()

        v_min, v_max = None, None
        if df_val is not None and not df_val.empty:
            val_times = pd.to_datetime(df_val[time_col])
            v_min, v_max = val_times.min(), val_times.max()
            if tr_max >= v_min:
                violations.append(
                    f"Train/Val Temporal Overlap: train max ({tr_max}) >= val min ({v_min}). Overlap duration: {tr_max - v_min}"
                )

        te_min, te_max = None, None
        if df_test is not None and not df_test.empty:
            test_times = pd.to_datetime(df_test[time_col])
            te_min, te_max = test_times.min(), test_times.max()

            ref_max = v_max if v_max is not None else tr_max
            ref_name = "Val" if v_max is not None else "Train"
            if ref_max >= te_min:
                violations.append(
                    f"{ref_name}/Test Temporal Overlap: {ref_name} max ({ref_max}) >= test min ({te_min}). Overlap duration: {ref_max - te_min}"
                )

        passed = len(violations) == 0
        return TemporalAuditResult(
            passed=passed,
            train_range=(tr_min, tr_max),
            val_range=(v_min, v_max) if v_min is not None else None,
            test_range=(te_min, te_max) if te_min is not None else None,
            violations=violations,
        )


# ==============================================================================
# 3. TRACK B CONTAMINATION AUDITOR
# ==============================================================================

@dataclass
class TrackBAuditResult:
    passed: bool
    feature_window: Tuple[pd.Timestamp, pd.Timestamp]
    target_window: Tuple[pd.Timestamp, pd.Timestamp]
    violations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED (ZERO CONTAMINATION)" if self.passed else "FAILED (CONTAMINATED)"
        msg = [f"[Track B Out-of-Time Contamination Audit] Status: {status}"]
        msg.append(f"  Feature Observation Window: {self.feature_window[0]} -> {self.feature_window[1]}")
        msg.append(f"  Target Prediction Window:   {self.target_window[0]} -> {self.target_window[1]}")
        if self.violations:
            msg.append("  Violations:")
            for v in self.violations:
                msg.append(f"    - {v}")
        return "\n".join(msg)


class TrackBContaminationAuditor:
    @classmethod
    def audit_windows(
        cls,
        feature_cutoff: pd.Timestamp,
        features_max_timestamp: pd.Timestamp,
        target_min_timestamp: pd.Timestamp,
        target_max_timestamp: pd.Timestamp,
    ) -> TrackBAuditResult:
        violations = []
        if features_max_timestamp > feature_cutoff:
            violations.append(
                f"Features leak past cutoff: max feature time ({features_max_timestamp}) > cutoff ({feature_cutoff})."
            )

        if target_min_timestamp < feature_cutoff:
            violations.append(
                f"Target window begins before cutoff: target min time ({target_min_timestamp}) < cutoff ({feature_cutoff})."
            )

        if features_max_timestamp >= target_min_timestamp:
            violations.append(
                f"Direct temporal collision: max feature time ({features_max_timestamp}) >= min target time ({target_min_timestamp})."
            )

        passed = len(violations) == 0
        return TrackBAuditResult(
            passed=passed,
            feature_window=(pd.Timestamp.min, features_max_timestamp),
            target_window=(target_min_timestamp, target_max_timestamp),
            violations=violations,
        )


# ==============================================================================
# 4. ADVERSARIAL ROBUSTNESS SUITE
# ==============================================================================

class AdversarialRobustnessSuite:
    @staticmethod
    def simulate_micro_splitting(
        tx_row: pd.Series,
        n_splits: int = 4,
        delta_seconds: int = 60,
    ) -> pd.DataFrame:
        orig_amt = float(tx_row["amt"])
        split_amt = round(orig_amt / n_splits, 2)
        base_time = pd.to_datetime(tx_row.get("trans_date_trans_time", "2020-01-01 12:00:00"))
        base_unix = int(tx_row.get("unix_time", 1577880000))

        splits = []
        for i in range(n_splits):
            row_copy = tx_row.copy()
            row_copy["amt"] = split_amt
            row_copy["trans_date_trans_time"] = str(base_time + pd.Timedelta(seconds=i * delta_seconds))
            if "unix_time" in row_copy:
                row_copy["unix_time"] = base_unix + (i * delta_seconds)
            splits.append(row_copy)

        return pd.DataFrame(splits)

    @staticmethod
    def perturb_coordinates(
        df: pd.DataFrame,
        lat_col: str = "merch_lat",
        long_col: str = "merch_long",
        sigma_deg: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        df_perturbed = df.copy()
        if lat_col in df_perturbed.columns:
            noise_lat = rng.normal(0, sigma_deg, size=len(df_perturbed))
            df_perturbed[lat_col] = (df_perturbed[lat_col] + noise_lat).clip(-90.0, 90.0)
        if long_col in df_perturbed.columns:
            noise_long = rng.normal(0, sigma_deg, size=len(df_perturbed))
            df_perturbed[long_col] = (df_perturbed[long_col] + noise_long).clip(-180.0, 180.0)
        return df_perturbed

    @staticmethod
    def shift_timestamps(
        df: pd.DataFrame,
        time_col: str = "trans_date_trans_time",
        hour_offset: int = 12,
    ) -> pd.DataFrame:
        df_shifted = df.copy()
        if time_col in df_shifted.columns:
            ts = pd.to_datetime(df_shifted[time_col]) + pd.Timedelta(hours=hour_offset)
            df_shifted[time_col] = ts
            if "unix_time" in df_shifted.columns:
                df_shifted["unix_time"] = df_shifted["unix_time"] + (hour_offset * 3600)
        return df_shifted

    @staticmethod
    def inject_edge_case_nans(
        df: pd.DataFrame,
        fraction: float = 0.05,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        df_stressed = df.copy()

        if numeric_cols:
            for col in numeric_cols:
                if col in df_stressed.columns:
                    mask = rng.rand(len(df_stressed)) < fraction
                    df_stressed.loc[mask, col] = np.nan

        if categorical_cols:
            for col in categorical_cols:
                if col in df_stressed.columns:
                    nan_mask = rng.rand(len(df_stressed)) < (fraction / 2.0)
                    unknown_mask = (rng.rand(len(df_stressed)) < (fraction / 2.0)) & ~nan_mask
                    df_stressed.loc[nan_mask, col] = np.nan
                    df_stressed.loc[unknown_mask, col] = "UNKNOWN_ADVERSARIAL_CATEGORY"

        return df_stressed

    @staticmethod
    def calculate_track_a_financial_loss(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        amounts: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        y_pred = (y_prob >= threshold).astype(int)

        tp = (y_pred == 1) & (y_true == 1)
        fp = (y_pred == 1) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)
        tn = (y_pred == 0) & (y_true == 0)

        cost_tp = tp.sum() * 5.0
        cost_fp = fp.sum() * 15.0
        cost_fn = (amounts[fn] + 25.0).sum()
        cost_tn = 0.0

        total_loss = cost_tp + cost_fp + cost_fn + cost_tn
        baseline_loss = (amounts[y_true == 1] + 25.0).sum()
        dollar_savings = baseline_loss - total_loss

        return {
            "threshold": float(threshold),
            "tp_count": int(tp.sum()),
            "fp_count": int(fp.sum()),
            "fn_count": int(fn.sum()),
            "tn_count": int(tn.sum()),
            "cost_tp": float(cost_tp),
            "cost_fp": float(cost_fp),
            "cost_fn": float(cost_fn),
            "total_financial_loss": float(total_loss),
            "baseline_approve_all_loss": float(baseline_loss),
            "net_dollar_savings": float(dollar_savings),
            "savings_percentage": float((dollar_savings / baseline_loss) * 100.0) if baseline_loss > 0 else 0.0,
        }

    @staticmethod
    def evaluate_model_robustness(
        model: Any,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        X_clean: pd.DataFrame,
        y_true: np.ndarray,
        amounts: np.ndarray,
        optimal_threshold: float,
        coord_cols: Tuple[str, str] = ("merch_lat", "merch_long"),
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        probs_clean = predict_fn(X_clean)
        loss_clean = AdversarialRobustnessSuite.calculate_track_a_financial_loss(
            y_true, probs_clean, amounts, optimal_threshold
        )

        X_perturbed = AdversarialRobustnessSuite.perturb_coordinates(
            X_clean, lat_col=coord_cols[0], long_col=coord_cols[1], sigma_deg=0.02
        )
        probs_coord = predict_fn(X_perturbed)
        loss_coord = AdversarialRobustnessSuite.calculate_track_a_financial_loss(
            y_true, probs_coord, amounts, optimal_threshold
        )
        max_prob_drift = float(np.max(np.abs(probs_clean - probs_coord)))
        mean_prob_drift = float(np.mean(np.abs(probs_clean - probs_coord)))

        X_nans = AdversarialRobustnessSuite.inject_edge_case_nans(
            X_clean, fraction=0.05, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        try:
            probs_nan = predict_fn(X_nans)
            nan_handled = True
            loss_nan = AdversarialRobustnessSuite.calculate_track_a_financial_loss(
                y_true, probs_nan, amounts, optimal_threshold
            )
        except Exception as e:
            nan_handled = False
            probs_nan = None
            loss_nan = {"error": str(e)}

        report = {
            "baseline": loss_clean,
            "coordinate_perturbation": {
                "financial_loss": loss_coord,
                "max_prob_drift": max_prob_drift,
                "mean_prob_drift": mean_prob_drift,
                "robustness_pass": bool(loss_coord["net_dollar_savings"] > 0 and mean_prob_drift < 0.05),
            },
            "edge_case_nans": {
                "pipeline_survived": nan_handled,
                "financial_loss": loss_nan,
            },
        }
        return report


# ==============================================================================
# 5. RUNTIME ASSERTIONS
# ==============================================================================

def assert_pci_gdpr_compliance(feature_columns: List[str]) -> None:
    result = PIIComplianceAuditor.audit_feature_columns(feature_columns)
    assert result.passed, f"PCI-DSS/GDPR Violation Detected:\n{result.summary()}"


def assert_temporal_split_integrity(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    time_col: str = "trans_date_trans_time",
) -> None:
    result = TemporalLeakageAuditor.audit_splits(df_train, df_val, df_test, time_col=time_col)
    assert result.passed, f"Temporal Leakage Invariant Violated:\n{result.summary()}"


def assert_track_b_zero_overlap(
    feature_max_ts: pd.Timestamp,
    target_min_ts: pd.Timestamp,
) -> None:
    assert feature_max_ts < target_min_ts, (
        f"Track B Target Contamination: feature max timestamp ({feature_max_ts}) "
        f"must be strictly less than target min timestamp ({target_min_ts})."
    )
