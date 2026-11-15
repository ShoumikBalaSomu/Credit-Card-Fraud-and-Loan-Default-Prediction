from typing import List, Tuple

import pandas as pd


def format_contrastive_explanation(
    task_type: str,
    outcome_name: str,
    instance_raw: pd.Series,
    proba: float,
    threshold: float,
    top_contributors: List[Tuple[str, float]] = None,
) -> str:
    lines = []
    lines.append(f"### Case: {outcome_name}")
    lines.append(f"- **Domain Task:** `{task_type.upper()}` Risk Classification")
    lines.append(
        f"- **Model Predicted Probability:** `{float(proba):.4f}` (Decision Threshold: `{float(threshold):.2f}`)"
    )
    status = "FLAGGED / POSITIVE" if float(proba) >= float(threshold) else "CLEARED / NEGATIVE"
    lines.append(f"- **Decision:** `{status}`")
    lines.append("- **Instance Attributes:**")
    for k, v in instance_raw.items():
        if k in [
            "amt",
            "category",
            "city_pop",
            "dist_to_merch_km",
            "age",
            "Income",
            "Age",
            "Experience",
            "House_Ownership",
            "Car_Ownership",
        ]:
            val_str = f"${v:,.2f}" if "amt" in k.lower() or "income" in k.lower() else str(v)
            lines.append(f"  - `{k}`: {val_str}")
    if top_contributors:
        lines.append("- **Key Attribution Drivers (SHAP values):**")
        for feat, impact in top_contributors:
            direction = "increased risk" if impact > 0 else "decreased risk"
            lines.append(f"  - `{feat}`: {impact:+.4f} ({direction})")
    lines.append("")
    return "\n".join(lines)
