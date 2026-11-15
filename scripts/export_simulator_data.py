import json
from pathlib import Path

import pandas as pd


def export_simulator_payload():
    payload = {}

    for task in ["fraud", "loan"]:
        with open(f"reports/results/{task}_results.json", "r", encoding="utf-8") as f:
            res = json.load(f)

        df_curve = pd.read_csv(f"reports/tables/{task}_cost_curve.csv")

        # Extract models table
        models_data = []
        for name, metrics in res["models"].items():
            models_data.append(
                {
                    "name": name,
                    "pr_auc": round(metrics["pr_auc"], 4),
                    "roc_auc": round(metrics["roc_auc"], 4),
                    "recall": round(metrics["recall"], 4),
                    "precision": round(metrics["precision"], 4),
                    "f1": round(metrics["f1"], 4),
                    "f2": round(metrics.get("f2", 0.0), 4),
                    "mcc": round(metrics.get("mcc", 0.0), 4),
                    "brier": round(metrics.get("brier", 0.0), 4),
                    "cost": round(metrics["total_cost"], 2),
                    "cost_savings": round(metrics["cost_savings_pct"], 1),
                    "fit_time": round(metrics.get("fit_time_sec", 0.0), 2),
                }
            )

        # Extract strategies table
        strat_data = []
        for strat_name, strat_metrics in res["cost_strategies"].items():
            strat_data.append(
                {
                    "strategy": strat_name,
                    "threshold": round(strat_metrics["threshold"], 3),
                    "recall": round(strat_metrics["recall"], 4),
                    "precision": round(strat_metrics["precision"], 4),
                    "f1": round(strat_metrics["f1"], 4),
                    "total_cost": round(strat_metrics["total_cost"], 2),
                    "savings_pct": round(strat_metrics["cost_savings_pct"], 1),
                }
            )

        payload[task] = {
            "optimal_threshold": res["optimal_threshold"],
            "cost_matrix": res["cost_matrix"],
            "curve": {
                "threshold": df_curve["threshold"].round(4).tolist(),
                "cost": df_curve["total_cost"].round(2).tolist(),
                "recall": df_curve["recall"].round(4).tolist(),
                "precision": df_curve["precision"].round(4).tolist(),
                "fn": df_curve["FN"].astype(int).tolist(),
                "fp": df_curve["FP"].astype(int).tolist(),
                "tp": df_curve["TP"].astype(int).tolist(),
                "tn": df_curve["TN"].astype(int).tolist(),
            },
            "models": models_data,
            "strategies": strat_data,
        }

    out_file = Path("docs/assets/data/simulator_data.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Simulator data updated successfully at {out_file} ({out_file.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    export_simulator_payload()
