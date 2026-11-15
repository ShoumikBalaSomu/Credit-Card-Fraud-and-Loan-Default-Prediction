from src.utils.io import load_json


def summarize_thresholds():
    for task in ["fraud", "loan"]:
        res = load_json(f"reports/results/{task}_results.json")
        print(f"=== THRESHOLD SUMMARY FOR {task.upper()} ===")
        print(f'Optimal Threshold: {res["optimal_threshold"]:.3f}')
        print(
            f'Baseline (0.50) Cost: ${res["cost_strategies"]["Baseline (Unweighted, p=0.5)"]["total_cost"]:,.2f}'
        )
        print(
            f'Optimal Cost: ${res["cost_strategies"]["Threshold Optimization (p*)"]["total_cost"]:,.2f}'
        )
        print(
            f'Savings: {res["cost_strategies"]["Threshold Optimization (p*)"]["cost_savings_pct"]:.2f}%\n'
        )


if __name__ == "__main__":
    summarize_thresholds()
