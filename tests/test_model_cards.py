from src.explain.model_cards import generate_model_card


def test_model_card_generation(tmp_path):
    card_path = tmp_path / "test_model_card.md"
    metrics = {
        "pr_auc": 0.8364,
        "roc_auc": 0.9942,
        "recall": 0.9167,
        "precision": 0.7240,
        "total_cost": 12160.0,
        "cost_savings_pct": 59.8,
        "threshold": 0.023,
        "brier": 0.0031,
    }
    cost_matrix_info = {
        "ratio_description": "40:1 Domain Cost Ratio",
        "cost_fn": 200.0,
        "cost_fp": 5.0,
    }

    generate_model_card(
        task="fraud",
        model_name="CatBoost",
        metrics=metrics,
        cost_matrix_info=cost_matrix_info,
        out_path=card_path,
    )

    assert card_path.exists()
    content = card_path.read_text()
    assert "Model Card" in content
    assert "CatBoost" in content
    assert "0.8364" in content
    assert "0.02" in content
    assert "Limitations & Governance" in content
