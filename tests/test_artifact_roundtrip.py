import tempfile
from pathlib import Path

from src.utils.artifacts import load_artifact, save_artifact


def test_joblib_roundtrip():
    data = {"model_name": "XGBoost", "best_params": {"lr": 0.05, "depth": 6}}
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.joblib"
        save_artifact(data, p)
        assert p.exists()
        loaded = load_artifact(p)
        assert loaded == data
