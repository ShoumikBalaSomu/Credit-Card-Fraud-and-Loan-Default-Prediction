from pathlib import Path
from typing import Any

import joblib


def save_artifact(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_artifact(path: str | Path) -> Any:
    return joblib.load(path)
