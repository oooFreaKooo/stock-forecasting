from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import joblib


class ModelRegistry:
    """Persist and load model artifacts per walk-forward fold."""

    def __init__(self, models_dir: Union[str, Path]) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def fold_dir(self, fold_id: int) -> Path:
        path = self.models_dir / f"fold_{fold_id:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_fold(
        self,
        fold_id: int,
        model: Any,
        calibrator: Any,
        feature_cols: list[str],
        metrics: dict[str, Any],
    ) -> Path:
        fold_path = self.fold_dir(fold_id)
        model.booster_.save_model(str(fold_path / "model.lgb"))
        joblib.dump(calibrator, fold_path / "calibrator.pkl")
        manifest = {
            "fold_id": fold_id,
            "feature_cols": feature_cols,
            "metrics": metrics,
        }
        (fold_path / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
        return fold_path

    def load_manifest(self, fold_id: int) -> dict:
        path = self.fold_dir(fold_id) / "feature_manifest.json"
        return json.loads(path.read_text())
