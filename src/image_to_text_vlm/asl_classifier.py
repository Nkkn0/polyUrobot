from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .asl_landmarks import HandPrediction


class ASLLandmarkClassifier:
    """Tiny ASL classifier trained on MediaPipe hand landmarks.

    The saved artifact should be produced by `scripts/train_asl_landmark_model.py`.
    It is normally a scikit-learn Pipeline with `predict` and `predict_proba`.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ASL model not found: {self.model_path}. Train one with: "
                "python scripts/train_asl_landmark_model.py --csv data/asl_landmarks.csv "
                "--output models/asl_landmark_mlp.joblib"
            )
        self.model = joblib.load(self.model_path)

    def predict(self, features: np.ndarray) -> HandPrediction:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        label = str(self.model.predict(x)[0])

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            confidence = float(np.max(proba))

        return HandPrediction(label=label, confidence=confidence, features=features)
