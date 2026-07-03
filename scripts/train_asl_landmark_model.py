#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny ASL classifier on MediaPipe landmark CSV data.")
    parser.add_argument("--csv", default="data/asl_landmarks.csv", help="Input CSV with label + 63 features.")
    parser.add_argument("--output", default="models/asl_landmark_mlp.joblib", help="Model output path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split.")
    return parser.parse_args()


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    y = np.asarray([row[0] for row in rows])
    x = np.asarray([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
    if x.shape[1] != 63:
        raise ValueError(f"Expected 63 landmark features, got {x.shape[1]}")
    return x, y


def main() -> None:
    args = parse_args()
    x, y = load_csv(args.csv)

    stratify = y if len(set(y)) > 1 and min(np.bincount(np.unique(y, return_inverse=True)[1])) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=stratify
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=700,
            random_state=42,
            early_stopping=True,
        ),
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print(classification_report(y_test, preds, zero_division=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
