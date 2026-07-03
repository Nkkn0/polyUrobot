from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HandPrediction:
    label: str
    confidence: float
    features: np.ndarray


def normalize_landmarks(landmarks: Iterable[Iterable[float]]) -> np.ndarray:
    """Normalize 21 MediaPipe hand landmarks into a scale/translation-light feature vector.

    Input shape must be (21, 3): x, y, z for each hand point. The output is a
    63-value float32 vector. We subtract the wrist point and scale by palm size,
    which makes the classifier less sensitive to where the hand appears in frame.
    """
    arr = np.asarray(list(landmarks), dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {arr.shape}")

    wrist = arr[0].copy()
    arr = arr - wrist

    # Use wrist -> middle-finger MCP distance as a stable palm scale.
    palm_scale = float(np.linalg.norm(arr[9]))
    if palm_scale < 1e-6:
        palm_scale = float(np.max(np.linalg.norm(arr, axis=1)))
    if palm_scale < 1e-6:
        palm_scale = 1.0

    arr = arr / palm_scale
    return arr.reshape(-1).astype(np.float32)


def landmarks_from_mediapipe_hand(hand_landmarks) -> np.ndarray:
    """Convert one MediaPipe hand result into normalized features."""
    points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    return normalize_landmarks(points)


def extract_features_from_bgr(frame_bgr, hands) -> np.ndarray | None:
    """Extract normalized hand landmark features from an OpenCV BGR frame.

    `hands` should be a `mediapipe.solutions.hands.Hands` object. Returns None
    when no hand is detected.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install OpenCV first: pip install opencv-python") from exc

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    return landmarks_from_mediapipe_hand(result.multi_hand_landmarks[0])


def save_landmark_row(csv_path: str | Path, label: str, features: np.ndarray) -> None:
    """Append one labeled landmark vector to a CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = ",".join([label, *[f"{float(x):.8f}" for x in features]])
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")
