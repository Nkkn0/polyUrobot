import numpy as np
import pytest

from image_to_text_vlm.asl_landmarks import normalize_landmarks


def test_normalize_landmarks_returns_63_features():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[9] = [0.0, 2.0, 0.0]
    features = normalize_landmarks(landmarks)
    assert features.shape == (63,)
    assert features.dtype == np.float32


def test_normalize_landmarks_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_landmarks([[0.0, 0.0, 0.0]])
