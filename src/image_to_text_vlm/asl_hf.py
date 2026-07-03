from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


DEFAULT_ASL_IMAGE_MODEL_ID = "abdollahhh/asl-sign-language-efficientnet-b0"


class ASLImageClassifier:
    """Light pretrained ASL image classifier from Hugging Face.

    This is useful for quick demos from still images or low-FPS webcam testing.
    For reliable real-time robot control, the MediaPipe-landmark classifier is
    usually easier to stabilize frame-to-frame.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_ASL_IMAGE_MODEL_ID,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "The pretrained ASL image classifier needs torch, transformers, and pillow. "
                "Install them with: pip install -r requirements-asl-hf.txt"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def _label_for_index(self, idx: int) -> str:
        # HF configs sometimes store id2label with int keys, sometimes string keys.
        return str(self.model.config.id2label.get(idx, self.model.config.id2label.get(str(idx), idx)))

    def predict_pil(self, image: Image.Image) -> tuple[str, float]:
        """Return the most likely ASL letter and confidence for a PIL image."""
        results = self.predict_topk_pil(image, k=1)
        return results[0]

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        """Return the most likely ASL letter and confidence for an image file."""
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")
        return self.predict_pil(image)

    def predict_topk_pil(self, image: Image.Image, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k ASL predictions as (label, confidence)."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0]

        k = max(1, min(int(k), probs.numel()))
        values, indices = self.torch.topk(probs, k=k)
        return [
            (self._label_for_index(int(idx.item())), float(score.item()))
            for score, idx in zip(values, indices)
        ]

    def predict_topk(self, image_path: str | Path, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k ASL predictions for an image file."""
        from PIL import Image

        image = Image.open(Path(image_path)).convert("RGB")
        return self.predict_topk_pil(image, k=k)
