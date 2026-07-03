from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageChops, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


DEFAULT_HANDWRITING_MODEL_ID = "microsoft/trocr-base-handwritten"
DEFAULT_MODEL_ID = DEFAULT_HANDWRITING_MODEL_ID
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class LetterPrediction:
    letter: str
    confidence: float
    raw_label: str


def _open_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def preprocess_handwritten_letter(
    image,
    *,
    crop: bool = True,
    invert: bool = False,
    save_preprocessed: str | None = None,
) -> Image.Image:
    img = _open_image(image)

    gray = ImageOps.grayscale(img)

    if crop:
        bg = Image.new("L", gray.size, 255)
        diff = ImageChops.difference(gray, bg)
        bbox = diff.getbbox()

        if bbox:
            pad = 30
            left = max(0, bbox[0] - pad)
            top = max(0, bbox[1] - pad)
            right = min(gray.size[0], bbox[2] + pad)
            bottom = min(gray.size[1], bbox[3] + pad)
            gray = gray.crop((left, top, right, bottom))

    if invert:
        gray = ImageOps.invert(gray)

    # Put the handwriting on a clean white canvas.
    gray.thumbnail((384, 384))
    canvas = Image.new("L", (384, 384), 255)
    x = (384 - gray.size[0]) // 2
    y = (384 - gray.size[1]) // 2
    canvas.paste(gray, (x, y))

    out = canvas.convert("RGB")

    if save_preprocessed:
        Path(save_preprocessed).parent.mkdir(parents=True, exist_ok=True)
        out.save(save_preprocessed)

    return out


preprocess_handwriting_image = preprocess_handwritten_letter


class HandwrittenLetterRecognizer:
    def __init__(
        self,
        model_id: str = DEFAULT_HANDWRITING_MODEL_ID,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or "cpu"
        self._processor = None
        self._model = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = TrOCRProcessor.from_pretrained(self.model_id)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def predict(
        self,
        image,
        *,
        top_k: int = 5,
        crop: bool = True,
        invert: bool = False,
        save_preprocessed: str | None = None,
    ) -> List[LetterPrediction]:
        img = preprocess_handwritten_letter(
            image,
            crop=crop,
            invert=invert,
            save_preprocessed=save_preprocessed,
        )

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=8)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned = "".join(ch for ch in text.upper() if ch.isalpha())

        if cleaned:
            predicted = cleaned[0]
        else:
            predicted = "?"

        return [
            LetterPrediction(
                letter=predicted,
                confidence=1.0 if predicted in LETTERS else 0.0,
                raw_label=text,
            )
        ]


HandwritingRecognizer = HandwrittenLetterRecognizer


def format_predictions(predictions: List[LetterPrediction]) -> str:
    lines = [
        "rank  letter  confidence  raw_output",
        "----  ------  ----------  ----------",
    ]

    for i, pred in enumerate(predictions, start=1):
        lines.append(
            f"{i:<4}  {pred.letter:<6}  {pred.confidence * 100:>8.2f}%  {pred.raw_label}"
        )

    return "\n".join(lines)
