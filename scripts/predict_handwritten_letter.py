#!/usr/bin/env python3
"""Predict the A-Z letter in a handwritten character image.

Example:
    python scripts/predict_handwritten_letter.py examples/my_letter.jpg --top-k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_to_text_vlm.handwriting_hf import (
    DEFAULT_HANDWRITING_MODEL_ID,
    HandwrittenLetterRecognizer,
    format_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten A-Z letter from an image.")
    parser.add_argument("image", type=Path, help="Path to an image containing one handwritten letter.")
    parser.add_argument("--model", default=DEFAULT_HANDWRITING_MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show.")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic crop around the ink.")
    parser.add_argument("--invert", action="store_true", help="Use if the letter is light on a dark background.")
    parser.add_argument(
        "--save-preprocessed",
        type=Path,
        default=None,
        help="Optional path to save the preprocessed image for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    recognizer = HandwrittenLetterRecognizer(model_id=args.model)
    predictions = recognizer.predict(
        args.image,
        top_k=args.top_k,
        crop=not args.no_crop,
        invert=args.invert,
        save_preprocessed=args.save_preprocessed,
    )

    best = predictions[0]
    print(f"Predicted letter: {best.letter} ({best.confidence:.2%})")
    print()
    print(format_predictions(predictions))

    if args.save_preprocessed:
        print(f"\nSaved preprocessed image to: {args.save_preprocessed}")


if __name__ == "__main__":
    main()
