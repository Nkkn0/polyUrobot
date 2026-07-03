#!/usr/bin/env python
from __future__ import annotations

import argparse

from image_to_text_vlm.asl_hf import ASLImageClassifier, DEFAULT_ASL_IMAGE_MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one ASL letter image using a light pretrained HF model.")
    parser.add_argument("image", help="Path to a hand-sign image.")
    parser.add_argument("--model-id", default=DEFAULT_ASL_IMAGE_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to print.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only a model already cached on this machine; do not download from Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clf = ASLImageClassifier(model_id=args.model_id, local_files_only=args.local_files_only)
    for label, score in clf.predict_topk(args.image, k=args.top_k):
        print(f"{label}\t{score:.4f}")


if __name__ == "__main__":
    main()
