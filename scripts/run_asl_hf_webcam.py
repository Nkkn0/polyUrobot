#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
from PIL import Image

from image_to_text_vlm.asl_hf import ASLImageClassifier, DEFAULT_ASL_IMAGE_MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test pretrained HF ASL letter recognition on a webcam feed.")
    parser.add_argument("--model-id", default=DEFAULT_ASL_IMAGE_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Usually 0.")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam image, like a selfie camera.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to display.")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Minimum confidence to accept a letter.")
    parser.add_argument("--stable-frames", type=int, default=7, help="Require same letter for this many recent frames.")
    parser.add_argument(
        "--center-crop",
        type=float,
        default=0.70,
        help="Fraction of the shortest frame side used for the center square crop. Use 1.0 for full frame.",
    )
    parser.add_argument("--save-dir", default="captures", help="Directory for frames saved with the 's' key.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only a model already cached on this machine; do not download from Hugging Face.",
    )
    return parser.parse_args()


def center_crop(frame, fraction: float):
    h, w = frame.shape[:2]
    fraction = max(0.1, min(float(fraction), 1.0))
    side = int(min(h, w) * fraction)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def stable_prediction(history: deque[str], stable_frames: int) -> str | None:
    if len(history) < stable_frames:
        return None
    recent = list(history)[-stable_frames:]
    if len(set(recent)) == 1:
        return recent[0]
    return None


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ASL model. The first run may download weights from Hugging Face...")
    classifier = ASLImageClassifier(model_id=args.model_id, local_files_only=args.local_files_only)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")

    history: deque[str] = deque(maxlen=max(args.stable_frames, 1))
    capture_count = 0

    print("Show one ASL letter inside the box. Press 'q' to quit, 's' to save the crop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        crop, (x1, y1, x2, y2) = center_crop(frame, args.center_crop)
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        top = classifier.predict_topk_pil(pil_image, k=args.top_k)
        label, score = top[0]
        accepted = label if score >= args.min_confidence else "?"
        history.append(accepted)
        stable = stable_prediction(history, args.stable_frames)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        status = f"Prediction: {label} ({score:.2f})"
        stable_text = f"Stable: {stable}" if stable and stable != "?" else "Stable: -"
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, stable_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y = 115
        for pred_label, pred_score in top:
            cv2.putText(
                frame,
                f"{pred_label}: {pred_score:.2f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            y += 28

        cv2.imshow("Pretrained HF ASL Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            capture_path = save_dir / f"asl_crop_{capture_count:04d}_{label}_{score:.2f}.jpg"
            cv2.imwrite(str(capture_path), crop)
            print(f"Saved {capture_path}")
            capture_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
