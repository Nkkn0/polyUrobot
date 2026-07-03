#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from image_to_text_vlm.asl_landmarks import extract_features_from_bgr, save_landmark_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL landmark samples from webcam.")
    parser.add_argument("--label", required=True, help="ASL label to collect, e.g. A, B, C.")
    parser.add_argument("--output", default="data/asl_landmarks.csv", help="CSV output path.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--max-samples", type=int, default=100, help="Stop after this many saved samples.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam view.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    saved = 0
    print("Press SPACE to save one sample. Press q to quit.")
    print(f"Collecting label: {args.label}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while saved < args.max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            features = extract_features_from_bgr(frame, hands)
            status = "HAND OK" if features is not None else "NO HAND"
            cv2.putText(frame, f"Label {args.label} | {status} | saved {saved}/{args.max_samples}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Collect ASL samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and features is not None:  # spacebar
                save_landmark_row(output, args.label.upper(), features)
                saved += 1
                print(f"Saved {saved}: {args.label.upper()}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} samples to {output}")


if __name__ == "__main__":
    main()
