#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp

from image_to_text_vlm.asl_classifier import ASLLandmarkClassifier
from image_to_text_vlm.asl_landmarks import extract_features_from_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time ASL recognition from webcam.")
    parser.add_argument("--model", default="models/asl_landmark_mlp.joblib", help="Trained joblib model path.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam view.")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Minimum confidence to accept a letter.")
    parser.add_argument("--stable-frames", type=int, default=7, help="Require same prediction for N frames.")
    parser.add_argument("--serial-port", default=None, help="Optional serial port to send stable letters to robot, e.g. /dev/ttyUSB0.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    return parser.parse_args()


def maybe_open_serial(port: str | None, baud: int):
    if not port:
        return None
    try:
        import serial
    except ImportError as exc:
        raise ImportError("Install pyserial first: pip install pyserial") from exc
    return serial.Serial(port, baudrate=baud, timeout=0.1)


def main() -> None:
    args = parse_args()
    clf = ASLLandmarkClassifier(args.model)
    ser = maybe_open_serial(args.serial_port, args.baud)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    recent: deque[str] = deque(maxlen=args.stable_frames)
    last_sent: str | None = None
    last_sent_at = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            features = extract_features_from_bgr(frame, hands)
            text = "No hand"
            if features is not None:
                pred = clf.predict(features)
                if pred.confidence >= args.min_confidence:
                    recent.append(pred.label)
                else:
                    recent.append("?")

                stable = len(recent) == args.stable_frames and len(set(recent)) == 1 and recent[0] != "?"
                text = f"{pred.label} ({pred.confidence:.2f})"

                if stable:
                    letter = recent[0]
                    now = time.time()
                    # Debounce so the robot does not receive the same letter every frame.
                    if letter != last_sent or now - last_sent_at > 1.5:
                        print(f"Stable letter: {letter}")
                        if ser is not None:
                            ser.write((letter + "\n").encode("utf-8"))
                        last_sent = letter
                        last_sent_at = now
            else:
                recent.clear()

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("ASL realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
