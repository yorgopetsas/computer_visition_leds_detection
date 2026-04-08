from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ledcheck.detector import LEDPlateDetector


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live LED plate checks from webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--config-dir",
        default="configs/plates",
        help="Directory with plate JSON config files.",
    )
    parser.add_argument(
        "--check-led",
        default="",
        help="Optional LED name to check only this LED.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single frame detection and exit.",
    )
    return parser.parse_args()


def print_result(result) -> None:
    print("=" * 60)
    print(result.message)
    if result.plate_name:
        print(f"Plate: {result.plate_name} ({result.plate_id})")
        print(f"Match score: {result.match_score:.3f}")
    if result.leds:
        for led in result.leds:
            state = "ON" if led["on"] else "OFF"
            print(
                f"- {led['name']:<22} {state:<3}  score={led['score']:.3f}  thr={led['threshold']:.3f}"
            )


def main() -> None:
    args = read_args()
    detector = LEDPlateDetector((ROOT / args.config_dir).resolve())
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        result = (
            detector.check_led(frame, args.check_led)
            if args.check_led
            else detector.detect(frame)
        )

        overlay = frame.copy()
        color = (0, 200, 0) if result.ok else (0, 0, 255)
        cv2.putText(
            overlay,
            result.message,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.imshow("led-check", overlay)
        if args.once:
            print_result(result)
            break
        key = cv2.waitKey(15) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord(" "):
            print_result(result)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

