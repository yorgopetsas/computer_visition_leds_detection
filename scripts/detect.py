from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ledcheck.detector import LEDPlateDetector


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live LED plate checks from webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--cam-width", type=int, default=1920, help="Requested camera width.")
    parser.add_argument("--cam-height", type=int, default=1080, help="Requested camera height.")
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
    parser.add_argument(
        "--fixed-corners",
        action="store_true",
        help="Use calibration corners directly (best for static camera/plate).",
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
    detector = LEDPlateDetector(
        (ROOT / args.config_dir).resolve(),
        use_config_corners=args.fixed_corners,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cv2.namedWindow("led-check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("led-check", 1400, 900)

    while True:
        if cv2.getWindowProperty("led-check", cv2.WND_PROP_VISIBLE) < 1:
            break
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
        if result.corners and len(result.corners) == 4:
            pts = np.array(result.corners, dtype=np.int32).reshape(-1, 1, 2)
            for p in pts:
                cv2.circle(overlay, tuple(p[0]), 6, (0, 255, 255), -1)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"{result.plate_name} score={result.match_score:.3f}",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        cv2.imshow("led-check", overlay)
        if args.once:
            print_result(result)
            while True:
                if cv2.getWindowProperty("led-check", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.waitKey(50) & 0xFF != 255:
                    break
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

