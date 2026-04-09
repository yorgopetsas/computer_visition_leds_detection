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
from ledcheck.io_utils import load_plate_configs
from ledcheck.vision import warp_plate
from ledcheck_beta.refine import refine_leds_local


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beta: AI-assisted PCB/LED detection pipeline.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1920)
    parser.add_argument("--cam-height", type=int, default=1080)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--config-dir", default="configs/plates")
    parser.add_argument("--fixed-corners", action="store_true")
    parser.add_argument("--interactive-startup", action="store_true")
    parser.add_argument("--retry-margin", type=float, default=0.02)
    parser.add_argument("--search-radius", type=int, default=16)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = read_args()
    configs = load_plate_configs((ROOT / args.config_dir).resolve())
    detector = LEDPlateDetector((ROOT / args.config_dir).resolve(), use_config_corners=args.fixed_corners)
    forced_plate_id: str | None = None
    manual_corners: list[tuple[int, int]] | None = None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cv2.namedWindow("led-beta", cv2.WINDOW_AUTOSIZE)
    scale = max(0.2, float(args.preview_scale))

    def to_display(img: np.ndarray) -> np.ndarray:
        if scale == 1.0:
            return img
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    def to_original_point(x: int, y: int) -> tuple[int, int]:
        if scale == 1.0:
            return int(x), int(y)
        return int(round(x / scale)), int(round(y / scale))

    if args.interactive_startup:
        ids = sorted(configs.keys())
        print("Select PCB model:")
        for i, pid in enumerate(ids, start=1):
            print(f"  {i}. {pid} ({configs[pid].display_name})")
        while True:
            raw = input("Model number: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(ids):
                forced_plate_id = ids[int(raw) - 1]
                break
            print("Invalid selection.")
        selected: list[tuple[int, int]] = []

        def on_click(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
                selected.append(to_original_point(x, y))

        cv2.setMouseCallback("led-beta", on_click)
        print("Click corners TL, TR, BR, BL then Enter.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            preview = frame.copy()
            cv2.putText(preview, "Click corners TL,TR,BR,BL then Enter", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for i, p in enumerate(selected):
                cv2.circle(preview, p, 8, (0, 255, 255), -1)
                cv2.putText(preview, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(selected) == 4:
                cv2.polylines(preview, [np.array(selected, dtype=np.int32)], True, (0, 255, 255), 2)
            cv2.imshow("led-beta", to_display(preview))
            key = cv2.waitKey(15) & 0xFF
            if key in (10, 13) and len(selected) == 4:
                manual_corners = list(selected)
                break
            if key == ord("r"):
                selected.clear()
            if key in (27, ord("q")):
                cap.release()
                cv2.destroyAllWindows()
                return
        cv2.setMouseCallback("led-beta", lambda *a: None)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        result = detector.detect(
            frame,
            retry_margin=args.retry_margin,
            forced_plate_id=forced_plate_id,
            override_corners=manual_corners,
        )
        overlay = frame.copy()
        cv2.putText(overlay, f"BETA: {result.message}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if result.ok and result.plate_id in configs and result.corners:
            plate = configs[result.plate_id]
            corners = np.array(result.corners, dtype=np.float32)
            warped = warp_plate(frame, corners, plate.canonical_size)
            refined = refine_leds_local(
                warped,
                plate,
                search_radius_px=args.search_radius,
                retry_margin=args.retry_margin,
            )

            src = np.array(
                [[0.0, 0.0], [plate.canonical_size[0] - 1.0, 0.0], [plate.canonical_size[0] - 1.0, plate.canonical_size[1] - 1.0], [0.0, plate.canonical_size[1] - 1.0]],
                dtype=np.float32,
            )
            m = cv2.getPerspectiveTransform(src, corners)
            for item in refined:
                p = np.array([[[float(item.refined_center[0]), float(item.refined_center[1])]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(p, m)[0][0]
                x, y = int(mapped[0]), int(mapped[1])
                color = (0, 220, 0) if item.raw_state == "ON" else (0, 170, 255) if item.raw_state == "RETRY" else (90, 90, 90)
                cv2.circle(overlay, (x, y), 7, color, 2)
                cv2.putText(
                    overlay,
                    f"{item.name}:{item.raw_state}",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )
            cv2.putText(
                overlay,
                f"Model: {plate.display_name}  score={result.match_score:.3f}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow("led-beta", to_display(overlay))
        if args.once:
            cv2.waitKey(0)
            break
        key = cv2.waitKey(15) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

