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

from ledcheck.io_utils import save_plate_config
from ledcheck.models import LEDConfig, PlateConfig
from ledcheck.vision import order_corners, warp_plate


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate plate corners and LED centers.")
    parser.add_argument("--plate-id", required=True, help="Unique plate id, e.g. electrico")
    parser.add_argument("--display-name", required=True, help="Human readable plate name.")
    parser.add_argument(
        "--label-hint",
        default="",
        help="Bottom text hint (e.g. ELECTRICO or E/S 3VF).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument("--cam-width", type=int, default=1920, help="Requested camera width.")
    parser.add_argument("--cam-height", type=int, default=1080, help="Requested camera height.")
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Display scale factor for calibration windows (e.g. 1.25).",
    )
    parser.add_argument(
        "--output",
        default="configs/plates",
        help="Output directory for plate config JSON.",
    )
    parser.add_argument(
        "--size",
        default="1600x960",
        help="Canonical warped size WxH, default 1600x960.",
    )
    parser.add_argument(
        "--leds",
        default="",
        help="Comma-separated LED names in click order, e.g. POWER,ALARM,RUN.",
    )
    return parser.parse_args()


def main() -> None:
    args = read_args()
    size_tokens = args.size.lower().split("x")
    canonical_size = (int(size_tokens[0]), int(size_tokens[1]))
    led_names = [x.strip() for x in args.leds.split(",") if x.strip()]
    if len(set(led_names)) != len(led_names):
        print("Warning: duplicate LED names detected. Single LED checks may be ambiguous.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    # Many USB webcams on Windows need MJPG to deliver true 1080p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution requested: {args.cam_width}x{args.cam_height}")
    print(f"Camera resolution actual:    {actual_w}x{actual_h}")
    if actual_w < args.cam_width or actual_h < args.cam_height:
        print("Warning: webcam did not apply full requested resolution.")

    corners = []
    frame_ref = {"frame": None}

    scale = max(0.2, float(args.preview_scale))

    def to_display(img):
        if scale == 1.0:
            return img
        return cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    def to_original_point(x: int, y: int) -> tuple[int, int]:
        if scale == 1.0:
            return x, y
        return int(round(x / scale)), int(round(y / scale))

    def on_corners(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append(to_original_point(x, y))

    cv2.namedWindow("calibrate-corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calibrate-corners", on_corners)
    print("Click 4 plate corners in this order: top-left, top-right, bottom-right, bottom-left.")
    print("Press ENTER when done, BACKSPACE to undo last point, or ESC to cancel.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_ref["frame"] = frame.copy()
        display = frame.copy()
        for i, p in enumerate(corners):
            cv2.circle(display, p, 5, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(i + 1),
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if len(corners) == 4:
            cv2.polylines(display, [np.array(corners, dtype=np.int32)], True, (0, 255, 255), 2)
        cv2.imshow("calibrate-corners", to_display(display))
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 8 and corners:
            corners.pop()
        if key in (10, 13) and len(corners) == 4:
            break

    ordered = order_corners(np.array(corners, dtype=np.float32))
    warped = warp_plate(frame_ref["frame"], ordered, canonical_size)

    led_points = []

    def on_leds(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            led_points.append(to_original_point(x, y))

    cv2.namedWindow("calibrate-leds", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calibrate-leds", on_leds)
    print("Now click LED centers on the warped plate image.")
    if led_names:
        print(f"Expected order ({len(led_names)}): {', '.join(led_names)}")
    print("Press ENTER when done, BACKSPACE to undo last LED.")

    while True:
        view = warped.copy()
        for i, p in enumerate(led_points):
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            label = led_names[i] if i < len(led_names) else f"LED_{i+1}"
            cv2.putText(view, label, (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("calibrate-leds", to_display(view))
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 8 and led_points:
            led_points.pop()
        if key in (10, 13) and led_points:
            break

    leds = []
    for idx, center in enumerate(led_points):
        name = led_names[idx] if idx < len(led_names) else f"LED_{idx+1}"
        leds.append(LEDConfig(name=name, center=center))

    label_points = []

    def on_label_roi(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(label_points) < 2:
            label_points.append(to_original_point(x, y))

    cv2.namedWindow("calibrate-label-roi", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calibrate-label-roi", on_label_roi)
    print("Mark model/name label ROI on warped image: click top-left then bottom-right.")
    print("Press ENTER when done, BACKSPACE to undo last point.")

    while True:
        view = warped.copy()
        for i, p in enumerate(label_points):
            cv2.circle(view, p, 5, (255, 0, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if len(label_points) == 2:
            p1, p2 = label_points
            x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
            cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.imshow("calibrate-label-roi", to_display(view))
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 8 and label_points:
            label_points.pop()
        if key in (10, 13) and len(label_points) == 2:
            break

    p1, p2 = label_points
    lx0, ly0 = min(p1[0], p2[0]), min(p1[1], p2[1])
    lx1, ly1 = max(p1[0], p2[0]), max(p1[1], p2[1])
    label_roi = [int(lx0), int(ly0), int(lx1), int(ly1)]
    label_crop = warped[ly0:ly1, lx0:lx1].copy()

    output_dir = (ROOT / args.output).resolve()
    config_path = output_dir / f"{args.plate_id}.json"
    template_path = output_dir / f"{args.plate_id}_template.png"
    label_template_path = output_dir / f"{args.plate_id}_label_template.png"
    cv2.imwrite(str(template_path), warped)
    if label_crop.size > 0:
        cv2.imwrite(str(label_template_path), label_crop)

    plate = PlateConfig(
        plate_id=args.plate_id,
        display_name=args.display_name,
        bottom_label_hint=args.label_hint,
        canonical_size=canonical_size,
        corners=[(int(x), int(y)) for x, y in ordered.tolist()],
        leds=leds,
        label_roi=label_roi,
        label_template_image=str(label_template_path),
        template_image=str(template_path),
    )
    save_plate_config(plate, config_path)
    print(f"Saved config: {config_path}")
    print(f"Saved template image: {template_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

