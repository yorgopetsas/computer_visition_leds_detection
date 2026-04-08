from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import sys
from datetime import datetime, timezone
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
        "--preview-scale",
        type=float,
        default=1.0,
        help="Display scale factor for preview window (e.g. 1.25).",
    )
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
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Frames required for stable ON/OFF decision (default 3).",
    )
    parser.add_argument(
        "--retry-margin",
        type=float,
        default=0.02,
        help="Uncertainty band around threshold that yields RETRY (default 0.02).",
    )
    parser.add_argument(
        "--show-led-overlay",
        action="store_true",
        help="Draw LED ROIs and per-LED states on the video frame.",
    )
    parser.add_argument(
        "--expect-led",
        default="",
        help="Expected check in format LED=ON or LED=OFF, e.g. AP1=ON.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional JSONL log file path for snapshot results.",
    )
    parser.add_argument(
        "--log-every-n-frames",
        type=int,
        default=0,
        help="If >0, append one log entry every N frames (no Space key required).",
    )
    parser.add_argument(
        "--guided-sequence",
        action="store_true",
        help="Show operator sidebar and test LEDs one-by-one in mapped order.",
    )
    return parser.parse_args()


def _stable_state(history: deque[str], stable_frames: int) -> str:
    if stable_frames <= 1:
        return history[-1] if history else "RETRY"
    if len(history) < stable_frames:
        return "RETRY"
    tail = list(history)[-stable_frames:]
    if "RETRY" in tail:
        return "RETRY"
    return tail[0] if all(x == tail[0] for x in tail) else "RETRY"


def _apply_temporal_smoothing(result, histories, stable_frames: int):
    if not result.leds:
        return result
    for led in result.leds:
        name = led["name"]
        raw = led.get("raw_state", "ON" if led.get("on") else "OFF")
        histories[name].append(raw)
        state = _stable_state(histories[name], stable_frames)
        led["state"] = state
        led["stable"] = state != "RETRY"
        led["on"] = state == "ON"
    return result


def _draw_led_overlay(overlay: np.ndarray, result) -> None:
    if (
        not result.leds
        or not result.corners
        or len(result.corners) != 4
        or not result.canonical_size
    ):
        return
    # Map canonical LED coordinates to frame coordinates.
    corners = np.array(result.corners, dtype=np.float32)
    width, height = float(result.canonical_size[0]), float(result.canonical_size[1])
    src = np.array(
        [[0.0, 0.0], [width - 1, 0.0], [width - 1, height - 1], [0.0, height - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, corners)
    for led in result.leds:
        center = led.get("center")
        if not center or len(center) != 2:
            continue
        p = np.array([[[float(center[0]), float(center[1])]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(p, m)[0][0]
        x, y = int(mapped[0]), int(mapped[1])
        state = led.get("state", led.get("raw_state", "RETRY"))
        if state == "ON":
            color = (0, 220, 0)
        elif state == "OFF":
            color = (80, 80, 80)
        else:
            color = (0, 170, 255)
        radius = int(max(4, led.get("radius", 8)))
        cv2.circle(overlay, (x, y), radius, color, 2)
        cv2.putText(
            overlay,
            f"{led['name']}:{state} {led['score']:.2f}/{led['threshold']:.2f}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )


def _wrap_lines(text: str, max_chars: int = 34) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def _draw_guided_sidebar(frame: np.ndarray, result, guided_index: int) -> np.ndarray:
    panel_w = 420
    h = frame.shape[0]
    panel = np.full((h, panel_w, 3), (255, 235, 205), dtype=np.uint8)  # light blue-ish
    font_color = (0, 0, 255)  # red text

    y = 40
    cv2.putText(panel, "PCB DETECTED", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2)
    y += 36
    pcb_name = result.plate_name if result.plate_name else "Unknown"
    for line in _wrap_lines(pcb_name, max_chars=24):
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, font_color, 2)
        y += 30

    y += 10
    cv2.putText(panel, "INSTRUCTIONS", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2)
    y += 34

    leds = result.leds or []
    if not leds:
        lines = ["Waiting for LED data..."]
    elif guided_index >= len(leds):
        lines = ["All LEDs tested.", "Sequence complete."]
    else:
        target = leds[guided_index]
        state = target.get("state", target.get("raw_state", "RETRY"))
        if state == "ON":
            lines = [
                f"LED {target['name']} detected ON.",
                "Press ENTER for next LED.",
                f"Score: {target['score']:.3f}",
            ]
        else:
            lines = [
                f"Now test LED {target['name']}.",
                "Turn only this LED ON.",
                f"Current state: {state}",
                f"Score: {target['score']:.3f}",
            ]
    for text in lines:
        for line in _wrap_lines(text):
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, font_color, 2)
            y += 28
        y += 4

    y = h - 80
    cv2.putText(panel, "Press Q or ESC to exit", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
    y += 28
    cv2.putText(panel, "Space: print current status", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
    y += 28
    cv2.putText(panel, "R: reset guided sequence", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
    y += 28
    cv2.putText(panel, "Enter: confirm next LED", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)

    return np.hstack([frame, panel])


def print_result(result) -> None:
    print("=" * 60)
    print(result.message)
    if result.plate_name:
        print(f"Plate: {result.plate_name} ({result.plate_id})")
        print(f"Match score: {result.match_score:.3f}")
    if result.leds:
        for led in result.leds:
            state = led.get("state", led.get("raw_state", "ON" if led["on"] else "OFF"))
            print(
                f"- {led['name']:<22} {state:<3}  score={led['score']:.3f}  thr={led['threshold']:.3f}"
            )


def _parse_expected(value: str) -> tuple[str, str] | None:
    if not value:
        return None
    if "=" not in value:
        raise ValueError("--expect-led must look like LED=ON or LED=OFF")
    led_name, expected = value.split("=", 1)
    led_name = led_name.strip()
    expected = expected.strip().upper()
    if expected not in {"ON", "OFF"}:
        raise ValueError("Expected value must be ON or OFF")
    if not led_name:
        raise ValueError("LED name cannot be empty")
    return led_name, expected


def _evaluate_expectation(result, expected: tuple[str, str] | None) -> tuple[str, str]:
    if expected is None:
        return "N/A", ""
    led_name, wanted = expected
    if not result.leds:
        return "RETRY", f"Check {led_name}={wanted} -> RETRY (no LED data)"
    target = next((x for x in result.leds if x["name"].lower() == led_name.lower()), None)
    if target is None:
        return "RETRY", f"Check {led_name}={wanted} -> RETRY (LED not found)"
    state = target.get("state", target.get("raw_state", "RETRY"))
    if state == "RETRY":
        return "RETRY", f"Check {led_name}={wanted} -> RETRY (uncertain)"
    return ("OK", f"Check {led_name}={wanted} -> OK") if state == wanted else ("NOT_OK", f"Check {led_name}={wanted} -> NOT_OK")


def _log_snapshot(path: str, result, check_status: str, check_message: str) -> None:
    if not path:
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "message": result.message,
        "plate_id": result.plate_id,
        "plate_name": result.plate_name,
        "match_score": result.match_score,
        "check_status": check_status,
        "check_message": check_message,
        "leds": result.leds or [],
    }
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = read_args()
    expected = _parse_expected(args.expect_led)
    detector = LEDPlateDetector(
        (ROOT / args.config_dir).resolve(),
        use_config_corners=args.fixed_corners,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cv2.namedWindow("led-check", cv2.WINDOW_AUTOSIZE)
    scale = max(0.2, float(args.preview_scale))

    def to_display(img):
        if scale == 1.0:
            return img
        return cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    histories = defaultdict(lambda: deque(maxlen=max(1, args.stable_frames)))
    frame_idx = 0
    guided_index = 0

    while True:
        if cv2.getWindowProperty("led-check", cv2.WND_PROP_VISIBLE) < 1:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        result = (
            detector.check_led(frame, args.check_led, retry_margin=args.retry_margin)
            if args.check_led
            else detector.detect(frame, retry_margin=args.retry_margin)
        )
        frame_idx += 1
        smooth_frames = 1 if args.once else max(1, args.stable_frames)
        result = _apply_temporal_smoothing(result, histories, smooth_frames)
        check_status, check_message = _evaluate_expectation(result, expected)

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
        if check_status != "N/A":
            check_color = (0, 220, 0) if check_status == "OK" else (0, 170, 255) if check_status == "RETRY" else (0, 0, 255)
            cv2.putText(
                overlay,
                check_message,
                (12, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                check_color,
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
        if args.show_led_overlay:
            _draw_led_overlay(overlay, result)
        if args.guided_sequence:
            overlay = _draw_guided_sidebar(overlay, result, guided_index)
        cv2.imshow("led-check", to_display(overlay))
        if args.once:
            print_result(result)
            if check_status != "N/A":
                print(check_message)
            _log_snapshot(args.log_file, result, check_status, check_message)
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
            if check_status != "N/A":
                print(check_message)
            _log_snapshot(args.log_file, result, check_status, check_message)
        if args.guided_sequence and key == ord("r"):
            guided_index = 0
        if args.guided_sequence and key in (10, 13):
            if result.leds and guided_index < len(result.leds):
                target = result.leds[guided_index]
                target_state = target.get("state", target.get("raw_state", "RETRY"))
                if target_state == "ON":
                    guided_index += 1
        if args.log_file and args.log_every_n_frames > 0 and (frame_idx % args.log_every_n_frames == 0):
            _log_snapshot(args.log_file, result, check_status, check_message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

