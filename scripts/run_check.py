from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operator-friendly wrapper around scripts/detect.py."
    )
    parser.add_argument(
        "--led",
        required=True,
        help="LED name to check, e.g. 35 or LEVA.",
    )
    parser.add_argument(
        "--expected",
        required=True,
        choices=["ON", "OFF", "on", "off"],
        help="Expected LED state.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--cam-width", type=int, default=1920, help="Camera width.")
    parser.add_argument("--cam-height", type=int, default=1080, help="Camera height.")
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Frames required for stable ON/OFF decision.",
    )
    parser.add_argument(
        "--retry-margin",
        type=float,
        default=0.02,
        help="Uncertainty band around threshold.",
    )
    parser.add_argument(
        "--no-fixed-corners",
        action="store_true",
        help="Disable fixed-corners mode and use automatic plate detection.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Keep running live mode. By default, runs one-shot mode.",
    )
    parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Show per-LED ROI overlays.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/checks.jsonl",
        help="JSONL log output path (default logs/checks.jsonl).",
    )
    parser.add_argument(
        "--log-every-n-frames",
        type=int,
        default=0,
        help="If >0 in live mode, append a log every N frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = read_args()
    expected = args.expected.upper()
    detect_script = ROOT / "scripts" / "detect.py"
    cmd = [
        sys.executable,
        str(detect_script),
        "--camera",
        str(args.camera),
        "--cam-width",
        str(args.cam_width),
        "--cam-height",
        str(args.cam_height),
        "--stable-frames",
        str(args.stable_frames),
        "--retry-margin",
        str(args.retry_margin),
        "--expect-led",
        f"{args.led}={expected}",
        "--log-file",
        args.log_file,
    ]
    if not args.no_fixed_corners:
        cmd.append("--fixed-corners")
    if args.show_overlay:
        cmd.append("--show-led-overlay")
    if args.live:
        if args.log_every_n_frames > 0:
            cmd.extend(["--log-every-n-frames", str(args.log_every_n_frames)])
    else:
        cmd.append("--once")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

