from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate acceptance criteria from JSONL check logs."
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to JSONL log created by scripts/detect.py --log-file.",
    )
    parser.add_argument(
        "--min-ok-rate",
        type=float,
        default=0.95,
        help="Minimum OK rate per plate (default 0.95).",
    )
    parser.add_argument(
        "--max-retry-rate",
        type=float,
        default=0.10,
        help="Maximum RETRY rate per plate (default 0.10).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main() -> None:
    args = read_args()
    path = Path(args.log_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Log file not found: {path}\n"
            "Hint: use detect with --log-file and either press Space, use --once, "
            "or set --log-every-n-frames > 0."
        )

    rows = load_rows(path)
    if not rows:
        raise RuntimeError("Log file is empty.")

    per_plate: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "not_ok": 0, "retry": 0}
    )
    for row in rows:
        plate = row.get("plate_id", "unknown")
        status = str(row.get("check_status", "N/A")).upper()
        per_plate[plate]["total"] += 1
        if status == "OK":
            per_plate[plate]["ok"] += 1
        elif status == "NOT_OK":
            per_plate[plate]["not_ok"] += 1
        elif status == "RETRY":
            per_plate[plate]["retry"] += 1

    print("=" * 72)
    print("Acceptance report")
    print(f"Log file: {path}")
    print(f"Rules: min_ok_rate={args.min_ok_rate:.2f}, max_retry_rate={args.max_retry_rate:.2f}")
    print("=" * 72)

    all_pass = True
    for plate, stats in sorted(per_plate.items()):
        total = max(1, stats["total"])
        ok_rate = stats["ok"] / total
        retry_rate = stats["retry"] / total
        passed = ok_rate >= args.min_ok_rate and retry_rate <= args.max_retry_rate
        all_pass = all_pass and passed
        print(
            f"{plate:<14} total={total:<4} OK={stats['ok']:<4} NOT_OK={stats['not_ok']:<4} "
            f"RETRY={stats['retry']:<4} ok_rate={pct(ok_rate):<7} retry_rate={pct(retry_rate):<7} "
            f"PASS={'YES' if passed else 'NO'}"
        )

    print("=" * 72)
    print("OVERALL PASS" if all_pass else "OVERALL FAIL")


if __name__ == "__main__":
    main()

