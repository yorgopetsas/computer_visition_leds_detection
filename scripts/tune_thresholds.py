from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


EXPECT_RE = re.compile(r"Check\s+(.+?)=(ON|OFF)\s+->", re.IGNORECASE)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LED thresholds from JSONL logs with expected ON/OFF checks."
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to JSONL log file produced by scripts/detect.py --log-file.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/plates",
        help="Directory containing plate config JSON files.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum sample count required per class (ON and OFF), default 6.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.01,
        help="Margin added between OFF and ON medians before midpoint, default 0.01.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply tuned thresholds to plate config JSON files.",
    )
    return parser.parse_args()


def parse_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_expected_led_and_state(check_message: str) -> tuple[str, str] | None:
    m = EXPECT_RE.search(check_message or "")
    if not m:
        return None
    return m.group(1).strip(), m.group(2).upper()


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> None:
    args = read_args()
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    rows = parse_rows(log_path)
    if not rows:
        raise RuntimeError("Log file is empty.")

    # plate_id -> led_name -> {"ON":[scores], "OFF":[scores]}
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"ON": [], "OFF": []})
    )

    for row in rows:
        plate_id = str(row.get("plate_id", "")).strip()
        if not plate_id:
            continue
        parsed = parse_expected_led_and_state(str(row.get("check_message", "")))
        if not parsed:
            continue
        exp_led, exp_state = parsed
        leds = row.get("leds", []) or []
        target = next((x for x in leds if str(x.get("name", "")).lower() == exp_led.lower()), None)
        if not target:
            continue
        score = float(target.get("score", 0.0))
        buckets[plate_id][exp_led][exp_state].append(score)

    config_dir = Path(args.config_dir)
    print("=" * 80)
    print("Threshold tuning report")
    print(f"Log file: {log_path}")
    print(f"Config dir: {config_dir}")
    print("=" * 80)

    any_update = False
    for plate_id in sorted(buckets.keys()):
        cfg_path = config_dir / f"{plate_id}.json"
        if not cfg_path.exists():
            print(f"[SKIP] {plate_id}: config not found ({cfg_path})")
            continue
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        leds_cfg = data.get("leds", [])
        led_cfg_index = {str(x.get("name", "")).lower(): x for x in leds_cfg}

        print(f"\nPlate: {plate_id}")
        plate_updated = False
        for led_name, stats in sorted(buckets[plate_id].items()):
            on_scores = stats["ON"]
            off_scores = stats["OFF"]
            if len(on_scores) < args.min_samples or len(off_scores) < args.min_samples:
                print(
                    f"- {led_name}: insufficient samples "
                    f"(ON={len(on_scores)}, OFF={len(off_scores)}), need >= {args.min_samples}"
                )
                continue
            on_med = median(on_scores)
            off_med = median(off_scores)
            if on_med <= off_med:
                print(
                    f"- {led_name}: overlapping distributions "
                    f"(OFF med={off_med:.4f}, ON med={on_med:.4f}), skipped"
                )
                continue
            low = off_med + args.safety_margin
            high = on_med - args.safety_margin
            tuned = clamp(0.5 * (low + high), 0.01, 0.95)
            item = led_cfg_index.get(led_name.lower())
            if not item:
                print(f"- {led_name}: LED not found in config, skipped")
                continue
            old = float(item.get("threshold", 0.12))
            print(
                f"- {led_name}: OFF med={off_med:.4f}, ON med={on_med:.4f}, "
                f"threshold {old:.4f} -> {tuned:.4f}"
            )
            item["threshold"] = round(tuned, 4)
            any_update = True
            plate_updated = True

        if args.write and plate_updated:
            cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  wrote: {cfg_path}")

    print("\nDone.")
    if not args.write:
        print("Dry run only. Re-run with --write to persist threshold updates.")
    elif not any_update:
        print("No thresholds were updated.")


if __name__ == "__main__":
    main()

