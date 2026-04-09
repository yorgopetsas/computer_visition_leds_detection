# Beta pipeline — LED / PCB detection (experimental)

This document describes the **beta** branch of functionality. It is separate from the stable production flow in `README.md` and `scripts/detect.py`.

Stable behavior is unchanged when you use only `detect.py` and `src/ledcheck/`.

---

## What “beta” includes today

| Area | Path | Role |
|------|------|------|
| Local LED refinement | `src/ledcheck_beta/refine.py` | Searches a small window around each calibrated LED center in canonical space and picks the best red-score position. |
| Beta runner | `scripts/detect_beta.py` | Live camera: optional interactive model + 4-corner startup, then warped view + refined LED overlays. |

Beta reuses stable pieces:

- `LEDPlateDetector` from `src/ledcheck/detector.py` (warp, plate match, configs)
- Plate JSON maps under `configs/plates/<plate_id>.json` (LED centers from calibration)

---

## How to run

From repo root, with dependencies installed (`pip install -r requirements.txt`):

```bash
python scripts/detect_beta.py --fixed-corners --interactive-startup --cam-width 1920 --cam-height 1080 --preview-scale 1.0
```

Flow:

1. Terminal: choose PCB model by number.
2. Window: click corners **TL → TR → BR → BL**, then **Enter**.
3. Live view: refined LED positions and states (ON / OFF / RETRY by score).

Useful flags:

| Flag | Meaning |
|------|---------|
| `--search-radius 16` | Pixels to search around each mapped LED center (default 16). |
| `--retry-margin 0.02` | Same uncertainty band as stable detector. |
| `--once` | Single frame then exit. |

---

## Planned / not in beta yet

- Text/OCR anchoring using labels printed beside LEDs (requires ROI or line layout per board).
- Optional integration with vision-language or detector models (e.g. small ONNX/YOLO for LED dots) — kept behind beta interfaces so stable code stays dependency-light.

---

## Git commits: convention for beta work

**Any commit that only changes beta code, beta docs, or experimental tooling under beta paths should use this subject line prefix:**

```text
[beta] <short description>
```

Examples:

- `[beta] Add OCR stub for label anchoring`
- `[beta] Tune search radius default in refine_leds_local`
- `[beta] Document detect_beta flags in README_beta.md`

**Do not** use `[beta]` for commits that only touch stable `detect.py`, production calibration, or core `ledcheck` unless the message also covers beta (then split into two commits or use a clear body: “Stable: … / Beta: …”).

Paths that are considered **beta-owned** (prefix `[beta]` when they are the only or main change):

- `src/ledcheck_beta/`
- `scripts/detect_beta.py`
- `README_beta.md`
- Optional future: `tests/test_beta_*.py`, `docs/beta/`

---

## Relationship to stable README

- **Stable:** `README.md` — calibration, `detect.py`, logging, guided mode, acceptance scripts.
- **Beta:** this file — `detect_beta.py`, `ledcheck_beta`, experiments.

When beta features are production-ready, they are merged into stable with **normal** commit messages (no `[beta]`).
