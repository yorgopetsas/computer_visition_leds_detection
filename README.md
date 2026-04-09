# LED Plate Checker (Webcam)

Starter implementation for robust LED ON/OFF checks on two known plates:

- `Edel Electrico` (light blue, bottom label `ELECTRICO`)
- `Edel E/S 3VF` (dark blue, bottom label `E/S 3VF`)

The detector is designed for production workflows:

- plate alignment through perspective warp
- per-LED calibration and thresholds
- red LED scoring with local contrast
- command-style single LED check

## Project layout

- `src/ledcheck/`: core detection modules
- `src/ledcheck_beta/`: experimental beta helpers (see [README_beta.md](README_beta.md))
- `scripts/calibrate.py`: interactive calibration utility
- `scripts/detect.py`: live webcam detector
- `scripts/detect_beta.py`: beta live runner (optional)
- `configs/plates/`: generated plate configs and templates

## 1) Setup

```bash
python -m pip install -r requirements.txt
```

## 2) Calibrate each plate

Run once per plate. During calibration:

1. click 4 corners in order (TL, TR, BR, BL)
2. click LED centers in the same order as names passed in `--leds`
3. mark model/name text ROI (top-left then bottom-right)
4. press `Enter` to save

Example for `K2 Electro`:

```bash
python scripts/calibrate.py --plate-id k2_es_electro --display-name "Edel K2 Electro" --label-hint "K2 Electro" --leds "35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,F1,F2,SUBIDA,BAJADA,RAPIDA,LENTA,A1,A2,CP,FT,41,40,39,37,36C,36"
```

Example for `E/S 3VF`:

```bash
python scripts/calibrate.py --plate-id es_3vf --display-name "Edel E/S 3VF" --label-hint "E/S 3VF" --leds "LEVA,M-RS1,RS2,A1,A2,CP"
```

This creates:

- `configs/plates/<plate_id>.json`
- `configs/plates/<plate_id>_template.png`

Useful calibration options:

- `--cam-width 1920 --cam-height 1080` to request HD capture
- `--size 1600x960` to save a higher-resolution warped template for dense LED layouts
- `--camera 1` to force the external USB webcam (instead of laptop camera)
- `Backspace` during corner or LED clicking to undo last point
- Physical metadata can be saved in each plate config:
  - `--plate-width-mm 163 --plate-height-mm 119`
  - `--camera-distance-mm 383 --distance-tolerance-mm 100 --max-tilt-deg 10`
  - `--background-notes "stable daylight, brown background"`

## 3) Run live detection

```bash
python scripts/detect.py
```

To discover camera indexes and pick your external webcam:

```bash
python scripts/detect.py --list-cameras
```

Then run detection with the selected camera, for example:

```bash
python scripts/detect.py --camera 1 --cam-width 1920 --cam-height 1080
```

**Beta (experimental):** local LED refinement and `detect_beta.py` are documented in [README_beta.md](README_beta.md), including the `[beta]` git commit prefix for beta-only changes.

## Camera intrinsics (one-time)

If you want `fx, fy, cx, cy` and lens distortion for more precise geometry in future updates:

1. Print a chessboard pattern (default expected: `9x6` inner corners).
2. Show it to the camera at multiple angles/distances.
3. Capture at least ~20 good frames.

Run:

```bash
python scripts/calibrate_camera.py --camera 1 --cam-width 1920 --cam-height 1080 --board-cols 9 --board-rows 6 --square-mm 20 --min-frames 20 --output "configs/camera_intrinsics.json"
```

The output JSON includes:

- `fx`, `fy`, `cx`, `cy`
- full camera matrix
- distortion coefficients
- RMS reprojection error

Keys:

- `Space`: print current decision to console
- `Q` or `Esc`: quit

Single-shot detection:

```bash
python scripts/detect.py --once
```

The `--once` mode keeps the result window open until a key is pressed, so you can inspect overlays.

For fixed laboratory setups (camera and plate position stable), use fixed-corners mode:

```bash
python scripts/detect.py --fixed-corners --cam-width 1920 --cam-height 1080
```

Interactive startup (recommended for test benches with slight placement drift):

```bash
python scripts/detect.py --fixed-corners --interactive-startup --show-led-overlay --guided-sequence --cam-width 1920 --cam-height 1080 --preview-scale 1.0
```

Interactive flow:

1. Select PCB model in terminal.
2. Click 4 PCB corners in order: top-left, top-right, bottom-right, bottom-left.
3. Press `Enter` to confirm and start guided test.

Corner selection hotkeys:

- `R`: reset selected corners
- `Q` / `Esc`: exit

For production-like behavior with temporal smoothing, uncertainty handling, and live LED ROI feedback:

```bash
python scripts/detect.py --fixed-corners --stable-frames 3 --retry-margin 0.02 --show-led-overlay
```

Guided one-by-one LED test with operator sidebar:

```bash
python scripts/detect.py --fixed-corners --show-led-overlay --guided-sequence --cam-width 1920 --cam-height 1080
```

Notes:

- `--stable-frames`: requires N consecutive frames before ON/OFF is accepted
- `--retry-margin`: defines an uncertainty band around each LED threshold
- `--show-led-overlay`: draws LED circles and `name:state score/threshold` on video
- `--guided-sequence`: sidebar instructions, auto-advance when current LED turns ON
- `--preview-scale`: scales display without changing detection geometry (e.g. `1.2`)
- guided mode hotkeys: `Enter` confirms next LED after ON detection, `R` resets sequence, `Space` prints status, `Q`/`Esc` exits

Command-style pass/fail check:

```bash
python scripts/detect.py --fixed-corners --expect-led "35=ON" --stable-frames 3 --retry-margin 0.02 --once
```

Optional snapshot logging (JSONL):

```bash
python scripts/detect.py --fixed-corners --expect-led "35=ON" --log-file "logs/checks.jsonl" --log-every-n-frames 30
```

Press `Space` to print and append one snapshot entry, or use `--log-every-n-frames` for automatic logging.

## 4) Validation checklist (lab)

Run:

```bash
python scripts/detect.py --fixed-corners --expect-led "35=ON" --stable-frames 3 --retry-margin 0.02 --show-led-overlay --cam-width 1920 --cam-height 1080
```

Test sequence:

1. Keep `35` OFF -> expect `NOT_OK` or `RETRY`
2. Turn `35` ON -> expect `OK`
3. Toggle quickly -> should briefly show `RETRY` before stabilizing

Single-shot check:

```bash
python scripts/detect.py --once --fixed-corners --expect-led "35=ON"
```

Logging check:

```bash
python scripts/detect.py --fixed-corners --expect-led "35=ON" --log-file "logs/checks.jsonl" --log-every-n-frames 30
```

Verify each line in `logs/checks.jsonl` includes timestamp, plate data, `check_status`, and LED scores.

Tuning guidance:

- Faster response: `--stable-frames 2`
- More robust response: `--stable-frames 4`
- Fewer `RETRY`: `--retry-margin 0.01`
- Safer uncertainty zone: `--retry-margin 0.03`

## 5) Calibrate and validate `E/S 3VF`

1. Calibrate second plate:

```bash
python scripts/calibrate.py --plate-id es_3vf --display-name "Edel E/S 3VF" --label-hint "E/S 3VF" --size 1600x960 --cam-width 1920 --cam-height 1080 --leds "LEVA,M-RS1,RS2,A1,A2,CP"
```

2. Run same live validation style (replace expected LED as needed):

```bash
python scripts/detect.py --fixed-corners --expect-led "LEVA=ON" --stable-frames 3 --retry-margin 0.02 --show-led-overlay --log-file "logs/checks.jsonl"
```

3. Collect samples for **both** plates in the same `logs/checks.jsonl`.

## 6) Unify acceptance criteria for both plates

Run acceptance report:

```bash
python scripts/validate_acceptance.py --log-file "logs/checks.jsonl" --min-ok-rate 0.95 --max-retry-rate 0.10
```

Suggested initial production targets:

- per-plate OK rate >= `95%`
- per-plate RETRY rate <= `10%`

When both plates pass repeatedly, keep those values as baseline and tighten over time.

## 7) Auto-tune LED thresholds from logs

After collecting ON/OFF expected checks in `logs/checks.jsonl`, run a dry run:

```bash
python scripts/tune_thresholds.py --log-file "logs/checks.jsonl" --config-dir "configs/plates" --min-samples 6 --safety-margin 0.01
```

If values look good, write tuned thresholds into config files:

```bash
python scripts/tune_thresholds.py --log-file "logs/checks.jsonl" --config-dir "configs/plates" --min-samples 6 --safety-margin 0.01 --write
```

Recommended test flow for this phase:

1. Run detection for a target LED with expected OFF and collect logs.
2. Run detection for the same LED with expected ON and collect logs.
3. Repeat enough times (at least 6 ON and 6 OFF samples per LED).
4. Run tuner dry run and inspect suggested threshold changes.
5. Apply with `--write`, then re-run validation checks.

## 8) Simple operator command

Use `run_check.py` for a simpler one-command workflow.

Single-shot check (default):

```bash
python scripts/run_check.py --led "35" --expected ON
```

Live mode:

```bash
python scripts/run_check.py --led "35" --expected ON --live --show-overlay --log-every-n-frames 30
```

Notes:

- fixed-corners mode is enabled by default
- disable with `--no-fixed-corners` if needed
- logs go to `logs/checks.jsonl` by default (override with `--log-file`)

Check one LED by name:

```bash
python scripts/detect.py --check-led "AP1"
```

## Notes for robust production use

- Keep `auto exposure` fixed if camera supports it.
- Recalibrate if camera/lens/focus changes.
- Tune per-LED `threshold` values in each plate JSON after collecting real samples.
- Add temporal smoothing (e.g. majority vote over 3 to 5 frames) in your final QA profile.

