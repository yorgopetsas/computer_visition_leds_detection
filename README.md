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
- `scripts/calibrate.py`: interactive calibration utility
- `scripts/detect.py`: live webcam detector
- `configs/plates/`: generated plate configs and templates

## 1) Setup

```bash
python -m pip install -r requirements.txt
```

## 2) Calibrate each plate

Run once per plate. During calibration:

1. click 4 corners in order (TL, TR, BR, BL)
2. click LED centers in the same order as names passed in `--leds`
3. press `Enter` to save

Example for `Electrico`:

```bash
python scripts/calibrate.py --plate-id electrico --display-name "Edel Electrico" --label-hint "ELECTRICO" --leds "LEVA,AP1,AP2,CP"
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
- `Backspace` during corner or LED clicking to undo last point

## 3) Run live detection

```bash
python scripts/detect.py
```

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

For production-like behavior with temporal smoothing, uncertainty handling, and live LED ROI feedback:

```bash
python scripts/detect.py --fixed-corners --stable-frames 3 --retry-margin 0.02 --show-led-overlay
```

Notes:

- `--stable-frames`: requires N consecutive frames before ON/OFF is accepted
- `--retry-margin`: defines an uncertainty band around each LED threshold
- `--show-led-overlay`: draws LED circles and `name:state score/threshold` on video

Command-style pass/fail check:

```bash
python scripts/detect.py --fixed-corners --expect-led "F2=ON" --stable-frames 3 --retry-margin 0.02 --once
```

Optional snapshot logging (JSONL):

```bash
python scripts/detect.py --fixed-corners --expect-led "F2=ON" --log-file "logs/checks.jsonl"
```

Press `Space` to print and append one snapshot entry.

## 4) Validation checklist (lab)

Run:

```bash
python scripts/detect.py --fixed-corners --expect-led "F2=ON" --stable-frames 3 --retry-margin 0.02 --show-led-overlay --cam-width 1920 --cam-height 1080
```

Test sequence:

1. Keep `F2` OFF -> expect `NOT_OK` or `RETRY`
2. Turn `F2` ON -> expect `OK`
3. Toggle quickly -> should briefly show `RETRY` before stabilizing

Single-shot check:

```bash
python scripts/detect.py --once --fixed-corners --expect-led "F2=ON"
```

Logging check:

```bash
python scripts/detect.py --fixed-corners --expect-led "F2=ON" --log-file "logs/checks.jsonl"
```

Press `Space` multiple times and verify each line in `logs/checks.jsonl` includes timestamp, plate data, `check_status`, and LED scores.

Tuning guidance:

- Faster response: `--stable-frames 2`
- More robust response: `--stable-frames 4`
- Fewer `RETRY`: `--retry-margin 0.01`
- Safer uncertainty zone: `--retry-margin 0.03`

Check one LED by name:

```bash
python scripts/detect.py --check-led "AP1"
```

## Notes for robust production use

- Keep `auto exposure` fixed if camera supports it.
- Recalibrate if camera/lens/focus changes.
- Tune per-LED `threshold` values in each plate JSON after collecting real samples.
- Add temporal smoothing (e.g. majority vote over 3 to 5 frames) in your final QA profile.

