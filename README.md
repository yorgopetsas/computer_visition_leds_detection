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

Check one LED by name:

```bash
python scripts/detect.py --check-led "AP1"
```

## Notes for robust production use

- Keep `auto exposure` fixed if camera supports it.
- Recalibrate if camera/lens/focus changes.
- Tune per-LED `threshold` values in each plate JSON after collecting real samples.
- Add temporal smoothing (e.g. majority vote over 3 to 5 frames) in your final QA profile.

