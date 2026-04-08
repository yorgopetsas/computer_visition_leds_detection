from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .io_utils import load_plate_configs
from .models import PlateConfig
from .vision import detect_plate_corners, evaluate_leds, match_plate, warp_plate


@dataclass
class DetectionResult:
    ok: bool
    message: str
    plate_id: str = ""
    plate_name: str = ""
    match_score: float = 0.0
    leds: list[dict] | None = None


class LEDPlateDetector:
    def __init__(self, config_dir: Path):
        self.configs: Dict[str, PlateConfig] = load_plate_configs(config_dir)
        self.templates: Dict[str, np.ndarray] = {}
        for plate in self.configs.values():
            template_path = Path(plate.template_image)
            if not template_path.exists():
                continue
            img = cv2.imread(str(template_path))
            if img is None:
                continue
            self.templates[plate.plate_id] = cv2.resize(img, plate.canonical_size)

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        corners = detect_plate_corners(frame_bgr)
        if corners is None:
            return DetectionResult(ok=False, message="Plate not found in frame.")

        # If templates exist, try match all plate types by warping to each canonical size.
        best_plate: Optional[PlateConfig] = None
        best_score = -1.0
        best_warp: Optional[np.ndarray] = None

        for plate in self.configs.values():
            warped = warp_plate(frame_bgr, corners, plate.canonical_size)
            template = self.templates.get(plate.plate_id)
            if template is None:
                # Config exists but no template yet. Keep first as fallback.
                if best_plate is None:
                    best_plate = plate
                    best_warp = warped
                    best_score = 0.0
                continue
            matched = match_plate(warped, [(plate, template)])
            if matched and matched.score > best_score:
                best_score = matched.score
                best_plate = plate
                best_warp = warped

        if best_plate is None or best_warp is None:
            return DetectionResult(ok=False, message="No valid plate configuration.")

        if self.templates and best_score < best_plate.confidence_threshold:
            return DetectionResult(
                ok=False,
                message=f"Plate detected but uncertain type (score={best_score:.3f}).",
                match_score=best_score,
            )

        leds = evaluate_leds(best_warp, best_plate)
        return DetectionResult(
            ok=True,
            message="Detection successful.",
            plate_id=best_plate.plate_id,
            plate_name=best_plate.display_name,
            match_score=max(0.0, best_score),
            leds=leds,
        )

    def check_led(self, frame_bgr: np.ndarray, led_name: str) -> DetectionResult:
        result = self.detect(frame_bgr)
        if not result.ok or not result.leds:
            return result
        led_name_norm = led_name.strip().lower()
        target = next((x for x in result.leds if x["name"].lower() == led_name_norm), None)
        if target is None:
            return DetectionResult(
                ok=False,
                message=f"LED '{led_name}' not found in detected plate '{result.plate_name}'.",
                plate_id=result.plate_id,
                plate_name=result.plate_name,
                match_score=result.match_score,
                leds=result.leds,
            )
        state = "ON" if target["on"] else "OFF"
        return DetectionResult(
            ok=True,
            message=f"LED '{target['name']}' is {state}.",
            plate_id=result.plate_id,
            plate_name=result.plate_name,
            match_score=result.match_score,
            leds=[target],
        )

