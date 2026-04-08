from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .io_utils import load_plate_configs
from .models import PlateConfig
from .vision import detect_plate_corners, evaluate_leds, extract_rect, template_similarity, warp_plate


@dataclass
class DetectionResult:
    ok: bool
    message: str
    plate_id: str = ""
    plate_name: str = ""
    match_score: float = 0.0
    leds: list[dict] | None = None
    corners: list[tuple[int, int]] | None = None
    canonical_size: tuple[int, int] | None = None


class LEDPlateDetector:
    def __init__(self, config_dir: Path, use_config_corners: bool = False):
        self.configs: Dict[str, PlateConfig] = load_plate_configs(config_dir)
        self.use_config_corners = use_config_corners
        self.templates: Dict[str, np.ndarray] = {}
        self.label_templates: Dict[str, np.ndarray] = {}
        self.expected_aspect_ratio = 1.65
        if self.configs:
            ratios = []
            for p in self.configs.values():
                w, h = p.canonical_size
                if h > 0:
                    ratios.append(max(w, h) / min(w, h))
            if ratios:
                self.expected_aspect_ratio = float(np.median(np.array(ratios)))
        for plate in self.configs.values():
            template_path = Path(plate.template_image)
            if not template_path.exists():
                continue
            img = cv2.imread(str(template_path))
            if img is None:
                continue
            self.templates[plate.plate_id] = cv2.resize(img, plate.canonical_size)
            label_path = Path(plate.label_template_image) if plate.label_template_image else Path("")
            if label_path.exists():
                label_img = cv2.imread(str(label_path))
                if label_img is not None:
                    self.label_templates[plate.plate_id] = label_img

    def _score_plate_match(self, warped: np.ndarray, plate: PlateConfig) -> float:
        full_template = self.templates.get(plate.plate_id)
        full_score = template_similarity(warped, full_template) if full_template is not None else 0.0
        label_score = 0.0
        label_template = self.label_templates.get(plate.plate_id)
        if label_template is not None and len(plate.label_roi) == 4:
            roi_img = extract_rect(warped, plate.label_roi)
            if roi_img.size > 0:
                label_score = template_similarity(roi_img, label_template)
        # Give label ROI a strong weight for plate identity certainty.
        if label_template is not None:
            return 0.45 * full_score + 0.55 * label_score
        return full_score

    def detect(self, frame_bgr: np.ndarray, retry_margin: float = 0.02) -> DetectionResult:
        # Fixed-corners mode is useful when camera and plate placement are static.
        if self.use_config_corners:
            best_plate: Optional[PlateConfig] = None
            best_score = -1.0
            best_warp: Optional[np.ndarray] = None
            best_corners: Optional[np.ndarray] = None
            for plate in self.configs.values():
                if len(plate.corners) != 4:
                    continue
                corners = np.array(plate.corners, dtype=np.float32)
                warped = warp_plate(frame_bgr, corners, plate.canonical_size)
                score = self._score_plate_match(warped, plate)
                if best_plate is None or score > best_score:
                    best_plate = plate
                    best_score = score
                    best_warp = warped
                    best_corners = corners
            if best_plate is None or best_warp is None or best_corners is None:
                return DetectionResult(ok=False, message="No valid fixed corners in config.")
            corners_list = [(int(x), int(y)) for x, y in best_corners.tolist()]
            leds = evaluate_leds(best_warp, best_plate, retry_margin=retry_margin)
            return DetectionResult(
                ok=True,
                message="Detection successful (fixed corners mode).",
                plate_id=best_plate.plate_id,
                plate_name=best_plate.display_name,
                match_score=max(0.0, best_score),
                leds=leds,
                corners=corners_list,
                canonical_size=best_plate.canonical_size,
            )

        corners = detect_plate_corners(
            frame_bgr,
            expected_aspect_ratio=self.expected_aspect_ratio,
        )
        if corners is None:
            return DetectionResult(ok=False, message="Plate not found in frame.")
        corners_list = [(int(x), int(y)) for x, y in corners.tolist()]

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
            score = self._score_plate_match(warped, plate)
            if score > best_score:
                best_score = score
                best_plate = plate
                best_warp = warped

        if best_plate is None or best_warp is None:
            return DetectionResult(ok=False, message="No valid plate configuration.")

        if self.templates and best_score < best_plate.confidence_threshold:
            return DetectionResult(
                ok=False,
                message=f"Plate detected but uncertain type (score={best_score:.3f}).",
                match_score=best_score,
                corners=corners_list,
                canonical_size=best_plate.canonical_size,
            )

        leds = evaluate_leds(best_warp, best_plate, retry_margin=retry_margin)
        return DetectionResult(
            ok=True,
            message="Detection successful.",
            plate_id=best_plate.plate_id,
            plate_name=best_plate.display_name,
            match_score=max(0.0, best_score),
            leds=leds,
            corners=corners_list,
            canonical_size=best_plate.canonical_size,
        )

    def check_led(
        self,
        frame_bgr: np.ndarray,
        led_name: str,
        retry_margin: float = 0.02,
    ) -> DetectionResult:
        result = self.detect(frame_bgr, retry_margin=retry_margin)
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
                corners=result.corners,
                canonical_size=result.canonical_size,
            )
        state = target.get("raw_state", "ON" if target["on"] else "OFF")
        return DetectionResult(
            ok=True,
            message=f"LED '{target['name']}' is {state}.",
            plate_id=result.plate_id,
            plate_name=result.plate_name,
            match_score=result.match_score,
            leds=[target],
            corners=result.corners,
            canonical_size=result.canonical_size,
        )

