from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ledcheck.models import LEDConfig, PlateConfig
from ledcheck.vision import red_led_score


@dataclass
class RefinedLED:
    name: str
    base_center: tuple[int, int]
    refined_center: tuple[int, int]
    threshold: float
    score: float
    raw_state: str
    on: bool


def _state_from_score(score: float, threshold: float, retry_margin: float) -> str:
    if score >= threshold + retry_margin:
        return "ON"
    if score <= threshold - retry_margin:
        return "OFF"
    return "RETRY"


def refine_leds_local(
    canonical_bgr: np.ndarray,
    plate: PlateConfig,
    search_radius_px: int = 16,
    retry_margin: float = 0.02,
) -> list[RefinedLED]:
    refined: list[RefinedLED] = []
    h, w = canonical_bgr.shape[:2]

    for led in plate.leds:
        cx, cy = int(led.center[0]), int(led.center[1])
        best_score = -1.0
        best_xy = (cx, cy)
        r = max(4, int(led.radius))

        x0 = max(r, cx - search_radius_px)
        x1 = min(w - r - 1, cx + search_radius_px)
        y0 = max(r, cy - search_radius_px)
        y1 = min(h - r - 1, cy + search_radius_px)

        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                probe = LEDConfig(name=led.name, center=(xx, yy), radius=r, threshold=led.threshold)
                score = red_led_score(canonical_bgr, probe)
                if score > best_score:
                    best_score = score
                    best_xy = (xx, yy)

        state = _state_from_score(best_score, led.threshold, retry_margin)
        refined.append(
            RefinedLED(
                name=led.name,
                base_center=(cx, cy),
                refined_center=best_xy,
                threshold=float(led.threshold),
                score=float(best_score),
                raw_state=state,
                on=state == "ON",
            )
        )

    return refined

