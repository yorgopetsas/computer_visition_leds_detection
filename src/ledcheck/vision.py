from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .models import LEDConfig, PlateConfig


@dataclass
class PlateMatch:
    plate_id: str
    score: float
    warped: np.ndarray


def order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_plate_corners(
    frame: np.ndarray,
    expected_aspect_ratio: float | None = None,
) -> Optional[np.ndarray]:
    candidates = detect_plate_candidates(frame, expected_aspect_ratio=expected_aspect_ratio, max_candidates=1)
    return candidates[0] if candidates else None


def detect_plate_candidates(
    frame: np.ndarray,
    expected_aspect_ratio: float | None = None,
    max_candidates: int = 5,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = (h * w) * 0.02
    expected = expected_aspect_ratio if expected_aspect_ratio else 1.65
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        if aspect < 1.2 or aspect > 2.8:
            continue
        box = cv2.boxPoints(rect).astype(np.float32)
        aspect_penalty = max(0.0, 1.0 - min(1.0, abs(aspect - expected) / expected))
        score = area * (0.5 + 0.5 * aspect_penalty)
        candidates.append((score, box))
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
    return [order_corners(x[1]) for x in ordered[: max(1, max_candidates)]]


def warp_plate(frame: np.ndarray, corners: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def _build_red_mask(roi_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 70], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 70, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


def red_led_score(canonical_bgr: np.ndarray, led: LEDConfig) -> float:
    cx, cy = led.center
    r = int(max(4, led.radius))
    y0 = max(0, cy - r)
    y1 = min(canonical_bgr.shape[0], cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(canonical_bgr.shape[1], cx + r + 1)
    roi = canonical_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    red_mask = _build_red_mask(roi)
    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    center_mean = float(np.mean(v))
    # Local contrast helps reject ambient lighting effects.
    ring_r = int(2.5 * r)
    yr0 = max(0, cy - ring_r)
    yr1 = min(canonical_bgr.shape[0], cy + ring_r + 1)
    xr0 = max(0, cx - ring_r)
    xr1 = min(canonical_bgr.shape[1], cx + ring_r + 1)
    ring = canonical_bgr[yr0:yr1, xr0:xr1]
    ring_v = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)[:, :, 2]
    ring_mean = float(np.mean(ring_v)) if ring_v.size else center_mean
    contrast = max(0.0, (center_mean - ring_mean) / 255.0)
    # Weighted score (0..1+)
    return 0.7 * red_ratio + 0.3 * contrast


def evaluate_leds(
    canonical_bgr: np.ndarray,
    plate: PlateConfig,
    retry_margin: float = 0.02,
) -> list[dict]:
    result = []
    for led in plate.leds:
        score = red_led_score(canonical_bgr, led)
        if score >= (led.threshold + retry_margin):
            raw_state = "ON"
        elif score <= (led.threshold - retry_margin):
            raw_state = "OFF"
        else:
            raw_state = "RETRY"
        result.append(
            {
                "name": led.name,
                "score": round(score, 4),
                "on": raw_state == "ON",
                "raw_state": raw_state,
                "threshold": led.threshold,
                "center": list(led.center),
                "radius": led.radius,
            }
        )
    return result


def template_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    corr = cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)
    return float(corr.max())


def extract_rect(img: np.ndarray, rect_xyxy: list[int]) -> np.ndarray:
    if len(rect_xyxy) != 4:
        return img[0:0, 0:0]
    x0, y0, x1, y1 = [int(v) for v in rect_xyxy]
    x0 = max(0, min(img.shape[1] - 1, x0))
    x1 = max(0, min(img.shape[1], x1))
    y0 = max(0, min(img.shape[0] - 1, y0))
    y1 = max(0, min(img.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return img[0:0, 0:0]
    return img[y0:y1, x0:x1]


def match_plate(
    warped: np.ndarray,
    plate_templates: Iterable[tuple[PlateConfig, np.ndarray]],
) -> Optional[PlateMatch]:
    best: Optional[PlateMatch] = None
    for plate, template in plate_templates:
        score = template_similarity(warped, template)
        if best is None or score > best.score:
            best = PlateMatch(plate_id=plate.plate_id, score=score, warped=warped)
    return best

