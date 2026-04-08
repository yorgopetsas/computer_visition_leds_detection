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


def detect_plate_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = (h * w) * 0.05
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        candidates.append((area, quad))
    if not candidates:
        return None
    # Choose the largest 4-point contour.
    quad = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    return order_corners(quad)


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


def evaluate_leds(canonical_bgr: np.ndarray, plate: PlateConfig) -> list[dict]:
    result = []
    for led in plate.leds:
        score = red_led_score(canonical_bgr, led)
        result.append(
            {
                "name": led.name,
                "score": round(score, 4),
                "on": bool(score >= led.threshold),
                "threshold": led.threshold,
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

