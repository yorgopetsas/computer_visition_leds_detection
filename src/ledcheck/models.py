from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field


Point = Tuple[int, int]


class LEDConfig(BaseModel):
    name: str
    center: Point
    radius: int = 8
    threshold: float = 0.12


class PlateConfig(BaseModel):
    plate_id: str
    display_name: str
    # Label hint used as a deterministic tie-break between templates.
    bottom_label_hint: str = ""
    canonical_size: Tuple[int, int] = (900, 540)
    corners: List[Point] = Field(default_factory=list)
    leds: List[LEDConfig] = Field(default_factory=list)
    template_image: str = ""
    confidence_threshold: float = 0.22

