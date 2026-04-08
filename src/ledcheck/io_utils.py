from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .models import PlateConfig


def load_plate_configs(config_dir: Path) -> Dict[str, PlateConfig]:
    configs: Dict[str, PlateConfig] = {}
    for path in sorted(config_dir.glob("*.json")):
        if path.name.endswith(".sample.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        plate = PlateConfig.model_validate(data)
        configs[plate.plate_id] = plate
    if not configs:
        raise FileNotFoundError(
            f"No plate config files found in {config_dir}. Run calibration first."
        )
    return configs


def save_plate_config(config: PlateConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.model_dump(), indent=2),
        encoding="utf-8",
    )

