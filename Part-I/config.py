import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_CFG = {
    "episodes": 800,
    "alpha": 0.2,
    "gamma": 0.95,
    "epsilonStart": 1.0,
    "epsilonEnd": 0.05,
    "epsilonDecayEpisodes": 700,
    "maxStepsPerEpisode": 400,
    "fpsVisual": 30,
    "fpsFast": 2400,
    "tileSize": 48,
    "seed": 42,
    "intrinsicScale": 0.3,
}


def load_config(level_id: int = 0) -> Dict[str, Any]:
    cfg = DEFAULT_CFG.copy()
    base_dir = Path(__file__).parent
    path = base_dir / f"config_level{level_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
        print(f"Loaded config_level{level_id}.json")
    return cfg
