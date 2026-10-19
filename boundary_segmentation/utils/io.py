"""Simple IO helpers."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def list_files(data_dir: str, pattern: str) -> List[str]:
    return sorted(glob(os.path.join(data_dir, pattern)))
