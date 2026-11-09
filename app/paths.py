from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(
    os.getenv("LECTIO_DATA_DIR")
    or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    or DEFAULT_DATA_DIR
).expanduser().resolve()
