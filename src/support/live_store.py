"""Small atomic local store. Corrupt state fails closed instead of erasing history."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class LiveStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None

    def load(self):
        if self.path is None or not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported live history schema")
        for key in ("owners", "keys"):
            if not isinstance(data.get(key), list):
                raise ValueError(f"Invalid live history: {key}")
        for key in ("conversations", "requests", "comparisons", "queue"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"Invalid live history: {key}")
        if not isinstance(data.get("input_states", {}), dict):
            raise ValueError("Invalid live history: input_states")
        if any(not isinstance(value, dict) for value in data.get("input_states", {}).values()):
            raise ValueError("Invalid live history: input state")
        return data

    def save(self, data):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
