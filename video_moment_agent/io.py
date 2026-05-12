from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Trace, TraceState


def load_trace(path: str | Path) -> Trace:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    states = [
        TraceState(
            t=float(item["t"]),
            state=dict(item.get("state", {})),
            frame=item.get("frame"),
        )
        for item in raw.get("states", [])
    ]
    states.sort(key=lambda item: item.t)
    return Trace(video_id=raw.get("video_id", Path(path).stem), states=states)


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
