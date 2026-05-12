from __future__ import annotations

import json
import subprocess
from pathlib import Path


def scaffold_video_trace(
    video_path: str | Path,
    output_json: str | Path,
    *,
    fps: float = 1.0,
) -> dict:
    """Extract frames with ffmpeg and create an empty state trace JSON.

    The generated trace is the handoff point for state grounding. A visual model
    can fill each item's `state` field using the corresponding `frame` image.
    """

    video = Path(video_path)
    output = Path(output_json)
    frames_dir = output.with_suffix("").parent / f"{output.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    pattern = frames_dir / "frame_%06d.jpg"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            str(pattern),
        ]
    )

    states = []
    for index, frame in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
        states.append(
            {
                "t": round(index / fps, 3),
                "frame": str(frame),
                "state": {},
            }
        )

    trace = {"video_id": video.stem, "states": states}
    output.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for scaffold-video but was not found") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed: {message}") from exc
