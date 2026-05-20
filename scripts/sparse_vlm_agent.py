from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_moment_agent.skills import RouterSkill


API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_programs(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    programs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        query_id = row.get("query_id") or row.get("id")
        if query_id:
            programs[query_id] = row
    return programs


def compact_program(program: dict[str, Any] | None) -> str:
    if not program:
        return "No state program is available. Infer the entities and event structure from the query."
    entities = [
        {
            "id": ent.get("id"),
            "type": ent.get("primary_type"),
            "required": ent.get("required", True),
        }
        for ent in program.get("entities", [])
    ]
    requirements = []
    for req in program.get("requirements", []):
        keep = {
            key: value
            for key, value in req.items()
            if key
            in {
                "id",
                "type",
                "subject",
                "action",
                "motion",
                "entity",
                "attribute",
                "relation",
                "object",
                "target",
                "before",
                "after",
                "required",
            }
        }
        requirements.append(keep)
    constraints = program.get("constraints", [])
    payload = {
        "entities": entities,
        "requirements": requirements,
        "constraints": constraints,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def requirement_types(program: dict[str, Any] | None) -> set[str]:
    if not program:
        return set()
    return {str(req.get("type", "")) for req in program.get("requirements", []) if req.get("type")}


def image_to_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call_vlm(api_key: str, image_path: Path, prompt: str, model: str, max_tokens: int = 500, retries: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid compact JSON. Do not use markdown."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return extract_json(raw["choices"][0]["message"]["content"]), raw
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def make_grid(frames: list[Path], times: list[float], output: Path, cols: int = 4, cell_w: int = 320) -> None:
    imgs = []
    for frame in frames:
        img = Image.open(frame).convert("RGB")
        ratio = cell_w / img.width
        imgs.append(img.resize((cell_w, int(img.height * ratio))))
    cell_h = max(img.height for img in imgs) + 30
    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, img in enumerate(imgs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + 30
        grid.paste(img, (x, y))
        draw.text((x + 8, y - 24), f"#{idx+1}  t={times[idx]:.1f}s", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, quality=92)


def pick_even(frames: list[Path], max_n: int) -> list[int]:
    if len(frames) <= max_n:
        return list(range(len(frames)))
    return sorted({round(i * (len(frames) - 1) / (max_n - 1)) for i in range(max_n)})


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def extract_evidence_times(payload: dict[str, Any]) -> list[float]:
    times: list[float] = []
    for item in payload.get("matched_evidence", []) or []:
        if not isinstance(item, str):
            continue
        for match in re.finditer(r"t\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s?", item):
            times.append(float(match.group(1)))
    return times


def postprocess_window(
    pred_start: float,
    pred_end: float,
    c0: float,
    c1: float,
    video_end: float,
    program: dict[str, Any] | None,
    fine: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    """Small verifier-style guardrail for sparse-frame VLM outputs."""
    types = requirement_types(program)
    duration = max(0.0, pred_end - pred_start)
    confidence = float(fine.get("confidence", 0.0) or 0.0)
    if {"state_hold", "interaction_hold", "relation_hold"} & types:
        min_duration = 6.0
        max_duration = 18.0
    elif {"motion_event"} & types:
        min_duration = 5.0
        max_duration = 16.0
    elif {"state_transition", "interaction_transition", "relation_transition"} & types:
        min_duration = 5.0
        max_duration = 14.0
    else:
        min_duration = 4.0
        max_duration = 14.0

    notes: list[str] = []
    if duration <= 0:
        evidence_times = extract_evidence_times(fine)
        if evidence_times:
            center = sum(evidence_times) / len(evidence_times)
            notes.append("expanded_degenerate_window_from_evidence")
        elif (c1 - c0) > 0 and (c1 - c0) <= max_duration:
            center = (c0 + c1) / 2
            notes.append("expanded_degenerate_window_from_candidate")
        else:
            center = pred_start if pred_start > 0 else min(video_end, max(0.0, c0))
            notes.append("expanded_degenerate_window_from_start")
        pred_start = max(0.0, center - min_duration / 2)
        pred_end = min(video_end, center + min_duration / 2)
    elif duration < min_duration and confidence >= 0.6:
        center = (pred_start + pred_end) / 2
        pred_start = max(0.0, center - min_duration / 2)
        pred_end = min(video_end, center + min_duration / 2)
        notes.append(f"expanded_short_window_to_{min_duration:g}s")

    duration = max(0.0, pred_end - pred_start)
    if duration > max_duration and {"state_transition", "interaction_transition", "relation_transition", "action_event"} & types:
        # If the model returns a whole persistent aftermath for a transition, keep
        # the earlier part of the candidate span where the change likely occurs.
        pred_end = min(pred_end, pred_start + max_duration)
        notes.append(f"capped_transition_window_to_{max_duration:g}s")

    pred_start = max(0.0, min(pred_start, video_end))
    pred_end = max(pred_start, min(pred_end, video_end))
    return pred_start, pred_end, {"types": sorted(types), "notes": notes}


def routing_quality(row: dict[str, Any]) -> float:
    return RouterSkill().routing_quality(row)


def should_auto_hard(row: dict[str, Any], program: dict[str, Any] | None) -> tuple[bool, list[str]]:
    return RouterSkill().should_hard(row, program)


def choose_auto_route(normal: dict[str, Any], hard: dict[str, Any], margin: float) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen, decision = RouterSkill(hard_margin=margin).choose(normal, hard)
    chosen = dict(chosen)
    chosen["route"] = decision.to_dict()
    return chosen, chosen["route"]


def run_one(
    api_key: str,
    query: dict[str, Any],
    frame_dir: Path,
    work_dir: Path,
    fps: float,
    model: str,
    program: dict[str, Any] | None = None,
    use_verifier: bool = True,
    hard_mode: bool = False,
    fine_max_frames: int = 16,
) -> dict[str, Any]:
    all_frames = sorted(frame_dir.glob("*.jpg"))
    if not all_frames:
        raise ValueError(f"No extracted frames found in {frame_dir}")
    coarse_idx = pick_even(all_frames, 8)
    coarse_frames = [all_frames[i] for i in coarse_idx]
    coarse_times = [i / fps for i in coarse_idx]
    coarse_grid = work_dir / f"{query['id']}_coarse.jpg"
    make_grid(coarse_frames, coarse_times, coarse_grid)
    program_text = compact_program(program)
    coarse_prompt = f"""Query: {query['query']}
State program: {program_text}
The image is a grid of sampled video frames. Each cell has a frame number and timestamp.
Find the most likely temporal interval containing the query event.
Use the state program as the contract:
- entities are the visual participants to look for;
- requirements describe the action, relation, state change, or state hold that must be grounded;
- constraints describe same-event or temporal consistency.
Charades-STA queries are answerable; if evidence is weak, still choose the best interval instead of returning an empty match.
Return JSON:
{{"relevant": true/false, "coarse_start": seconds, "coarse_end": seconds, "chosen_frames": [frame_numbers], "missing_requirements": ["..."], "evidence": "short reason"}}"""
    coarse, coarse_raw = call_vlm(api_key, coarse_grid, coarse_prompt, model)
    time.sleep(2)

    c0 = float(coarse.get("coarse_start", 0.0))
    c1 = float(coarse.get("coarse_end", coarse_times[-1] if coarse_times else 0.0))
    if c1 < c0:
        c0, c1 = c1, c0
    video_end = (len(all_frames) - 1) / fps
    # Charades-STA queries are answerable. If the coarse scan is uncertain or
    # returns a degenerate interval, keep the fine stage broad instead of
    # over-trusting a false negative from sparse visual evidence.
    if hard_mode:
        c0, c1 = 0.0, video_end
    elif coarse.get("relevant") is False or (c1 - c0) < 4.0:
        c0, c1 = 0.0, video_end
    else:
        c0 = max(0.0, c0 - 2.0)
        c1 = min(video_end, c1 + 2.0)
        if c0 > video_end or c1 <= c0:
            c0, c1 = 0.0, video_end
    fine_idx = [i for i in range(len(all_frames)) if c0 <= i / fps <= c1]
    if not fine_idx:
        fine_idx = pick_even(all_frames, fine_max_frames)
    if len(fine_idx) > fine_max_frames:
        fine_idx = [fine_idx[i] for i in pick_even([Path(str(x)) for x in fine_idx], fine_max_frames)]
    fine_frames = [all_frames[i] for i in fine_idx]
    fine_times = [i / fps for i in fine_idx]
    fine_grid = work_dir / f"{query['id']}_fine.jpg"
    make_grid(fine_frames, fine_times, fine_grid)
    fine_prompt = f"""Query: {query['query']}
State program: {program_text}
The image is a denser grid from the candidate window. Decide the final moment boundaries.
Boundary rules:
- For a transition, start at the first visible preparation/contact/change and end when the required after-state is first established.
- For a hold state, include the continuous interval where the state is visible, not just one representative frame.
- For motion/action events, include the whole visible action, not only the most salient middle frame.
- Do not extend into unrelated aftermath after the requirement is already complete.
- The final interval must have positive duration. Do not return pred_start == pred_end unless the whole video has only one frame.
{"- HARD MODE: this grid may cover the whole video. If several similar events appear, list the candidate intervals first, then choose the one that best matches every entity and requirement in the state program. Do not choose a later similar event just because it is more visually salient." if hard_mode else ""}
Return JSON:
{{"pred_start": seconds, "pred_end": seconds, "confidence": 0_to_1, "top_candidates": [{{"start": seconds, "end": seconds, "score": 0_to_1, "reason": "..."}}], "matched_requirements": ["requirement ids or descriptions"], "matched_evidence": ["#frame t=seconds: reason"], "verifier": {{"complete": true/false, "temporal_order_ok": true/false, "same_event": true/false, "boundary_quality": "tight|too_short|too_long|uncertain"}}}}
Only use visible evidence from the frames."""
    fine, fine_raw = call_vlm(api_key, fine_grid, fine_prompt, model)
    verifier: dict[str, Any] | None = None
    verifier_raw: dict[str, Any] = {}
    if use_verifier:
        time.sleep(2)
        verifier_prompt = f"""Query: {query['query']}
State program: {program_text}
Candidate JSON: {json.dumps(fine, ensure_ascii=False)}
You are the consistency verifier. Check completeness, temporal order, same-event consistency, and boundary quality against the frame grid.
If the candidate is too short, expand to cover the full visible action/state. If it is too long, trim unrelated aftermath.
The verified interval must have positive duration; never return a zero-length [0, 0] interval for an answerable query.
{"If top_candidates are present, choose the highest scoring candidate that satisfies the state program, not necessarily the first visually salient action." if hard_mode else ""}
Return JSON:
{{"verified": true/false, "pred_start": seconds, "pred_end": seconds, "fix_reason": "short", "checks": {{"complete": true/false, "temporal_order_ok": true/false, "same_event": true/false, "boundary_quality": "tight|too_short|too_long|uncertain"}}}}"""
        verifier, verifier_raw = call_vlm(api_key, fine_grid, verifier_prompt, model)
    pred_start = float(fine.get("pred_start", c0))
    pred_end = float(fine.get("pred_end", c1))
    if verifier:
        pred_start = float(verifier.get("pred_start", pred_start))
        pred_end = float(verifier.get("pred_end", pred_end))
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    pred_start, pred_end, postprocess = postprocess_window(pred_start, pred_end, c0, c1, video_end, program, fine)
    gt_start = float(query["start"])
    gt_end = float(query["end"])
    return {
        "id": query["id"],
        "video_id": query["video_id"],
        "query": query["query"],
        "gt_start": gt_start,
        "gt_end": gt_end,
        "pred_start": round(pred_start, 3),
        "pred_end": round(pred_end, 3),
        "iou": round(iou(pred_start, pred_end, gt_start, gt_end), 4),
        "coarse": coarse,
        "fine": fine,
        "verifier": verifier,
        "postprocess": postprocess,
        "program": program,
        "usage": {
            "coarse": coarse_raw.get("usage", {}),
            "fine": fine_raw.get("usage", {}),
            "verifier": verifier_raw.get("usage", {}),
        },
        "method": "program_sparse_grid_agent" if program else "sparse_grid_agent",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Program-guided sparse VLM agent: coarse grid scan + fine grid localization.")
    parser.add_argument("annotations_jsonl")
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--work-dir", default="runs/sparse_agent_work")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--programs-jsonl", default=None, help="Optional Step1 program JSONL keyed by query_id/id.")
    parser.add_argument("--no-verifier", action="store_true", help="Disable the third verifier VLM call.")
    parser.add_argument("--hard-mode", action="store_true", help="Use full-video dense fine grid and candidate enumeration for difficult examples.")
    parser.add_argument("--auto-hard-mode", action="store_true", help="Run normal mode first, then automatically rerun hard mode for uncertain examples without using GT.")
    parser.add_argument("--auto-hard-margin", type=float, default=0.2)
    parser.add_argument("--fine-max-frames", type=int, default=16)
    parser.add_argument("--hard-fine-max-frames", type=int, default=24)
    args = parser.parse_args()

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing SILICONFLOW_API_KEY")

    queries = read_jsonl(args.annotations_jsonl)
    if args.video_id:
        queries = [q for q in queries if q["video_id"] == args.video_id]
    if args.limit_queries is not None:
        queries = queries[: args.limit_queries]
    programs = load_programs(args.programs_jsonl)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done_ids = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row["id"])
                    rows.append(row)

    with out.open("a", encoding="utf-8") as fh:
        for q in queries:
            if q["id"] in done_ids:
                continue
            try:
                normal_row = run_one(
                    api_key,
                    q,
                    Path(args.frame_root) / q["video_id"],
                    Path(args.work_dir),
                    args.fps,
                    args.model,
                    programs.get(q["id"]),
                    use_verifier=not args.no_verifier,
                    hard_mode=args.hard_mode,
                    fine_max_frames=args.fine_max_frames,
                )
                row = normal_row
                if args.auto_hard_mode:
                    trigger, reasons = should_auto_hard(normal_row, programs.get(q["id"]))
                    normal_row["route"] = {
                        "selected": "normal",
                        "auto_hard_triggered": trigger,
                        "trigger_reasons": reasons,
                        "normal_score": round(routing_quality(normal_row), 4),
                    }
                    if trigger:
                        hard_row = run_one(
                            api_key,
                            q,
                            Path(args.frame_root) / q["video_id"],
                            Path(args.work_dir) / "hard",
                            args.fps,
                            args.model,
                            programs.get(q["id"]),
                            use_verifier=not args.no_verifier,
                            hard_mode=True,
                            fine_max_frames=args.hard_fine_max_frames,
                        )
                        row, route = choose_auto_route(normal_row, hard_row, args.auto_hard_margin)
                        route["auto_hard_triggered"] = True
                        route["trigger_reasons"] = reasons
                        row["route"] = route
                        row["normal_candidate"] = normal_row
                        row["hard_candidate"] = hard_row
            except Exception as exc:
                row = {
                    "id": q["id"],
                    "video_id": q["video_id"],
                    "query": q["query"],
                    "gt_start": q["start"],
                    "gt_end": q["end"],
                    "pred_start": None,
                    "pred_end": None,
                    "iou": 0.0,
                    "error": str(exc),
                    "method": "sparse_grid_agent",
                }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(2)

    miou = sum(r["iou"] for r in rows) / max(1, len(rows))
    print(json.dumps({"output_jsonl": str(out), "rows": len(rows), "mIoU": round(miou, 4)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
