from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_moment_agent.skills import RouterSkill, RunnerSkill


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def metrics(rows: list[dict[str, Any]], gt_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return RunnerSkill().metrics(rows, gt_by_id)


def choose_by_margin(row: dict[str, Any], margin: float) -> tuple[dict[str, Any], str]:
    normal = row.get("normal_candidate")
    hard = row.get("hard_candidate")
    if not normal or not hard:
        return row, "existing"
    choice, decision = RouterSkill(hard_margin=margin).choose(normal, hard)
    return choice, decision.selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-tune the normal-vs-hard router using stored candidates."
    )
    parser.add_argument("annotations_jsonl")
    parser.add_argument("predictions_jsonl")
    parser.add_argument("--margins", default="-0.5,-0.2,0,0.2,0.5,1.0")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument(
        "--select-margin",
        type=float,
        default=None,
        help="If set, write predictions selected with this margin.",
    )
    args = parser.parse_args()

    gt_by_id = {row["id"]: row for row in read_jsonl(args.annotations_jsonl)}
    rows = read_jsonl(args.predictions_jsonl)
    results = []
    for margin in [float(item) for item in args.margins.split(",") if item.strip()]:
        selected = []
        hard_selected = 0
        for row in rows:
            choice, name = choose_by_margin(row, margin)
            selected.append(choice)
            hard_selected += int(name == "hard")
        result = {
            "margin": margin,
            "hard_selected": hard_selected,
            **metrics(selected, gt_by_id),
        }
        results.append(result)
    print(json.dumps({"input": args.predictions_jsonl, "results": results}, ensure_ascii=False, indent=2))

    if args.output_jsonl and args.select_margin is not None:
        selected = [choose_by_margin(row, args.select_margin)[0] for row in rows]
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in selected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
