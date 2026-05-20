from __future__ import annotations

import argparse
import json

from video_moment_agent.skills import RunnerSkill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge shard predictions, optionally overlay reruns, export error queue, and evaluate."
    )
    parser.add_argument("annotations_jsonl")
    parser.add_argument("--prediction-jsonl", action="append", required=True)
    parser.add_argument("--overlay-jsonl", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--error-annotations-jsonl", default=None)
    parser.add_argument("--keep-error-overlays", action="store_true")
    args = parser.parse_args()

    runner = RunnerSkill()
    annotations = runner.read_jsonl(args.annotations_jsonl)
    merged = runner.merge(
        annotations,
        args.prediction_jsonl,
        overlay_files=args.overlay_jsonl,
        keep_error_overlays=args.keep_error_overlays,
    )
    runner.write_jsonl(args.output_jsonl, merged.rows)

    if args.error_annotations_jsonl:
        gt_by_id = {row["id"]: row for row in annotations}
        error_annotations = [gt_by_id[row["id"]] for row in merged.error_rows if row["id"] in gt_by_id]
        runner.write_jsonl(args.error_annotations_jsonl, error_annotations)

    result = {
        "annotations": args.annotations_jsonl,
        "output_jsonl": args.output_jsonl,
        "prediction_files": args.prediction_jsonl,
        "overlay_files": args.overlay_jsonl,
        "duplicates_ignored": merged.duplicates_ignored,
        "overlaid_rows": merged.overlaid_rows,
        "missing": len(merged.missing_ids),
        "error_queue": args.error_annotations_jsonl,
        **merged.metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
