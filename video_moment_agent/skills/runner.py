from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MergeResult:
    rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    duplicates_ignored: int
    overlaid_rows: int
    missing_ids: list[str]
    error_rows: list[dict[str, Any]]


class RunnerSkill:
    """Batch runner utilities for shard merge, error queues, and evaluation."""

    @staticmethod
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        with p.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @staticmethod
    def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def interval_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
        inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = max(pred_end, gt_end) - min(pred_start, gt_start)
        return inter / union if union > 0 else 0.0

    def safe_iou(self, row: dict[str, Any], gt: dict[str, Any]) -> float:
        try:
            return self.interval_iou(
                float(row["pred_start"]),
                float(row["pred_end"]),
                float(gt["start"]),
                float(gt["end"]),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    def metrics(self, rows: list[dict[str, Any]], gt_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
        evaluated = []
        for row in rows:
            gt = gt_by_id.get(row["id"])
            if not gt:
                continue
            iou = self.safe_iou(row, gt)
            row["iou"] = round(iou, 4)
            evaluated.append(iou)
        denom = max(1, len(evaluated))
        route_rows = [row.get("route") or {} for row in rows]
        usage_total = 0
        for row in rows:
            for usage in (row.get("usage") or {}).values():
                usage_total += int(usage.get("total_tokens") or 0)
            for candidate_name in ("normal_candidate", "hard_candidate"):
                candidate = row.get(candidate_name) or {}
                for usage in (candidate.get("usage") or {}).values():
                    usage_total += int(usage.get("total_tokens") or 0)
        return {
            "rows": len(rows),
            "num_evaluated": len(evaluated),
            "mIoU": round(sum(evaluated) / denom, 4),
            "R@0.3": round(sum(iou >= 0.3 for iou in evaluated) / denom, 4),
            "R@0.5": round(sum(iou >= 0.5 for iou in evaluated) / denom, 4),
            "R@0.7": round(sum(iou >= 0.7 for iou in evaluated) / denom, 4),
            "errors": sum(1 for row in rows if row.get("error")),
            "hard_triggered": sum(1 for route in route_rows if route.get("auto_hard_triggered")),
            "hard_selected": sum(1 for route in route_rows if route.get("selected") == "hard"),
            "total_tokens_recorded": usage_total,
        }

    def merge(
        self,
        annotations: list[dict[str, Any]],
        prediction_files: list[str | Path],
        overlay_files: list[str | Path] | None = None,
        keep_error_overlays: bool = False,
    ) -> MergeResult:
        gt_by_id = {row["id"]: row for row in annotations}
        rows_by_id: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []

        for path in prediction_files:
            for row in self.read_jsonl(path):
                query_id = row["id"]
                if query_id in rows_by_id:
                    duplicates.append(query_id)
                    continue
                rows_by_id[query_id] = row

        overlaid = 0
        for path in overlay_files or []:
            for row in self.read_jsonl(path):
                if row.get("error") and not keep_error_overlays:
                    continue
                rows_by_id[row["id"]] = row
                overlaid += 1

        ordered_rows = [rows_by_id[row["id"]] for row in annotations if row["id"] in rows_by_id]
        missing_ids = [row["id"] for row in annotations if row["id"] not in rows_by_id]
        error_rows = [row for row in ordered_rows if row.get("error")]
        return MergeResult(
            rows=ordered_rows,
            metrics=self.metrics(ordered_rows, gt_by_id),
            duplicates_ignored=len(duplicates),
            overlaid_rows=overlaid,
            missing_ids=missing_ids,
            error_rows=error_rows,
        )
