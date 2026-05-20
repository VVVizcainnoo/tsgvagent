from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a reproducible shell plan for sharded sparse-agent batches."
    )
    parser.add_argument("--annotations-jsonl", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--programs-jsonl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--work-prefix", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--script-path", default="scripts/sparse_vlm_agent.py")
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--auto-hard-mode", action="store_true")
    parser.add_argument("--fine-max-frames", type=int, default=16)
    parser.add_argument("--hard-fine-max-frames", type=int, default=24)
    parser.add_argument("--auto-hard-margin", type=float, default=0.2)
    parser.add_argument("--output-sh", required=True)
    args = parser.parse_args()

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        ': "${SILICONFLOW_API_KEY:?Missing SILICONFLOW_API_KEY}"',
        "",
    ]
    commands = []
    for idx in range(args.num_shards):
        shard = f"{args.output_prefix}_shard{idx}.jsonl"
        pred = f"{args.output_prefix}_shard{idx}_predictions.jsonl"
        log = f"{args.output_prefix}_shard{idx}.log"
        work = f"{args.work_prefix}_shard{idx}_work"
        cli = [
            args.python_bin,
            args.script_path,
            shard,
            "--frame-root",
            args.frame_root,
            "--output-jsonl",
            pred,
            "--work-dir",
            work,
            "--programs-jsonl",
            args.programs_jsonl,
            "--model",
            args.model,
            "--fine-max-frames",
            str(args.fine_max_frames),
        ]
        if args.auto_hard_mode:
            cli.extend(
                [
                    "--auto-hard-mode",
                    "--hard-fine-max-frames",
                    str(args.hard_fine_max_frames),
                    "--auto-hard-margin",
                    str(args.auto_hard_margin),
                ]
            )
        commands.append(
            "SILICONFLOW_API_KEY=\"$SILICONFLOW_API_KEY\" nohup "
            + " ".join(cli)
            + f" > {log} 2>&1 &"
        )
    lines.extend(commands)
    lines.append("echo started")

    out = Path(args.output_sh)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(0o755)
    print(
        json.dumps(
            {
                "output_sh": str(out),
                "num_shards": args.num_shards,
                "prediction_glob": f"{args.output_prefix}_shard*_predictions.jsonl",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
