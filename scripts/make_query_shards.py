from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split unfinished JSONL queries into round-robin shards.")
    parser.add_argument("annotations_jsonl")
    parser.add_argument("--done-jsonl", default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.annotations_jsonl))
    done_ids = set()
    if args.done_jsonl and Path(args.done_jsonl).exists():
        done_ids = {row["id"] for row in read_jsonl(Path(args.done_jsonl))}

    remaining = [row for row in rows if row["id"] not in done_ids]
    shards = [[] for _ in range(args.num_shards)]
    for idx, row in enumerate(remaining):
        shards[idx % args.num_shards].append(row)

    output_paths = []
    for idx, shard in enumerate(shards):
        path = Path(f"{args.output_prefix}{idx}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for row in shard:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_paths.append(str(path))

    print(
        json.dumps(
            {
                "input": args.annotations_jsonl,
                "done": len(done_ids),
                "remaining": len(remaining),
                "num_shards": args.num_shards,
                "shard_sizes": [len(shard) for shard in shards],
                "outputs": output_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
