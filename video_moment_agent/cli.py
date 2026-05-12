from __future__ import annotations

import argparse

from .io import dump_json, load_trace
from .search import locate_moments
from .state_program import induce_state_program
from .video import scaffold_video_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="video_moment_agent",
        description="Locate video moments through state-program induction and trace search.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    induce = subparsers.add_parser("induce", help="Convert a query into P=(S, Delta, C).")
    induce.add_argument("query")

    locate = subparsers.add_parser("locate", help="Search a grounded state trace.")
    locate.add_argument("trace_json")
    locate.add_argument("query")
    locate.add_argument("--top-k", type=int, default=3)
    locate.add_argument("--window-seconds", type=float, default=8.0)

    scaffold = subparsers.add_parser(
        "scaffold-video",
        help="Extract frames with ffmpeg and create an empty state trace.",
    )
    scaffold.add_argument("video_path")
    scaffold.add_argument("output_json")
    scaffold.add_argument("--fps", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "induce":
        program = induce_state_program(args.query)
        print(dump_json(program.to_dict()))
        return

    if args.command == "locate":
        trace = load_trace(args.trace_json)
        program = induce_state_program(args.query)
        results = locate_moments(
            trace,
            program,
            top_k=args.top_k,
            window_seconds=args.window_seconds,
        )
        print(dump_json({"program": program.to_dict(), "results": [_result_to_dict(item) for item in results]}))
        return

    if args.command == "scaffold-video":
        trace = scaffold_video_trace(args.video_path, args.output_json, fps=args.fps)
        print(dump_json({"output_json": args.output_json, "frames": len(trace["states"])}))


def _result_to_dict(result):
    return {
        "start_t": result.start_t,
        "end_t": result.end_t,
        "score": result.score,
        "score_parts": {
            "S_state": result.state_score,
            "S_transition": result.transition_score,
            "S_constraint": result.constraint_score,
        },
        "matched_deltas": [
            {
                "var": item.delta.var,
                "before": item.before_value,
                "after": item.after_value,
                "start_t": item.start_t,
                "end_t": item.end_t,
            }
            for item in result.matched_deltas
        ],
        "verification": result.verification.__dict__,
    }
