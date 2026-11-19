"""Evaluate the frozen rebuttal winner on five seeds and the paper's A/B streams."""
import argparse
import json
from pathlib import Path
import upgd_rebuttal as protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("jobs", "run", "report", "status"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=protocol.DEFAULT_OUTPUT)
        p.add_argument("--arms", nargs="+", choices=("selected", "incumbent", "cap"), default=["selected"])
        if name == "run":
            protocol.add_run_arguments(p)
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    if args.command == "run":
        protocol.dispatch(args, "eval", args.arms)
        return 0
    if args.command == "report":
        value = protocol.evaluation_report(args.output_dir, args.arms)
    else:
        value = protocol.jobs(args.output_dir, "eval", args.arms)
        if args.command == "status":
            value = [dict(id=j["id"], status=(protocol.result(args.output_dir, j, required=False) or {}).get("status", "pending_or_partial"),
                          **protocol.timing_summary(args.output_dir/"jobs"/j["id"]/"timing.jsonl")) for j in value]
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
