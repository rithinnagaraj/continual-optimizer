"""Prepare and run isolated resumable UPGD rebuttal tuning jobs."""
import argparse
import json
from pathlib import Path
import upgd_rebuttal as protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "init", "jobs", "run", "shortlist", "select", "report", "status"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=protocol.DEFAULT_OUTPUT)
        if name in {"plan", "init"}:
            p.add_argument("--smoke", action="store_true", help="Separate reduced synthetic protocol; never scientific evidence")
        if name in {"jobs", "run", "status"}:
            p.add_argument("--stage", choices=("search", "confirm", "reference"), default="search")
        if name == "run":
            protocol.add_run_arguments(p)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = protocol.protocol(args.smoke)
    elif args.command == "init":
        result = protocol.initialize(args.output_dir, args.smoke)
    elif args.command == "jobs":
        result = protocol.jobs(args.output_dir, args.stage)
    elif args.command == "run":
        protocol.dispatch(args, args.stage)
        return 0
    elif args.command == "shortlist":
        result = protocol.make_shortlist(args.output_dir)
    elif args.command == "select":
        result = protocol.choose(args.output_dir)
    elif args.command == "report":
        result = protocol.tuning_report(args.output_dir)
    else:
        result = []
        for j in protocol.jobs(args.output_dir, args.stage):
            row = protocol.result(args.output_dir, j, required=False)
            result.append(dict(id=j["id"], status=row["status"] if row else "pending_or_partial",
                **protocol.timing_summary(args.output_dir/"jobs"/j["id"]/"timing.jsonl")))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
