import argparse
import json
from pathlib import Path
import torch
from .constants import SEEDS, CAP, STREAMS, MECHANISMS, VERSION

COMMANDS = ("reproduce", "m1-faithfulness", "m2-k-dial", "m3-state-causality",
            "m4-importance-assignment", "m5-predict", "m6-release", "analyze")

def parser():
    p = argparse.ArgumentParser(description="Standalone CAP mechanisms: implementation validation precedes scientific runs")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        q = sub.add_parser(name)
        q.add_argument("--data-dir", type=Path, default=Path("data"))
        q.add_argument("--output-dir", type=Path, default=Path("results/cap_mechanistic_final"))
        q.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        q.add_argument("--device", default="cpu")
        q.add_argument("--resume", action="store_true")
        q.add_argument("--allow-unverified-reference", action="store_true")
        q.add_argument("--smoke", action="store_true", help="Synthetic data, 4 reference tasks × 3 updates; writes under output-dir/smoke")
        q.add_argument("--dry-run", action="store_true", help="Print protocol without creating data or running training")
        q.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; 1 avoids small-MLP overhead")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or any(s not in SEEDS for s in args.seeds):
        raise ValueError(f"Seeds must be a unique subset of {SEEDS}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.output_dir = args.output_dir/"smoke"
    if args.dry_run:
        print(json.dumps({"command": args.command, "version": VERSION, "baseline": CAP,
                          "streams": STREAMS, "mechanisms": MECHANISMS, "smoke": args.smoke,
                          "output_dir": str(args.output_dir), "executes_training": False}, indent=2))
        return 0
    if args.command == "reproduce":
        from .runner import reproduce
        reproduce(args)
    elif args.command == "m1-faithfulness":
        from .faithfulness import run
        run(args)
    elif args.command == "m5-predict":
        from .prediction import run
        run(args)
    elif args.command == "analyze":
        from .analysis import analyze
        analyze(args)
    else:
        from .runner import causal
        causal(args, args.command[:2])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
