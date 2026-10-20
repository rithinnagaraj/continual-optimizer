import argparse
import json
from pathlib import Path
import torch
from .constants import SEEDS, CAP, STREAMS, MECHANISMS, VERSION
from . import protocol

COMMANDS = ("reference", "m1-faithfulness", "m2-k-dial", "m3-state-causality",
            "m4-importance-assignment", "analyze")

def parser():
    p = argparse.ArgumentParser(description="Standalone CAP mechanisms: implementation validation precedes scientific runs")
    sub = p.add_subparsers(dest="command", required=True)
    for name in (*COMMANDS, 'all'):
        q = sub.add_parser(name)
        q.add_argument('--model', choices=('mlp','cnn','both'), required=True)
        q.add_argument('--patch-size', type=int, choices=(2,4,8,16), default=4)
        q.add_argument('--lr-selection', type=Path, help='Completed sweep JSON; puts outputs in a selection-specific subdirectory')
        q.add_argument('--paper-settings', action='store_true', help='Fixed paper rates: MLP 2e-4, CNN 3e-4; separate output namespace')
        q.add_argument("--data-dir", type=Path, default=Path("data"))
        q.add_argument("--output-dir", type=Path, default=Path("results/cifar_mechanistic"))
        q.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        q.add_argument("--device", default="cpu")
        q.add_argument("--resume", action="store_true")
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
    if args.paper_settings and args.lr_selection:
        raise ValueError('Choose fixed paper settings or a newly selected pilot rate')
    torch.set_num_threads(args.threads)
    args.allow_unverified_reference = args.smoke
    base = args.output_dir
    default_lr = json.loads((Path(__file__).parent/'configs/final_cap.json').read_text())['lr']
    command = args.command
    for model in (('mlp','cnn') if args.model=='both' else (args.model,)):
        protocol.configure(model,args.patch_size)
        CAP['lr'] = default_lr
        protocol.LR_SELECTION = None
        selected_base = base
        if args.paper_settings:
            CAP_PAPER_RATES = {'mlp': 2e-4, 'cnn': 3e-4}
            CAP['lr'] = CAP_PAPER_RATES[model]
            protocol.LR_SELECTION = 'paper-fixed-rates-v1'
            selected_base = base/'paper_fixed'
        if args.lr_selection:
            from .lr_selection import load_selection
            from .checkpointing import source_version
            CAP['lr'], protocol.LR_SELECTION = load_selection(args.lr_selection,model,args.patch_size,source_version(),CAP)
            selected_base = base/('lrsel_'+protocol.LR_SELECTION[:16])
        args.output_dir = selected_base/model/('smoke' if args.smoke else 'scientific')
        for name in (COMMANDS if command=='all' else (command,)):
            args.command=name
            execute(args)
    return 0

def execute(args):
    if args.dry_run:
        print(json.dumps({"command": args.command, "version": VERSION, "baseline": CAP, 'protocol':protocol.specification(),
                          "streams": STREAMS, "mechanisms": MECHANISMS, "smoke": args.smoke,
                          "output_dir": str(args.output_dir), "executes_training": False}, indent=2))
        return 0
    print(f'Starting {protocol.ARCHITECTURE}: {args.command}', flush=True)
    if args.command == "reference":
        from .runner import reproduce
        reproduce(args)
    elif args.command == "m1-faithfulness":
        from .faithfulness import run
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
