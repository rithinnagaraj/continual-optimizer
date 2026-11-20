"""UPGD native utility / Fisher / random M1 probes. Never trains a model."""
import argparse
import json
from pathlib import Path
import upgd_rebuttal as parent
import upgd_m1 as protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    for name in ('init', 'run', 'status', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--source-dir', type=Path, default=parent.DEFAULT_OUTPUT)
        p.add_argument('--output-dir', type=Path, default=protocol.DEFAULT_OUTPUT)
        if name == 'run':
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--seeds', type=int, nargs='+')
            group.add_argument('--all', action='store_true')
            p.add_argument('--device', default='cuda')
            p.add_argument('--data-dir', type=Path, default=protocol.ROOT/'data')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--max-groups', type=parent.positive,
                           help='Pause each seed after this many additional groups; full Fisher preparation first')
            p.add_argument('--max-fisher-domains', type=parent.positive,
                           help='Pause after this many additional Fisher domains if preparation remains')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        value = dict(**protocol.settings(), groups_per_seed=len(protocol.groups(protocol.settings(), 42)),
                     training_updates=0, output=str(protocol.DEFAULT_OUTPUT))
    else:
        p = protocol.campaign(args.source_dir, args.output_dir, initialize=args.command=='init')
        if args.command == 'run':
            seeds = p['seeds'] if args.all else args.seeds
            if len(seeds)!=len(set(seeds)) or set(seeds)-set(p['seeds']):
                parser.error('Choose unique seeds from the frozen reporting set')
            from upgd_m1_runtime import run_job
            value = []
            for seed in seeds:
                result = run_job(p, args.source_dir, args.output_dir, seed, data_dir=args.data_dir,
                    device=args.device, resume=args.resume, max_groups=args.max_groups,
                    max_fisher_domains=args.max_fisher_domains)
                value.append(dict(seed=seed, status=result['status']))
        elif args.command == 'report':
            value = protocol.report(p, args.output_dir)
        elif args.command == 'status':
            value = []
            for seed in p['seeds']:
                cell = protocol.directory(args.output_dir, seed)
                result = protocol.verify_result(p, args.output_dir, seed)
                value.append(dict(seed=seed, status='complete' if result else 'pending_or_partial',
                    committed_groups=len(protocol.completed_rows(p, args.output_dir, seed)),
                    total_groups=len(protocol.groups(p, seed)),
                    committed_fisher_domains=len(list((cell/'fisher').glob('*.json'))),
                    **parent.timing_summary(cell/'timing.jsonl')))
        else:
            value = dict(campaign_hash=parent.digest(p), smoke=p['smoke'], seeds=p['seeds'],
                         groups_per_seed=len(protocol.groups(p, p['seeds'][0])))
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
