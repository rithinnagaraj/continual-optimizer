"""CIFAR-100 AdamW/UPGD pilots, LR selection, and 80-task evaluation."""
import argparse
import json
from pathlib import Path
import cifar_baselines as protocol
import upgd_rebuttal as shared


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan')
    for name in ('init','jobs','run','select','status','report'):
        p=sub.add_parser(name)
        p.add_argument('--output-dir',type=Path,default=protocol.DEFAULT_OUTPUT)
        if name=='init': p.add_argument('--smoke',action='store_true')
        if name in ('jobs','run','status','report'):
            p.add_argument('--stage',choices=('pilot','search','eval'),required=True)
            p.add_argument('--arms',nargs='+',choices=('adamw','upgd','cap'),default=['adamw','upgd'])
        if name=='report': p.add_argument('--include-archived-cap',action='store_true')
        if name=='run': shared.add_run_arguments(p)
    args=parser.parse_args(argv)
    if args.command=='plan':
        p=protocol.protocol()
        value=dict(protocol=p,pilot_jobs=[j['id'] for j in protocol.jobs(p,None,'pilot')],
                   search_jobs=40,default_eval_jobs=40,search_updates=200000,default_eval_updates=16000000,
                   pilot_updates=4000,default_output=str(protocol.DEFAULT_OUTPUT))
    else:
        p=protocol.campaign(args.output_dir,initialize=args.command=='init',smoke=getattr(args,'smoke',False))
        if args.command=='init': value=dict(campaign_hash=shared.digest(p),smoke=p['smoke'])
        elif args.command=='select': value=protocol.select(p,args.output_dir)
        elif args.command=='report':
            if args.stage=='pilot': value=protocol.pilot_report(p,args.output_dir)
            elif args.stage=='eval': value=protocol.report(p,args.output_dir,args.arms,args.include_archived_cap)
            else: value=protocol.select(p,args.output_dir,write=False)
        else:
            if len(set(args.arms))!=len(args.arms): parser.error('Arms must be unique')
            if args.stage!='eval' and args.arms!=['adamw','upgd']: parser.error('--arms applies only to evaluation')
            jobs=protocol.jobs(p,args.output_dir,args.stage,args.arms)
            if args.command=='jobs': value=jobs
            elif args.command=='status':
                value=[dict(id=j['id'],status=(protocol.result(args.output_dir,j,False) or {}).get('status','pending_or_partial'),
                            **shared.timing_summary(args.output_dir/'jobs'/j['id']/'timing.jsonl')) for j in jobs]
            else:
                requested=set(args.job_ids or [])
                if requested-{j['id'] for j in jobs}: parser.error('Unknown job IDs')
                chosen=jobs if args.all else [j for j in jobs if j['id'] in requested]
                if args.max_jobs is not None: chosen=[j for j in chosen if protocol.result(args.output_dir,j,False) is None][:args.max_jobs]
                if args.dry_run: value=chosen
                else:
                    from cifar_baselines_runtime import run_job
                    value=[]
                    for job in chosen:
                        r=run_job(args.output_dir,job,data_dir=args.data_dir,device=args.device,resume=args.resume,
                                  checkpoint_every=args.checkpoint_every,stop_after_updates=args.stop_after_updates)
                        value.append(dict(id=job['id'],status=r['status']))
    print(json.dumps(value,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
