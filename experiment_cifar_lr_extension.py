"""Run a bounded CIFAR LR extension, then evaluate its separately frozen selection."""
import argparse
import json
from pathlib import Path
import cifar_lr_extension as protocol
import upgd_rebuttal as shared


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('init','jobs','run','select','status','report'):
        p=sub.add_parser(name)
        p.add_argument('--output-dir',type=Path,default=protocol.DEFAULT_OUTPUT)
        if name=='init': p.add_argument('--parent-dir',type=Path,default=protocol.base.DEFAULT_OUTPUT)
        if name in ('jobs','run','status','report'):
            p.add_argument('--stage',choices=('search','eval'),required=True)
            p.add_argument('--arms',nargs='+',choices=('adamw','upgd','cap'),default=['adamw','upgd'])
        if name=='report': p.add_argument('--include-archived-cap',action='store_true')
        if name=='run': shared.add_run_arguments(p)
    args=parser.parse_args(argv)
    p=protocol.campaign(args.output_dir,initialize=args.command=='init',parent=getattr(args,'parent_dir',None))
    if args.command=='init':
        value=dict(campaign_hash=shared.digest(p),smoke=p['smoke'],new_jobs=12,new_updates=12*p['search_updates'])
    elif args.command=='select': value=protocol.select(p,args.output_dir)
    elif args.command=='report':
        value=protocol.select(p,args.output_dir,write=False) if args.stage=='search' else protocol.report(p,args.output_dir,args.arms,args.include_archived_cap)
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
                value=[]
                with protocol.worker_protocol() as runtime:
                    for job in chosen:
                        r=runtime.run_job(args.output_dir,job,data_dir=args.data_dir,device=args.device,resume=args.resume,
                            checkpoint_every=args.checkpoint_every,stop_after_updates=args.stop_after_updates)
                        value.append(dict(id=job['id'],status=r['status']))
    print(json.dumps(value,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
