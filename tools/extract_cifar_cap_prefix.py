"""Recover matched CAP 80-task metrics from the pinned historical review archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import upgd_rebuttal as shared

ARCHIVE_SHA256='8236000fda453531dbb853eae5aa94e23ce4737191fd22977c64a1409f812d02'


def extract(archive):
    if shared.file_hash(archive)!=ARCHIVE_SHA256:
        raise RuntimeError('Not the pinned historical CIFAR review archive')
    rows=[];sources={}
    with zipfile.ZipFile(archive) as z:
        for model in ('mlp','cnn'):
            prefix=f'{model}/scientific/reproduction/'
            config=json.loads(z.read(prefix+'run_config.json'))
            gate=json.loads(z.read(prefix+'reproduction_gate.json'))
            if config['smoke'] or config['allow_unverified_reference'] or not gate['passed'] or not gate['valid_reference']:
                raise RuntimeError('Invalid historical scientific reference')
            assert config['seeds']==shared.REPORT_SEEDS
            assert config['protocol']['patch_size']==4
            assert config['streams']['updates_per_task']==5000 and config['streams']['pool_size']==40
            assert config['streams']['test_examples']==2000 and config['streams']['test_subset_seed']==999
            assert shared.digest(config['baseline'])==gate['baseline_config_sha256']
            tables={name:list(csv.DictReader(io.StringIO(z.read(prefix+f'summary/{name}.csv').decode()))) for name in ('tasks','pool')}
            for seed in shared.REPORT_SEEDS:
                task=[r for r in tables['tasks'] if int(r['seed'])==seed and r['stream']=='a']
                pool=[r for r in tables['pool'] if int(r['seed'])==seed and r['stream']=='b']
                assert sorted(int(r['task']) for r in task)==list(range(1,201))
                assert len(pool)==8000 and len({(int(r['task']),int(r['domain'])) for r in pool})==8000
                for t in range(1,201):
                    subset=[r for r in pool if int(r['task'])==t]
                    assert sorted(int(r['domain']) for r in subset)==list(range(40))
                    assert sum(r['is_current']=='True' for r in subset)==1
                for r in task+pool:
                    assert r['source_version']==config['source_version']
                    assert r['experiment_config_sha256']==shared.digest(config)
                    assert r['baseline_config_sha256']==shared.digest(config['baseline'])
                # Confirm the interpretation against recorded full-horizon gate values.
                old_p=mean(float(r['historical_sampled_plasticity']) for r in task if int(r['task'])>100)
                old_r=mean(float(r['accuracy']) for r in pool if int(r['task'])==200 and r['is_current']=='False')
                import math
                assert math.isclose(old_p,next(r['observed'] for r in gate['rows'] if r['seed']==seed and r['metric']=='P_sampled_100step'),abs_tol=1e-12)
                assert math.isclose(old_r,next(r['observed'] for r in gate['rows'] if r['seed']==seed and r['metric']=='R'),abs_tol=1e-12)
                pv=mean(float(r['historical_sampled_plasticity']) for r in task if 40<int(r['task'])<=80)
                rv=mean(float(r['accuracy']) for r in pool if int(r['task'])==80 and r['is_current']=='False')
                rows.append(dict(model=model,arm='cap_archived',seed=seed,plasticity=pv,retention=rv,balanced=min(pv,rv),
                    provenance='historical CAP 80-task prefix; not a same-environment rerun'))
            sources[model]=dict(config=config,gate_scope=gate['validation_scope'],
                table_sha256={name:hashlib.sha256(z.read(prefix+f'summary/{name}.csv')).hexdigest() for name in ('tasks','pool')})
    return dict(schema='cifar-cap-prefix-80-v1',archive_sha256=ARCHIVE_SHA256,eval_tasks=80,patch_size=4,
        per_seed=rows,sources=sources,
        metrics=dict(plasticity='mean historical_sampled_plasticity over stream A tasks 41-80',
                     retention='mean non-current stream B accuracy at task 80',balanced='min(P,R) per seed'),
        limitations=['Derived from archived task/pool summaries; no checkpoint bytes in review archive.',
                     'Historical CAP and new baseline environments may differ; no exact trajectory equality claimed.',
                     'CAP LR selection used seed 42; new baseline development seed is disjoint from reporting seeds.',
                     '80-task prefixes, not original 200-task endpoints; earlier exploratory test access remains disclosed.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/cifar_cap_reference/prefix_80.json')
    args=parser.parse_args()
    value=extract(args.archive)
    shared.freeze_json(args.output,value)
    print(json.dumps(dict(output=str(args.output),per_seed=value['per_seed']),indent=2))


if __name__=='__main__': main()
