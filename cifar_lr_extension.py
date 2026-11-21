"""Bounded, separately frozen CIFAR LR extension reusing the original search."""
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev
import os
import sys
import cifar_baselines as base
import upgd_rebuttal as shared

DEFAULT_OUTPUT = base.ROOT/'results/cifar_baselines_extended'
EXTRA_RATES = [.03, .1, .3]


def protocol(parent):
    old = base.campaign(parent)
    selected = shared.read_json(Path(parent)/'selected_lrs.json')
    if selected['campaign_hash'] != shared.digest(old) or selected['smoke'] != old['smoke']:
        raise RuntimeError('Parent selection/campaign mismatch')
    return dict(old, version='cifar-baselines-lr-extension-v1',
        learning_rates=old['learning_rates']+EXTRA_RATES,
        parent_campaign_hash=shared.digest(old), parent_selection_hash=shared.digest(selected),
        parent_rate_count=len(old['learning_rates']),
        tuning_scope='13 LR candidates per baseline/architecture; original 10 reused; bounded 3-rate extension; no automatic expansion',
        source_hashes={**old['source_hashes'], **{name:shared.file_hash(base.ROOT/name) for name in
            ('cifar_lr_extension.py','experiment_cifar_lr_extension.py')}})


def parent_path(output):
    link=shared.read_json(Path(output)/'parent.json')
    return (Path(output)/link['relative_output']).resolve()


def campaign(output, initialize=False, parent=None):
    output=Path(output)
    if initialize:
        if parent is None: raise ValueError('Parent campaign required')
        parent=Path(parent).resolve()
        if parent==output.resolve(): raise ValueError('Extension needs a separate output directory')
        # An extension must be decided before seeing the benchmark outcomes.
        if any((parent/'jobs').glob('eval-*/job.json')):
            raise RuntimeError('Parent evaluation already started; do not extend tuning after test access')
        old=base.campaign(parent)
        checked=base.select(old,parent,write=False)
        if shared.read_json(parent/'selected_lrs.json')!=checked:
            raise RuntimeError('Parent frozen selection differs from verified search')
        shared.freeze_json(output/'parent.json',dict(relative_output=Path(os.path.relpath(parent,output.resolve())).as_posix()))
        return shared.freeze_json(output/'campaign.json',protocol(parent))
    p=shared.read_json(output/'campaign.json')
    if p!=protocol(parent_path(output)):
        raise RuntimeError('Frozen extension/parent/source changed; preserve the existing campaign')
    return p


def jobs(p, output, stage, arms=base.ARMS):
    if stage=='search':
        return [base.spec(p,stage,m,a,lr_index=i) for m in base.MODELS for a in base.ARMS
                for i in range(p['parent_rate_count'],len(p['learning_rates']))]
    if stage!='eval': raise ValueError('Extension stages are search and eval')
    selected=select(p,output,write=False)
    if shared.read_json(Path(output)/'selected_lrs.json')!=selected:
        raise RuntimeError('Frozen extended selection differs from verified search')
    if any(v['at_grid_boundary'] for v in selected['selected'].values()):
        raise RuntimeError('Extended winner is still at a grid boundary: review before evaluation')
    return [base.spec(p,stage,m,a,seed=s,stream=t,
        lr=p['cap_learning_rates'][m] if a=='cap' else selected['selected'][f'{m}-{a}']['lr'])
        for m in base.MODELS for a in arms for s in p['report_seeds'] for t in ('a','b')]


result=base.result


def select(p,output,write=True):
    parent=parent_path(output);old=base.campaign(parent)
    old_selection=shared.read_json(parent/'selected_lrs.json')
    if shared.digest(old_selection)!=p['parent_selection_hash']:
        raise RuntimeError('Parent evidence changed')
    previous=[base.result(parent,j) for j in base.jobs(old,parent,'search')]
    if {r['job']['id']:shared.digest(r) for r in previous}!=old_selection['result_hashes']:
        raise RuntimeError('Parent search artifacts changed')
    added=[result(output,j) for j in jobs(p,output,'search')]
    rows=previous+added;winners={}
    for model in base.MODELS:
        for arm in base.ARMS:
            candidates=[r for r in rows if r['job']['model']==model and r['job']['arm']==arm and r['status']=='complete']
            if not candidates: raise RuntimeError(f'All candidates diverged for {model}/{arm}')
            best=min(candidates,key=lambda r:(-r['metrics']['validation_accuracy'],r['metrics']['validation_loss'],r['job']['lr']))
            job=best['job']
            winners[f'{model}-{arm}']=dict(lr=job['lr'],lr_index=job['lr_index'],metrics=best['metrics'],
                at_grid_boundary=job['lr_index'] in (0,len(p['learning_rates'])-1),
                source='parent' if job['lr_index']<p['parent_rate_count'] else 'extension')
    value=dict(campaign_hash=shared.digest(p),smoke=p['smoke'],selected=winners,selection_rule=p['selection'],
        parent_campaign_hash=p['parent_campaign_hash'],
        result_hashes={r['job']['id']:shared.digest(r) for r in rows})
    return shared.freeze_json(Path(output)/'selected_lrs.json',value) if write else value


@contextmanager
def worker_protocol():
    """Use the tested numerical worker with this frozen job/provenance provider."""
    import cifar_baselines_runtime as runtime
    original=runtime.protocol
    runtime.protocol=sys.modules[__name__]
    try: yield runtime
    finally: runtime.protocol=original


def report(p,output,arms=base.ARMS,include_archived_cap=False):
    data={j['id']:result(output,j) for j in jobs(p,output,'eval',arms)}
    if any(r['status']!='complete' for r in data.values()):
        raise RuntimeError('Diverged evaluation: cannot publish complete benchmark')
    per_seed=[]
    for model in base.MODELS:
        for arm in arms:
            for seed in p['report_seeds']:
                a=data[f'eval-{model}-{arm}-{seed}-a']['metrics']['plasticity']
                b=data[f'eval-{model}-{arm}-{seed}-b']['metrics']['retention']
                per_seed.append(dict(model=model,arm=arm,seed=seed,plasticity=a,retention=b,
                    balanced=min(a,b),provenance='new baseline campaign with extended LR search'))
    if include_archived_cap:
        if p['smoke']: raise ValueError('Never mix archived scientific results with smoke')
        ref=shared.read_json(base.CAP_REFERENCE)
        if ref['schema']!='cifar-cap-prefix-80-v1' or ref['eval_tasks']!=p['eval_tasks'] or ref['patch_size']!=p['patch_size']:
            raise RuntimeError('Archived CAP reference horizon/protocol mismatch')
        for model in base.MODELS:
            cfg=ref['sources'][model]['config']
            keys=('updates_per_task','batch_size','pool_size','normalization_mean','normalization_std',
                  'a_permutation_seed','b_permutation_seed','schedule_seed','test_subset_seed','test_examples')
            if any(cfg['streams'][k]!=p[k] for k in keys) or cfg['seeds']!=p['report_seeds']:
                raise RuntimeError('Archived CAP stream/data/seed protocol differs')
            if cfg['protocol']['architecture']!=model or cfg['protocol']['patch_size']!=p['patch_size']:
                raise RuntimeError('Archived CAP architecture differs')
        per_seed+=ref['per_seed']
    summary=[]
    for model,arm in sorted({(r['model'],r['arm']) for r in per_seed}):
        rows=[r for r in per_seed if r['model']==model and r['arm']==arm]
        summary.append(dict(model=model,arm=arm,seeds=len(rows),provenance=rows[0]['provenance'],
            **{k:mean(r[k] for r in rows) for k in ('plasticity','retention','balanced')},
            **{k+'_std':stdev(r[k] for r in rows) for k in ('plasticity','retention','balanced')}))
    directory=Path(output)/'analysis'
    shared.write_csv(directory/'per_seed.csv',per_seed,list(per_seed[0]))
    shared.write_csv(directory/'benchmark.csv',summary,list(summary[0]))
    shared.atomic_json(directory/'evaluation_manifest.json',dict(campaign_hash=shared.digest(p),smoke=p['smoke'],
        parent_campaign_hash=p['parent_campaign_hash'],result_hashes={k:shared.digest(r) for k,r in data.items()},
        archived_cap_hash=shared.file_hash(base.CAP_REFERENCE) if include_archived_cap else None))
    return summary
