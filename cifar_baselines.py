"""Frozen CIFAR-100 baseline protocol, bounded timing jobs, and reporting gates."""
from pathlib import Path
from statistics import mean, stdev
import math
import upgd_rebuttal as shared

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/'results/cifar_baselines'
MODELS = ('mlp', 'cnn')
ARMS = ('adamw', 'upgd')
RATES = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 1/300, 1e-2]
CAP_REFERENCE = ROOT/'docs/cifar_cap_reference/prefix_80.json'


def source_hashes():
    names = ['cifar_baselines.py', 'cifar_baselines_runtime.py', 'experiment_cifar_baselines.py',
             'upgd_rebuttal.py', 'upgd_rebuttal_runtime.py', 'model.py', 'cap_adam.py', shared.UPGD_SOURCE,
             'docs/cifar_cap_reference/prefix_80.json']
    paths = [ROOT/name for name in names]
    for tree in ('cap_mechanistic/cap_mech', 'cap_mechanistic_cifar/cifar_mech'):
        paths += [p for p in sorted((ROOT/tree).rglob('*')) if p.suffix in ('.py', '.json')]
    hashes = {p.relative_to(ROOT).as_posix():shared.file_hash(p) for p in paths}
    if hashes[shared.UPGD_SOURCE] != shared.UPGD_SHA256:
        raise RuntimeError('Vendored UPGD source changed')
    return hashes


def protocol(smoke=False):
    return dict(version='cifar-baselines-v1', smoke=smoke, models=list(MODELS), arms=list(ARMS),
        report_seeds=shared.REPORT_SEEDS, development_seed=31415, split_seed=777,
        learning_rates=RATES, selection='highest final single-task validation accuracy; then lowest loss, then lowest LR',
        search_updates=8 if smoke else 5000, eval_tasks=4 if smoke else 80,
        updates_per_task=4 if smoke else 5000, pilot_updates=8 if smoke else 1000,
        observation_interval=2 if smoke else 100, validation_interval=4 if smoke else 1000,
        batch_size=4 if smoke else 128, eval_batch_size=8 if smoke else 128,
        pool_size=2 if smoke else 40, patch_size=4,
        holdout_examples=20 if smoke else 5000, test_examples=20 if smoke else 2000,
        test_subset_seed=999, pilot_subset_seed=998,
        normalization_mean=[.5071,.4867,.4408], normalization_std=[.2675,.2565,.2761],
        a_permutation_seed=100000, b_permutation_seed=200001, schedule_seed=555,
        adamw=dict(betas=[.9,.999], eps=1e-8, weight_decay=.01),
        upgd=dict(sigma=1e-4, beta_utility=.9999, weight_decay=0.),
        cap_learning_rates=dict(mlp=.0002, cnn=.0003),
        pilot_learning_rates=dict(adamw=.0003, upgd=.001),
        sampler='persistent-worker RandomSampler semantics; short batches retained; no task reset',
        plasticity='stream A pre-update minibatch accuracy every 100 updates over tasks 41-80 (production)',
        retention='final stream B held-out accuracy averaged over 39 non-current domains (production)',
        balanced='min(P,R) per seed, then mean and sample SD',
        tuning_scope='LR only; train-only validation; no test-based selection; no automatic grid expansion',
        source_hashes=source_hashes())


def campaign(output, initialize=False, smoke=False):
    path=Path(output)/'campaign.json'
    if initialize:
        return shared.freeze_json(path, protocol(smoke))
    p=shared.read_json(path)
    if p != protocol(p['smoke']):
        raise RuntimeError('Frozen CIFAR protocol/source changed; use a separate campaign')
    return p


def spec(p, stage, model, arm, seed=None, stream=None, lr_index=None, lr=None):
    if model not in MODELS or arm not in (*ARMS, 'cap'):
        raise ValueError('Unknown model or optimizer')
    if stage=='search':
        name=f'search-{model}-{arm}-{lr_index:02d}'
        seed,stream,lr=p['development_seed'],'a',p['learning_rates'][lr_index]
        tasks,task_updates=1,p['search_updates']
    elif stage=='pilot':
        name=f'pilot-{model}-{arm}'
        seed,stream,lr=p['development_seed'],'b',p['pilot_learning_rates'][arm]
        tasks,task_updates=1,p['pilot_updates']
    elif stage=='eval':
        name=f'eval-{model}-{arm}-{seed}-{stream}'
        tasks,task_updates=p['eval_tasks'],p['updates_per_task']
    else:
        raise ValueError('Unknown stage')
    return dict(id=name,stage=stage,model=model,arm=arm,seed=seed,stream=stream,lr=lr,
                lr_index=lr_index,tasks=tasks,task_updates=task_updates,updates=tasks*task_updates,
                campaign_hash=shared.digest(p))


def jobs(p, output, stage, arms=ARMS):
    if stage=='pilot':
        return [spec(p,stage,m,a) for m in MODELS for a in ARMS]
    if stage=='search':
        return [spec(p,stage,m,a,lr_index=i) for m in MODELS for a in ARMS for i in range(len(RATES))]
    if stage!='eval':
        raise ValueError('Unknown stage')
    selected=select(p,output,write=False)
    if shared.read_json(Path(output)/'selected_lrs.json')!=selected:
        raise RuntimeError('Frozen LR selection differs from complete search evidence')
    return [spec(p,stage,m,a,seed=s,stream=t,
                 lr=p['cap_learning_rates'][m] if a=='cap' else selected['selected'][f'{m}-{a}']['lr'])
            for m in MODELS for a in arms for s in p['report_seeds'] for t in ('a','b')]


def result(output, job, required=True):
    cell=Path(output)/'jobs'/job['id']
    path=cell/'result.json'
    if not path.exists():
        if required: raise RuntimeError(f'Missing terminal result: {job["id"]}')
        return None
    r=shared.read_json(path)
    if r['job']!=job or r['status'] not in ('complete','diverged'):
        raise RuntimeError('Result job/status mismatch')
    if r['smoke'] != shared.read_json(Path(output)/'campaign.json')['smoke']:
        raise RuntimeError('Smoke/production result mismatch')
    if r['status']=='complete' and r['step']!=job['updates']:
        raise RuntimeError('Shortened training horizon')
    for name in ('latest.pt','tasks.csv','scalars.csv','pool.csv','validation.csv','environment.json','data_identity.json'):
        if shared.file_hash(cell/name)!=r['artifact_hashes'][name]:
            raise RuntimeError(f'Changed result artifact: {job["id"]}/{name}')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in r['metrics'].values()):
        raise RuntimeError('Invalid result metrics')
    return r


def select(p,output,write=True):
    rows=[result(output,j) for j in jobs(p,output,'search')]
    winners={}
    for model in MODELS:
        for arm in ARMS:
            candidates=[r for r in rows if r['job']['model']==model and r['job']['arm']==arm and r['status']=='complete']
            if not candidates: raise RuntimeError(f'All learning rates diverged for {model}/{arm}')
            candidates.sort(key=lambda r:(-r['metrics']['validation_accuracy'],r['metrics']['validation_loss'],r['job']['lr']))
            best=candidates[0]
            winners[f'{model}-{arm}']=dict(lr=best['job']['lr'],lr_index=best['job']['lr_index'],
                metrics=best['metrics'],at_grid_boundary=best['job']['lr_index'] in (0,len(RATES)-1))
    value=dict(campaign_hash=shared.digest(p),smoke=p['smoke'],selected=winners,
               selection_rule=p['selection'],result_hashes={r['job']['id']:shared.digest(r) for r in rows})
    return shared.freeze_json(Path(output)/'selected_lrs.json',value) if write else value


def pilot_report(p,output):
    rows=[]
    for job in jobs(p,output,'pilot'):
        r=result(output,job)
        if r['status']!='complete': raise RuntimeError('Pilot diverged; inspect before projection')
        events=shared.read_events(Path(output)/'jobs'/job['id']/'timing.jsonl')
        blocks=[e for e in events if e['event']=='checkpoint' and e.get('block_updates',0)>0]
        # Ignore the first block's startup kernels when later blocks exist.
        steady=blocks[1:] or blocks
        rate=sum(e['block_training_seconds'] for e in steady)/sum(e['block_updates'] for e in steady)
        single=r['pilot_single_domain_seconds']; pool=r['evaluation_seconds']
        a=rate*p['eval_tasks']*p['updates_per_task']+p['eval_tasks']*single
        b=rate*p['eval_tasks']*p['updates_per_task']+p['eval_tasks']*pool
        tuning=len(RATES)*(rate*p['search_updates']+single*(p['holdout_examples']/p['test_examples'])*(p['search_updates']//p['validation_interval']))
        rows.append(dict(model=job['model'],arm=job['arm'],training_ms_per_update=rate*1000,
            pool_evaluation_seconds=pool,single_domain_seconds=single,
            peak_cuda_allocated_mib=r['peak_cuda_allocated_bytes']/2**20,
            recorded_pilot_seconds=r['timing']['observed_wall_seconds'],
            projected_lr_search_hours=tuning/3600,projected_eval_a_hours=a/3600,
            projected_eval_b_hours=b/3600,projected_five_seed_eval_hours=5*(a+b)/3600))
    value=dict(campaign_hash=shared.digest(p),smoke=p['smoke'],rows=rows,
        projected_baseline_search_hours=sum(r['projected_lr_search_hours'] for r in rows),
        projected_baseline_evaluation_hours=sum(r['projected_five_seed_eval_hours'] for r in rows),
        caveats=['First task and provisional pilot LRs only; not selected scientific results.',
                 'Extrapolation excludes future checkpoint/setup/admission overhead; allow headroom.',
                 'Pilot uses 45k train and 2k train-holdout images; scientific evaluation uses 50k train and 2k official-test images.'])
    shared.atomic_json(Path(output)/'analysis/pilot_report.json',value)
    return value


def report(p,output,arms=ARMS,include_archived_cap=False):
    data={j['id']:result(output,j) for j in jobs(p,output,'eval',arms)}
    if any(r['status']!='complete' for r in data.values()):
        raise RuntimeError('Diverged evaluation: cannot publish complete benchmark')
    per_seed=[]
    for model in MODELS:
        for arm in arms:
            for seed in p['report_seeds']:
                a=data[f'eval-{model}-{arm}-{seed}-a']['metrics']['plasticity']
                b=data[f'eval-{model}-{arm}-{seed}-b']['metrics']['retention']
                per_seed.append(dict(model=model,arm=arm,seed=seed,plasticity=a,retention=b,balanced=min(a,b),provenance='new baseline campaign'))
    if include_archived_cap:
        if p['smoke']: raise ValueError('Never mix archived scientific results with smoke')
        ref=shared.read_json(CAP_REFERENCE)
        if ref['schema']!='cifar-cap-prefix-80-v1' or ref['eval_tasks']!=p['eval_tasks'] or ref['patch_size']!=p['patch_size']:
            raise RuntimeError('Archived CAP reference horizon/protocol mismatch')
        for model in MODELS:
            cfg=ref['sources'][model]['config']
            keys=('updates_per_task','batch_size','pool_size','normalization_mean','normalization_std',
                  'a_permutation_seed','b_permutation_seed','schedule_seed','test_subset_seed','test_examples')
            if any(cfg['streams'][k]!=p[k] for k in keys) or cfg['seeds']!=p['report_seeds']:
                raise RuntimeError('Archived CAP stream/data/seed protocol differs')
            if cfg['protocol']['architecture']!=model or cfg['protocol']['patch_size']!=p['patch_size']:
                raise RuntimeError('Archived CAP architecture differs')
        per_seed += ref['per_seed']
    summary=[]
    for model,arm in sorted({(r['model'],r['arm']) for r in per_seed}):
        rows=[r for r in per_seed if r['model']==model and r['arm']==arm]
        summary.append(dict(model=model,arm=arm,seeds=len(rows),provenance=rows[0]['provenance'],
            **{k:mean(r[k] for r in rows) for k in ('plasticity','retention','balanced')},
            **{k+'_std':stdev(r[k] for r in rows) for k in ('plasticity','retention','balanced')}))
    directory=Path(output)/'analysis';directory.mkdir(parents=True,exist_ok=True)
    shared.write_csv(directory/'per_seed.csv',per_seed,list(per_seed[0]))
    shared.write_csv(directory/'benchmark.csv',summary,list(summary[0]))
    shared.atomic_json(directory/'evaluation_manifest.json',dict(campaign_hash=shared.digest(p),smoke=p['smoke'],
        result_hashes={k:shared.digest(r) for k,r in data.items()},
        archived_cap_hash=shared.file_hash(CAP_REFERENCE) if include_archived_cap else None))
    return summary
