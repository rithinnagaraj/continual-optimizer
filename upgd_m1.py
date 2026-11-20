"""Frozen protocol and evidence checks for the UPGD M1 follow-up (no training)."""
from pathlib import Path
import math
from statistics import mean, stdev
import upgd_rebuttal as parent

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'results/upgd_m1'
LAYERS = ('network.1.weight', 'network.3.weight', 'network.5.weight')
SHAPES = ((300, 784), (300, 300), (10, 300))


def settings(smoke=False):
    return dict(version='upgd-m1-v1', smoke=smoke, seeds=parent.REPORT_SEEDS,
                fraction=.01, layers=list(LAYERS), domains=2 if smoke else 40,
                checkpoint_task=4 if smoke else 200, fisher_examples=10 if smoke else 200,
                fisher_subset_seed=510000, fisher_microbatch=16,
                eval_batch_size=1000, noise_draws=1 if smoke else 3,
                noise_scale=.1, random_groups=2 if smoke else 20, probe_seed=410000,
                bootstrap_resamples=10000, bootstrap_seed=610000,
                utility='signed bias-corrected avg_utility; equivalent to pre-sigmoid protection ranking for a positive global maximum',
                fisher='frozen mean per-example squared cross-entropy gradients on stratified official-test subsets, all pool domains',
                primary_loss='mean cross-entropy over all pool domains, matching CAP M1',
                secondary_loss='mean cross-entropy over non-current pool domains',
                inference='average layers within seed; paired contrasts; bootstrap whole seeds')


def source_hashes():
    return {**parent.source_hashes(), **{name: parent.file_hash(ROOT/name) for name in
            ('upgd_m1.py', 'upgd_m1_runtime.py', 'experiment_upgd_m1.py')}}


def build_manifest(source):
    source = Path(source)
    p = parent.campaign(source)
    jobs = {j['seed']: j for j in parent.jobs(source, 'eval') if j['stream'] == 'b'}
    sources = {}
    identities, environments = [], []
    task = settings(p['smoke'])['checkpoint_task']
    for seed in p['report_seeds']:
        job = jobs[seed]
        r = parent.result(source, job)
        if r['status'] != 'complete':
            raise RuntimeError('M1 requires completed stream-B evaluation jobs')
        name = f'task_{task:03d}.pt'
        checkpoint = source/'jobs'/job['id']/name
        if parent.file_hash(checkpoint) != r['checkpoint_hashes'][name]:
            raise RuntimeError(f'Changed source checkpoint: {job["id"]}/{name}')
        sources[str(seed)] = dict(job=job, result_hash=parent.digest(r),
                                 checkpoint=name, checkpoint_hash=r['checkpoint_hashes'][name])
        identities.append(parent.read_json(checkpoint.parent/'data_identity.json'))
        environments.append(parent.read_json(checkpoint.parent/'environment.json'))
    if any(i != identities[0] for i in identities) or any(e != environments[0] for e in environments):
        raise RuntimeError('Source data/environment varies across seeds')
    return dict(**settings(p['smoke']), parent_campaign_hash=parent.digest(p),
                selected_config_hash=parent.digest(parent.read_json(source/'selected_config.json')),
                sources=sources, data_identity=identities[0], environment=environments[0],
                source_hashes=source_hashes())


def campaign(source, output, initialize=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use separate, non-nested source and M1 output directories')
    expected = build_manifest(source)
    path = output/'campaign.json'
    if initialize:
        return parent.freeze_json(path, expected)
    if parent.read_json(path) != expected:
        raise RuntimeError('M1 source/protocol changed; do not mix campaigns')
    return expected


def groups(p, seed):
    result = []
    for layer_id, (layer, shape) in enumerate(zip(LAYERS, SHAPES)):
        fixed = p['probe_seed']+seed*100000+p['checkpoint_task']*1000+layer_id*100
        specs = [(c, g, 0) for c in ('utility', 'fisher') for g in ('top', 'bottom')]
        specs += [('random', 'random', r) for r in range(p['random_groups'])]
        for comparator, group, rep in specs:
            result.append(dict(index=len(result), layer=layer, comparator=comparator,
                               group=group, replicate=rep, fraction=p['fraction'],
                               coordinate_count=max(1, int(math.prod(shape)*p['fraction'])),
                               random_seed=fixed+rep*10000))
    return result


def directory(output, seed):
    return Path(output)/'jobs'/f'm1-{seed}'


def checked_json(path, identity):
    value = parent.read_json(path)
    if value['identity'] != identity:
        raise RuntimeError(f'Changed evidence identity: {path}')
    return value


def identity(p, seed):
    return dict(campaign_hash=parent.digest(p), seed=seed,
                checkpoint_hash=p['sources'][str(seed)]['checkpoint_hash'])


def completed_rows(p, output, seed, required=False):
    cell = directory(output, seed)
    specs = groups(p, seed)
    paths = sorted((cell/'groups').glob('*.json'))
    if len(paths) > len(specs):
        raise RuntimeError('Unexpected extra groups')
    result = []
    for i, path in enumerate(paths):
        if path.name != f'{i:03d}.json':
            raise RuntimeError('Missing/nonsequential committed group')
        row = checked_json(path, identity(p, seed))
        if row['spec'] != specs[i]:
            raise RuntimeError('Committed group differs from protocol')
        if len(row['draw_damages_by_domain']) != p['noise_draws'] or any(
                len(d) != p['domains'] for d in row['draw_damages_by_domain']):
            raise RuntimeError('Incomplete perturbation draws')
        if not all(math.isfinite(v) for d in row['draw_damages_by_domain'] for v in d):
            raise RuntimeError('Nonfinite damage')
        result.append(row)
    if required and len(result) != len(specs):
        raise RuntimeError(f'Incomplete M1 seed {seed}: {len(result)}/{len(specs)} groups')
    return result


def verify_result(p, output, seed):
    cell = directory(output, seed)
    path = cell/'result.json'
    if not path.exists():
        return None
    result = checked_json(path, identity(p, seed))
    if result['status'] != 'complete' or result['smoke'] != p['smoke']:
        raise RuntimeError('Invalid completed result')
    rows = completed_rows(p, output, seed, required=True)
    if result['group_hashes'] != {f'{i:03d}.json': parent.digest(row) for i, row in enumerate(rows)}:
        raise RuntimeError('Changed committed groups')
    for name, expected in result['artifact_hashes'].items():
        if parent.file_hash(cell/name) != expected:
            raise RuntimeError(f'Changed M1 artifact: {name}')
    return result


def report(p, output):
    import numpy as np
    all_rows, seed_rows = [], []
    results = {}
    for seed in p['seeds']:
        result = verify_result(p, output, seed)
        if result is None:
            raise RuntimeError(f'M1 seed {seed} is incomplete')
        results[str(seed)] = parent.digest(result)
        baseline = checked_json(directory(output, seed)/'baseline.json', identity(p, seed))
        current = baseline['current_domain']
        for row in completed_rows(p, output, seed, required=True):
            for scope, domains in (('all_domains', list(range(p['domains']))),
                                   ('noncurrent_domains', [d for d in range(p['domains']) if d != current])):
                draws = [mean(draw[d] for d in domains) for draw in row['draw_damages_by_domain']]
                all_rows.append(dict(seed=seed, scope=scope, **row['spec'], damage=mean(draws)))
        for scope in ('all_domains', 'noncurrent_domains'):
            layer_values = []
            for layer in LAYERS:
                subset = [r for r in all_rows if r['seed']==seed and r['scope']==scope and r['layer']==layer]
                random = [r['damage'] for r in subset if r['comparator']=='random']
                values = {f'{r["comparator"]}_{r["group"]}': r['damage'] for r in subset if r['comparator']!='random'}
                for comp in ('utility', 'fisher'):
                    top, bottom = values[f'{comp}_top'], values[f'{comp}_bottom']
                    layer_values.append(dict(comparator=comp, top_damage=top, bottom_damage=bottom,
                        random_damage=mean(random), top_minus_random=top-mean(random),
                        top_minus_bottom=top-bottom,
                        top_percentile=100*mean(float(x<top)+.5*float(x==top) for x in random)))
            for comp in ('utility', 'fisher'):
                subset = [r for r in layer_values if r['comparator']==comp]
                seed_rows.append(dict(seed=seed, scope=scope, comparator=comp,
                                      **{k: mean(r[k] for r in subset) for k in subset[0] if k!='comparator'}))
    summaries = []
    for scope in ('all_domains', 'noncurrent_domains'):
        for comp in ('utility', 'fisher', 'utility_minus_fisher'):
            for metric in ('top_damage', 'bottom_damage', 'random_damage', 'top_minus_random', 'top_minus_bottom', 'top_percentile'):
                values = []
                for seed in p['seeds']:
                    rows = {r['comparator']: r for r in seed_rows if r['scope']==scope and r['seed']==seed}
                    values.append(rows['utility'][metric]-rows['fisher'][metric] if comp=='utility_minus_fisher' else rows[comp][metric])
                x = np.array(values)
                rng = np.random.default_rng(p['bootstrap_seed'])
                draws = rng.integers(0, len(x), (p['bootstrap_resamples'], len(x)))
                low, high = np.quantile(x[draws].mean(1), [.025, .975])
                summaries.append(dict(scope=scope, comparator=comp, metric=metric, mean=mean(values),
                                      std=stdev(values), ci_low=float(low), ci_high=float(high), n_seeds=len(values)))
    root = Path(output)/'analysis'
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (('group_damage', all_rows), ('per_seed', seed_rows), ('summary', summaries)):
        parent.write_csv(root/f'{name}.csv', rows, list(rows[0]))
    parent.atomic_json(root/'report.json', dict(campaign_hash=parent.digest(p), smoke=p['smoke'],
        result_hashes=results, summary=summaries,
        artifact_hashes={name: parent.file_hash(root/name) for name in ('group_damage.csv', 'per_seed.csv', 'summary.csv')}))
    return dict(smoke=p['smoke'], completed_seeds=len(results),
                summary=[r for r in summaries if r['scope']=='all_domains' and r['metric']=='top_minus_random'])
