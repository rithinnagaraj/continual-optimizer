"""Read-only checkpoint probes with atomic domain/group commits and bounded work."""
import csv
import math
import os
from pathlib import Path
import time
import uuid

import torch
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F

import upgd_rebuttal as parent
import upgd_rebuttal_runtime as training
import upgd_m1 as protocol
from model import MLP300
from cap_mechanistic.cap_mech.data import stratified_indices


def native_utility(saved, model):
    """Use signed, bias-corrected native utility, avoiding sigmoid rounding ties."""
    names = [name for name, _ in model.named_parameters()]
    if saved['parameter_names'] != names:
        raise RuntimeError('Checkpoint parameter order changed')
    params = dict(model.named_parameters())
    values, steps, raw_maxima = {}, [], []
    visited = []
    for group in saved['optimizer']['param_groups']:
        if len(group['names']) != len(group['params']):
            raise RuntimeError('Optimizer name/parameter mismatch')
        for name, pid in zip(group['names'], group['params']):
            visited.append(name)
            state = saved['optimizer']['state'][pid]
            raw = state['avg_utility']
            if name not in params or raw.shape != params[name].shape or not torch.isfinite(raw).all():
                raise RuntimeError('Invalid native utility tensor')
            step = state['step']
            if step != saved['progress']['step']:
                raise RuntimeError('Utility is not from the final training update')
            correction = 1-group['beta_utility']**step
            if correction <= 0:
                raise RuntimeError('Invalid utility bias correction')
            values[name] = raw.detach().clone()/correction
            raw_maxima.append(raw.max().item())
            steps.append(step)
    if visited != names or max(raw_maxima) <= 0:
        raise RuntimeError('Cannot establish native protection ranking')
    return {name: values[name] for name in protocol.LAYERS}, dict(
        global_raw_max=max(raw_maxima), optimizer_steps=steps,
        ranking='signed bias-corrected avg_utility; positive common global scaling is order-preserving')


def fisher_domain(model, x, y, permutation, indices, denominator, device, microbatch):
    """True per-example gradient squares, not square of a batch mean gradient."""
    params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    out = {name: torch.zeros_like(params[name]) for name in protocol.LAYERS}
    def loss(parameters, xb, yb):
        logits = functional_call(model, (parameters, buffers), (xb.unsqueeze(0),))
        return F.cross_entropy(logits, yb.unsqueeze(0))
    gradient = vmap(grad(loss), in_dims=(None, 0, 0))
    for start in range(0, len(indices), microbatch):
        idx = indices[start:start+microbatch]
        gradients = gradient(params, x[idx][:, permutation].to(device), y[idx].to(device))
        for name in out:
            out[name].add_(gradients[name].detach().square().sum(0)/denominator)
    if any(not torch.isfinite(v).all() for v in out.values()):
        raise RuntimeError('Nonfinite Fisher estimate')
    return {k: v.cpu() for k, v in out.items()}


@torch.no_grad()
def evaluate(model, x, y, permutations, device, batch_size):
    losses, accuracies = [], []
    for domain in range(len(permutations)):
        total, correct = 0., 0
        for start in range(0, len(y), batch_size):
            labels = y[start:start+batch_size].to(device)
            logits = model(x[start:start+batch_size][:, permutations[domain]].to(device))
            total += F.cross_entropy(logits, labels, reduction='sum').item()
            correct += int((logits.argmax(1) == labels).sum().item())
        losses.append(total/len(y))
        accuracies.append(correct/len(y))
    if not all(math.isfinite(v) for v in losses):
        raise RuntimeError('Nonfinite probe loss')
    return dict(losses=losses, accuracies=accuracies)


@torch.no_grad()
def perturb(parameter, indices, loss_fn, baseline, seed, draws, scale):
    original = parameter.detach().clone()
    generator = torch.Generator().manual_seed(seed)
    sigma = scale*original.std().item()
    damages, squared_norms = [], []
    try:
        for _ in range(draws):
            noise = torch.zeros_like(parameter).flatten()
            noise[indices] = torch.randn(len(indices), generator=generator).to(parameter.device)*sigma
            noise = noise.view_as(parameter)
            parameter.copy_(original+noise)
            plus = loss_fn()['losses']
            parameter.copy_(original-noise)
            minus = loss_fn()['losses']
            damages.append([(a+b)/2-c for a, b, c in zip(plus, minus, baseline, strict=True)])
            squared_norms.append(float(noise.square().sum().item()))
    finally:
        parameter.copy_(original)
    if not all(math.isfinite(v) for d in damages for v in d):
        raise RuntimeError('Nonfinite perturbation damage')
    return dict(draw_damages_by_domain=damages, sigma=sigma, draw_squared_noise_norms=squared_norms)


def selected_indices(spec, scores, n):
    if spec['comparator'] == 'random':
        g = torch.Generator().manual_seed(spec['random_seed'])
        return torch.randperm(n, generator=g)[:spec['coordinate_count']]
    order = torch.argsort(scores[spec['comparator']][spec['layer']].flatten(), stable=True)
    return order[-spec['coordinate_count']:] if spec['group']=='top' else order[:spec['coordinate_count']]


def write_rankings(path, scores):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['layer', 'coordinate', 'utility_rank', 'fisher_rank'])
        for name in protocol.LAYERS:
            ranks = {k: torch.argsort(torch.argsort(v[name].flatten(), stable=True), stable=True).tolist()
                     for k, v in scores.items()}
            for coordinate, (u, f) in enumerate(zip(ranks['utility'], ranks['fisher'], strict=True)):
                writer.writerow([name, coordinate, u, f])
    os.replace(temporary, path)


def run_job(p, source, output, seed, *, data_dir, device='cuda', resume=False,
            max_groups=None, max_fisher_domains=None):
    if seed not in p['seeds'] or any(v is not None and v <= 0 for v in (max_groups, max_fisher_domains)):
        raise ValueError('Invalid seed or work bound')
    cell = protocol.directory(output, seed)
    with parent.file_lock(cell/'worker.lock'):
        existing = protocol.verify_result(p, output, seed)
        if existing:
            if not resume:
                raise FileExistsError('Completed job; use --resume to verify and skip')
            print(f'm1-{seed}: verified complete; skipped', flush=True)
            return existing
        if (cell/'job.json').exists() and not resume:
            raise FileExistsError('Partial job; use --resume')
        ident = protocol.identity(p, seed)
        parent.freeze_json(cell/'job.json', ident)
        device = torch.device(device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        training.seed_everything(seed)
        env = training.environment(device)
        if env != p['environment']:
            raise RuntimeError('Use the same environment/device as the source checkpoint')
        parent.freeze_json(cell/'environment.json', env)
        attempt, start = uuid.uuid4().hex, time.perf_counter()
        def event(kind, **values):
            training.synchronize(device)
            parent.append_event(cell/'timing.jsonl', dict(attempt=attempt, event=kind,
                unix_time=time.time(), elapsed_seconds=time.perf_counter()-start, **values))
        event('start', max_groups=max_groups, max_fisher_domains=max_fisher_domains)
        try:
            parent_plan = parent.campaign(source)
            item = p['sources'][str(seed)]
            checkpoint = Path(source)/'jobs'/item['job']['id']/item['checkpoint']
            if parent.file_hash(checkpoint) != item['checkpoint_hash']:
                raise RuntimeError('Source checkpoint changed')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['identity'] != dict(job=item['job'], environment=env, data=p['data_identity']):
                raise RuntimeError('Checkpoint identity mismatch')
            if saved['progress']['status'] != 'complete' or saved['progress']['step'] != item['job']['updates']:
                raise RuntimeError('Incomplete source checkpoint')
            model = MLP300(784, 10).to(device)
            model.load_state_dict(saved['model'])
            model.eval()
            utility, utility_info = native_utility(saved, model)
            tx, ty, x, y, data_id, train_idx, eval_idx = training.load_data(parent_plan, 'eval', data_dir)
            del tx, ty
            if data_id != p['data_identity'] or not torch.equal(eval_idx, saved['evaluation_indices']) or not torch.equal(train_idx, saved['train_indices']):
                raise RuntimeError('Probe dataset differs from source evaluation')
            parent.freeze_json(cell/'data_identity.json', data_id)
            schedule, permutations = training.stream_spec(parent_plan, item['job'])
            if saved['schedule'] != schedule or len(permutations) != p['domains']:
                raise RuntimeError('Stream mismatch')
            current = schedule[-1]
            baseline_path = cell/'baseline.json'
            loss_fn = lambda: evaluate(model, x, y, permutations, device, p['eval_batch_size'])
            if baseline_path.exists():
                baseline = protocol.checked_json(baseline_path, ident)
            else:
                before = time.perf_counter()
                baseline = dict(identity=ident, current_domain=current, **loss_fn())
                historical = {r['domain']:r['accuracy'] for r in saved['progress']['pool'] if r['task']==p['checkpoint_task']}
                # Different inference batch sizes can move an example at a decision boundary.
                if any(abs(baseline['accuracies'][d]-historical[d]) > 1/len(y)+1e-12 for d in permutations):
                    raise RuntimeError('Source accuracy reproduction differs by more than one example per domain')
                parent.atomic_json(baseline_path, baseline)
                event('baseline', seconds=time.perf_counter()-before)
            fisher_root = cell/'fisher'
            fisher_root.mkdir(exist_ok=True)
            fisher = {name: torch.zeros_like(v) for name, v in utility.items()}
            fresh_domains = 0
            for domain in range(p['domains']):
                tensor_path = fisher_root/f'{domain:03d}.pt'
                meta_path = fisher_root/f'{domain:03d}.json'
                indices = stratified_indices(y, p['fisher_examples'], p['fisher_subset_seed']+domain)
                if meta_path.exists():
                    meta = protocol.checked_json(meta_path, ident)
                    if meta['domain'] != domain or meta['indices_hash'] != training.tensor_hash(indices) or parent.file_hash(tensor_path)!=meta['tensor_hash']:
                        raise RuntimeError('Changed Fisher cache')
                    contribution = torch.load(tensor_path, map_location='cpu', weights_only=True)
                else:
                    before = time.perf_counter()
                    contribution = fisher_domain(model, x, y, permutations[domain], indices,
                        p['domains']*p['fisher_examples'], device, p['fisher_microbatch'])
                    training.atomic_checkpoint(tensor_path, contribution)
                    parent.atomic_json(meta_path, dict(identity=ident, domain=domain,
                        indices_hash=training.tensor_hash(indices), tensor_hash=parent.file_hash(tensor_path)))
                    fresh_domains += 1
                    event('fisher_domain', domain=domain, seconds=time.perf_counter()-before)
                    print(f'm1-{seed}: Fisher {domain+1}/{p["domains"]}', flush=True)
                for name in fisher:
                    value = contribution[name]
                    if value.shape != fisher[name].shape or not torch.isfinite(value).all() or (value<0).any():
                        raise RuntimeError('Invalid cached Fisher')
                    fisher[name].add_(value)
                if max_fisher_domains is not None and fresh_domains >= max_fisher_domains and domain < p['domains']-1:
                    event('paused', stage='fisher', committed_domains=domain+1)
                    return dict(status='paused', stage='fisher', committed_domains=domain+1)
            scores = dict(utility=utility, fisher=fisher)
            score_path, meta_path = cell/'scores.pt', cell/'scores.json'
            if meta_path.exists():
                meta = protocol.checked_json(meta_path, ident)
                if parent.file_hash(score_path)!=meta['tensor_hash'] or parent.file_hash(cell/'rankings.csv')!=meta['rankings_hash']:
                    raise RuntimeError('Changed scores/rankings cache')
                cached = torch.load(score_path, map_location='cpu', weights_only=True)
                if any(not torch.equal(scores[c][n], cached[c][n]) for c in scores for n in protocol.LAYERS):
                    raise RuntimeError('Scores no longer match source/cache')
            else:
                training.atomic_checkpoint(score_path, scores)
                write_rankings(cell/'rankings.csv', scores)
                parent.atomic_json(meta_path, dict(identity=ident, tensor_hash=parent.file_hash(score_path),
                    rankings_hash=parent.file_hash(cell/'rankings.csv'), native_utility=utility_info))
            before_rows = protocol.completed_rows(p, output, seed)
            group_root = cell/'groups'
            group_root.mkdir(exist_ok=True)
            specs = protocol.groups(p, seed)
            params = dict(model.named_parameters())
            for spec in specs[len(before_rows):]:
                before = time.perf_counter()
                parameter = params[spec['layer']]
                indices = selected_indices(spec, scores, parameter.numel()).to(device)
                values = perturb(parameter, indices, loss_fn, baseline['losses'], spec['random_seed'], p['noise_draws'], p['noise_scale'])
                row = dict(identity=ident, spec=spec, **values,
                           selected_indices_hash=training.tensor_hash(indices.cpu()))
                parent.atomic_json(group_root/f'{spec["index"]:03d}.json', row)
                event('group', index=spec['index'], layer=spec['layer'], comparator=spec['comparator'],
                      group=spec['group'], seconds=time.perf_counter()-before)
                print(f'm1-{seed}: group {spec["index"]+1}/{len(specs)} {spec["layer"]} {spec["comparator"]}/{spec["group"]}', flush=True)
                if max_groups is not None and spec['index']+1-len(before_rows)>=max_groups and spec['index']+1<len(specs):
                    event('paused', stage='groups', committed_groups=spec['index']+1)
                    return dict(status='paused', stage='groups', committed_groups=spec['index']+1)
            if any(not torch.equal(v.detach().cpu(), saved['model'][name]) for name, v in model.state_dict().items()):
                raise RuntimeError('Probe failed to restore source model exactly')
            rows = protocol.completed_rows(p, output, seed, required=True)
            event('complete', groups=len(rows))
            artifacts = ['baseline.json', 'scores.pt', 'scores.json', 'rankings.csv', 'environment.json', 'data_identity.json']
            artifacts += [f'fisher/{d:03d}.{ext}' for d in range(p['domains']) for ext in ('pt', 'json')]
            result = dict(identity=ident, status='complete', smoke=p['smoke'],
                group_hashes={f'{i:03d}.json':parent.digest(row) for i, row in enumerate(rows)},
                artifact_hashes={name:parent.file_hash(cell/name) for name in artifacts},
                timing=parent.timing_summary(cell/'timing.jsonl'))
            parent.atomic_json(cell/'result.json', result)
            return result
        except BaseException as error:
            event('error', error_type=type(error).__name__, message=str(error))
            raise
