"""Read-only single-task diagnostic; does not alter scientific results or checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from cifar_mech import protocol
from cifar_mech.constants import CAP
from cifar_mech.data import Data
from cifar_mech.model import make_model, parameter_groups
from cifar_mech.optim import CAPAdam
from cifar_mech.seed import seed_everything
from cifar_mech.streams import IndexSampler, permutation
from cifar_mech.checkpointing import source_version


@torch.no_grad()
def evaluate(model, x, y, perm, device):
    model.eval()
    correct, total_loss = 0, 0.0
    for start in range(0, len(y), 128):
        xb = x[start:start+128][:, perm].to(device)
        yb = y[start:start+128].to(device)
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total_loss += F.cross_entropy(logits, yb, reduction='sum').item()
    model.train()
    return correct/len(y), total_loss/len(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--patch-size', type=int, choices=[2, 4, 8, 16], default=4)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lrs', type=float, nargs='+', default=[CAP['lr'], .001, .0003])
    args = parser.parse_args()
    if args.steps < 1 or any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        parser.error('Use positive steps and finite positive learning rates')
    # Avoid accidental repeat of the dataset download.
    if not all((args.data_dir/'cifar-100-python'/f).is_file() for f in ('train', 'test', 'meta')):
        parser.error('Existing CIFAR-100 files not found; set --data-dir to the directory used by the run')
    torch.set_num_threads(1)
    protocol.configure(args.model, args.patch_size)
    data = Data(args.data_dir)
    perm = permutation('a', 1)  # Exactly the first Stream-A task.
    train_indices = torch.randperm(len(data.train_y), generator=torch.Generator().manual_seed(777))[:2000]
    train_x, train_y = data.train_x[train_indices], data.train_y[train_indices]
    print(json.dumps({'model': args.model, 'patch_size': args.patch_size, 'seed': args.seed,
                      'steps_per_cell': args.steps, 'learning_rates': args.lrs,
                      'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                      'device': args.device, 'gpu': torch.cuda.get_device_name(args.device)
                      if args.device.startswith('cuda') else None,
                      'scientific_source_sha256': source_version(), 'cap_baseline': CAP}, indent=2), flush=True)
    print('Training eval: fixed 2,000 training examples. Test eval: original fixed 2,000 test examples.', flush=True)
    print('Both accuracies are evaluated at the SAME step. Binding is cumulative; Adam has no boxes.', flush=True)
    print('Arm   | LR         | Step  | Train acc | Test acc | Train loss | Test loss | Binding', flush=True)
    rows, initial_hash, batch_hash = [], None, None
    for lr in args.lrs:
        for arm in ('cap', 'adam'):
            seed_everything(args.seed)
            model = make_model().to(args.device)
            h = hashlib.sha256()
            for p in model.parameters():
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            if initial_hash is None:
                initial_hash = h.hexdigest()
            assert h.hexdigest() == initial_hash, 'Initial weights differ between cells'
            groups = parameter_groups(model)
            opt = CAPAdam(groups, lr=lr) if arm == 'cap' else torch.optim.Adam(
                groups, lr=lr, betas=tuple(CAP['betas']), eps=CAP['eps'],
                weight_decay=0.0, amsgrad=False, foreach=False, fused=False)
            sampler = IndexSampler(len(data.train_y), args.seed, 128)
            batches = hashlib.sha256()
            bind_sum = 0.0
            for step in range(1, args.steps+1):
                indices = sampler.next_indices()
                batches.update(indices.numpy().tobytes())
                x, y = data.batch(indices, perm, args.device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss: {arm}, lr={lr}, step={step}')
                loss.backward()
                opt.step()
                if arm == 'cap':
                    bind_sum += opt.last_bind_rate
                if step % 1000 == 0 or step == args.steps:
                    train_acc, train_loss = evaluate(model, train_x, train_y, perm, args.device)
                    test_acc, test_loss = evaluate(model, data.test_x, data.test_y, perm, args.device)
                    binding = bind_sum/step if arm == 'cap' else None
                    row = dict(arm=arm, lr=lr, step=step, train_accuracy=train_acc,
                               test_accuracy=test_acc, train_loss=train_loss, test_loss=test_loss,
                               mean_binding=binding)
                    rows.append(row)
                    label = f'{binding:.2%}' if binding is not None else 'n/a'
                    print(f'{arm:5} | {lr:.7f} | {step:5} | {train_acc:9.2%} | {test_acc:8.2%} | '
                          f'{train_loss:10.4f} | {test_loss:9.4f} | {label}', flush=True)
            if batch_hash is None:
                batch_hash = batches.hexdigest()
            assert batches.hexdigest() == batch_hash, 'Minibatches differ between cells'
            del model, opt, groups
    print('\nMatched initial weights and minibatch sequences verified for every cell.', flush=True)
    print('Diagnostic only: one seed, one task. Test results are not an unbiased tuning-validation score.', flush=True)
    print('RESULTS_JSON=' + json.dumps(rows, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
