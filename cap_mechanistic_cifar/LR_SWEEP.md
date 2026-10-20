# Single-task CAP learning-rate pilot

`sweep_learning_rates.py` selects a learning rate for each architecture and
writes `selected_lrs.json`. `run_all.sh` requires that completed selection.
The launcher adds `lrsel_<selection-hash>` to the output root, followed by
`mlp/scientific` or `cnn/scientific`. Checkpoint resume requires matching source
and protocol identities.

Both models use CAP, seed 42, Stream A task 1, default 4x4-pixel patches,
5,000 updates per candidate. Ten rates are tested separately per architecture:
`1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 1/300, 1e-2`.
All other CAP settings remain fixed. This is 20 single-task training runs.

The script splits the original CIFAR training data into 45,000 training and
5,000 validation examples (50/class), deterministically with split seed 777.
It never opens the CIFAR test split. Candidates within an architecture share
initialization and minibatch sequences. The selected LR maximizes validation
accuracy at the final 5,000th update; ties favor lower validation loss and then
lower LR. Intermediate evaluations are informational, not early stopping.
Nonfinite candidates are recorded and excluded; other errors stop execution.

Completed candidate results are cached under `lr_sweep/<config-hash>/`. Rerunning
the same command reuses them; an interrupted candidate starts over. Partial
sweeps never produce a new selection file. The selected JSON preserves all rates,
outcomes, split/dataset hashes, environment and selection rule. Changing protocol
or source requires a new sweep, not hand-editing the selected rate.

## Bash

From the repository root, download CIFAR-100 once, then run the pilot and
selected-rate suite:

```bash
python -c "from torchvision.datasets import CIFAR100; CIFAR100('data', train=True, download=True)"
python run_experiments.py cifar-lr-sweep --device cuda
python run_experiments.py mechanistic-cifar-selected --device cuda --resume
```

For a non-default patch size, use the same `--patch-size` during the sweep and
`PATCH_SIZE=... bash run_all.sh` afterwards. The launcher rejects mismatches.
Use `LR_SELECTION=/path/to/selection.json` for an explicitly named selection file.
Direct CLI commands need `--lr-selection selected_lrs.json` to use the selected
rates and matching output paths. `--dry-run` shows the effective LR per model.

Example result path:
`../results/mechanistic_cifar/lrsel_<hash>/mlp/scientific/`.
Main experiments train on the full original 50,000 training examples after
selection; they do not retain the pilot's 45,000-example restriction.

## Interpretation and checks

This is a one-seed, one-task pilot selecting single-task validation performance,
not a guarantee of best continual-learning retention or best behavior on later
tasks. It does not tune k or timescales. Earlier diagnostics inspected test
accuracy, so the combined development process includes exploratory comparisons;
the pilot's train-only split does not make that earlier process a held-out
test-set evaluation.

See [VALIDATION.md](../docs/VALIDATION.md) for test commands and execution status.
