# Mechanistic experiments

The MNIST and CIFAR suites are independently packaged inside this source tree.
The root launcher uses their local code directly, so editable installation is
optional. Run their unit tests separately to avoid similarly named test modules:

```bash
python -m pytest -q
cd cap_mechanistic
python -m pytest -q tests
python -m pytest -q diagnostics/test_validation_launcher.py
cd ../cap_mechanistic_cifar
python -m pytest -q tests test_lr_selection.py
cd ..
```

## MNIST M1–M6

```bash
python run_experiments.py mechanistic-mnist --smoke --device cpu
python run_experiments.py mechanistic-mnist --device cuda --resume
```

Those strict/smoke commands run reference → M3 → M2 → M4 → M1 → M5 → M6 → analysis.
For the reference and bounded-validation pipeline used by the table/figure
commands, run:

```bash
python run_experiments.py mechanistic-mnist-paper --device cuda --resume
```

This runs complete reference diagnostics, exact bounded implementation validation,
then M1–M6 and analysis. Discrepant historical endpoints retain their failed labels.
A failed bounded check blocks every mechanism. Both S1 streams evaluate in batches
of 128, as specified in Appendix A.
Smoke uses synthetic data and reduced horizons under
`results/mechanistic_mnist/smoke/`. Full outputs use
`results/mechanistic_mnist/`. All five seeds are retained in smoke for M5's nested
validation. M6 requires the earlier mechanisms.

| Mechanism | Question |
|---|---|
| M1 faithfulness | Does importance rank the damage caused by matched coordinate perturbations, relative to Fisher and magnitude? |
| M2 k dial | How do finite box-radius changes and disabling projection affect adaptation and retention? |
| M3 state causality | Which effects follow from removing the box, recentering anchors, resetting importance, or resetting CAP state while preserving Adam moments? |
| M4 assignment | Does the coordinate assignment of importance matter, compared with within-layer shuffles and reversal? |
| M5 prospective prediction | Do prespecified optimizer-state features predict next-task adaptation under nested leave-one-seed-out validation? |
| M6 release dial | How do release timescales, including infinity, affect bound importance and behavior? |

The scientific reference is five seeds, two 200-task MNIST streams, 5,000
updates/task, MLP300, batch 128. Stream B uses 40 fixed permutations in five
shuffled blocks. Historical P samples every 100th update after update 500000;
dense all-update P is stored separately. Historical R averages the final 39
non-current domains. See the packaged JSON configurations and
`cap_mechanistic/PROTOCOL_NOTES.md` for exact semantics.

The historical numerical gate is deliberately strict (`5e-4` absolute tolerance
for each seed/metric). The separate strict stage stops if it fails. Smoke deliberately
cannot pass it, and the launcher's smoke-only override preserves unverified
labels. Gate outcomes are computed from each reference run.

### Alternative bounded implementation validation

The diagnostics support a separately labeled admission policy when
historical endpoints differ across environments. A full reference bank must
already exist. From the repository root:

```bash
python cap_mechanistic/diagnostics/run_validated_mechanisms.py --output-dir results/mechanistic_mnist --data-dir data --device cuda --resume
```

The launcher checks pinned, bundled optimizer/model snapshots, all 45 reference
checkpoints and the recorded source/configuration identity. It compares 1,000
real-data updates for every seed and stream, plus 16 continuation updates from
each checkpoint. It runs mechanisms only after exact agreement. The certificate
does not change the failed historical gate or claim million-update equivalence;
reports retain their conservative unverified labels. A changed core source hash
is incompatible with this policy. See `cap_mechanistic/diagnostics/README.md`.

## CIFAR-100 M1–M4: MLP and CNN

```bash
python run_experiments.py mechanistic-cifar --smoke --device cpu
python run_experiments.py mechanistic-cifar --device cuda --resume
```

For the S2/S3 learning rates, run:

```bash
python run_experiments.py mechanistic-cifar-paper --device cuda --resume
```

This uses MLP lr=2e-4 and CNN lr=3e-4 directly, under
`results/mechanistic_cifar/paper_fixed/`. It does not rerun rate selection.

The full default `mechanistic-cifar` is fixed-configuration transfer of the final CAP settings.
It runs each architecture's own CIFAR reference bank, M1–M4 and analysis.
Input permutations rearrange 4×4 spatial patches identically in all RGB channels.
Architectures are analyzed separately, with 100 output classes and no BatchNorm
or dropout. Reference admission checks integrity/completeness, not a comparison
with historical MNIST endpoints or a CIFAR accuracy threshold.

Optional learning-rate pilot, performed separately for both architectures:

```bash
python -c "from torchvision.datasets import CIFAR100; CIFAR100('data', train=True, download=True)"
python run_experiments.py cifar-lr-sweep --device cuda
python run_experiments.py mechanistic-cifar-selected --device cuda --resume
```

The pilot uses seed 42, one 5,000-update task and a stratified train-only
45,000/5,000 split. Ten rates are evaluated for each architecture. It writes
`cap_mechanistic_cifar/selected_lrs.json`; selected-rate outputs receive a
selection-hash subdirectory, keeping them separate from fixed-rate outputs.
The main reference then trains on all 50,000 training examples. The pilot is
single-task selection and does not establish optimal continual retention.

Raw commands support `--model mlp`, `--model cnn`, individual M1–M4 commands,
`--patch-size` and `--dry-run`; run them from `cap_mechanistic_cifar/`.
Use the same patch size in the pilot and selected-rate experiment.

## Interpretation

Both suites use paired branches and whole-seed bootstrap inference. Coordinate
groups, perturbation repeats, shuffled assignments and checkpoints are not
independent seeds. Missing, mixed or unverified evidence remains labeled as such.
Synthetic smoke, unit tests and bounded numerical checks establish implementation
properties; they are not completed scientific experiments.
