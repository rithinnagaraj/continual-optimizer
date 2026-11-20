# Priority 2: native UPGD utility faithfulness

This follow-up addresses reviewer 2 important request 2: whether native utility
from another boundary-free optimizer identifies consequential weights, and how
CAP's inspectability relates to existing methods. It never trains a model.
The paper write-up will wait until the planned experiments are complete.

## Frozen scope and interpretation

Use the five selected UPGD stream-B checkpoints at task 200 (1M updates) from
priority 1. Compare rankings **within each UPGD host model**, rather than comparing
raw damage magnitudes between different optimizers' models. This first experiment
uses only the 1% fraction and final checkpoint. CAP's broader original M1 includes
additional fractions/checkpoint ages; do not describe the smaller follow-up as
a full replication of every CAP M1 cell.

| Component | Frozen choice |
|---|---|
| Reporting seeds | 42, 123, 451, 789, 2026 |
| Weight matrices | `network.1.weight`, `network.3.weight`, `network.5.weight`; biases excluded |
| Coordinates per group | 1% per matrix: 2,352 / 900 / 30 |
| Utility ranking | Signed, bias-corrected stored `avg_utility`; top and bottom |
| Fisher ranking | Frozen mean per-example squared cross-entropy gradients; top and bottom |
| Random control | 20 groups per matrix, sampled without replacement within each group |
| Noise | Three Gaussian draws, each evaluated with both positive and negative sign |
| Noise standard deviation | 0.1 × original weight matrix sample SD |
| Damage | Mean of positive/negative perturbed losses minus unperturbed loss |
| Primary loss | Equal-weight mean cross-entropy over all 40 pool domains |
| Secondary loss | Same evaluations restricted to the 39 non-current domains |
| Evaluation data | Priority-1 fixed official-test subset: 2,000 examples, seed 999 |
| Fisher samples | 200 examples/domain, stratified by label; seed 510000 + domain |
| Fisher microbatch | 16 per-example gradients (`torch.func.grad` / `vmap`) |
| Evaluation batch | 1,000, matching the existing CAP M1 evaluation batch size |
| Bootstrap | 10,000 whole-seed resamples, seed 610000; average matrices within seed |

The test subset and Fisher samples intentionally follow the existing CAP M1
diagnostic. Fisher uses true labels from a subset of these evaluation examples;
it is a frozen, retrospective diagnostic, **not online Fisher or an EWC training
baseline**, and is not an independent generalization estimate. No model or
hyperparameter is selected with these probe losses.

Utility is signed, not absolute. With the positive global utility maximum verified
in each checkpoint, bias-corrected native utility has the same ordering as UPGD's
pre-sigmoid protection score. Ranking before sigmoid avoids numerical saturation
ties. Stable coordinate-order tie breaking is used for utility and Fisher.

The perturbation seeds, layer scaling, three antithetic draws, stratification,
and 20 random controls follow the existing CAP M1 definitions. One group has
six complete pool evaluations. There are 72 groups per seed, 360 groups total,
and 2,160 complete pool evaluations plus baselines. There is no magnitude-ranking
arm in this minimal follow-up. Random groups and noise draws are shared across
comparators as in the existing protocol.

Report top-minus-random and top-minus-bottom damage for utility and Fisher;
also report paired utility-minus-Fisher contrasts. Layers, noise draws, and random
groups are repeated measurements, not independent seeds. Confidence intervals
resample the five seeds after averaging matrices. These intervals have the usual
limitations of only five independent seeds. The full-pool primary outcome matches
CAP M1; the non-current outcome is secondary and must not replace it post hoc.

## Source and runtime requirements

Use the same L4 machine and virtual environment as priority 1. No additional
packages are required beyond its existing torch, torchvision, and NumPy. The
source campaign remains `results/upgd_rebuttal`; M1 writes separately to
`results/upgd_m1`. Keep all priority-1 checkpoints and manifests in place.

Initialization verifies the frozen priority-1 protocol, search/confirmation
selection evidence, five complete stream-B evaluation results, and actual final
checkpoint hashes. It freezes a separate M1 manifest with source code, checkpoint,
environment, and data identities. Runtime rejects source/environment/data changes.
It does not update priority-1 checkpoints, utilities, or optimizer state.

The same environment includes Python/torch/torchvision/CUDA versions, device,
determinism, TF32, and thread settings. Do not upgrade packages between stages.
For a migrated machine use the matching environment and copied original files;
do not bypass an identity error.

## Commands: bounded L4 pilot first

From the existing clone, update and initialize. These commands do not run probes:

```bash
cd ~/continual-optimizer-rebuttal
git pull --ff-only &&
.venv/bin/python tools/audit_release.py &&
.venv/bin/python experiment_upgd_m1.py init
```

Run one bounded pilot on seed 42:

```bash
export CUDA_VISIBLE_DEVICES=0
nohup .venv/bin/python -u experiment_upgd_m1.py run \
  --seeds 42 --device cuda --resume --max-groups 6 \
  > results/l4_pilot/m1-pilot.log 2>&1 < /dev/null &
echo "M1 pilot process ID: $!"
```

This first builds the full Fisher diagnostic for this one checkpoint (40 domains
× 200 examples), then runs six groups on the first weight matrix: utility top/
bottom, Fisher top/bottom, and two random controls. It pauses without shortening
the scientific protocol. This is a work-count bound, not a wall-clock deadline.
No new training and no other seeds run. The pilot is part of the final experiment;
its completed work is reused on resume.

```bash
tail -n 20 results/l4_pilot/m1-pilot.log
.venv/bin/python experiment_upgd_m1.py status
```

Wait for seed 42 to show six committed groups and for its log to report `paused`.
Do not repeatedly rerun the bounded command just to check status: each invocation
with `--resume` processes six **additional** groups.

After the pilot has paused, package feedback:

```bash
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-m1-pilot-feedback.tar.gz \
  -C results upgd_m1 l4_pilot
```

Send `results/l4-m1-pilot-feedback.tar.gz` before launching the full run. It includes
the manifest, environment/data identities, per-domain Fisher timings, per-group
timings/damage, baseline losses, rankings, and probe seeds. It excludes binary
checkpoints/caches; keep those on the L4 machine for resume.

The previous **1–6 GPU-hour** full-run allowance is provisional. Estimate remaining
time from the measured baseline, Fisher-domain, and perturbation-group timings.
The pilot covers the largest matrix but only one checkpoint; allow headroom for
other seeds and groups. CPU setup/ranking/export overhead and GPU evaluation are
included in recorded attempt time; dependency import/admission checks before the
attempt clock and idle time are excluded.

## Full run, after pilot review

```bash
nohup .venv/bin/python -u experiment_upgd_m1.py run \
  --all --device cuda --resume \
  > results/l4_pilot/m1-full.log 2>&1 < /dev/null &
echo "M1 full-run process ID: $!"
```

After all five seeds show `complete`:

```bash
.venv/bin/python experiment_upgd_m1.py report &&
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-m1-feedback.tar.gz \
  -C results upgd_m1 l4_pilot
```

The report refuses incomplete seeds or changed completed artifacts. Inspect
`analysis/per_seed.csv`, `analysis/summary.csv`, and `analysis/report.json`.
`analysis/group_damage.csv` contains matrix-level results for both domain scopes.

## Recovery and validation

There is an OS lock per seed. Run one worker per L4; never share an active seed
between workers. Separate output directories prevent writes into source runs.
Fisher contributions commit after each domain; perturbations commit after each
complete group. An interruption repeats only the uncommitted unit. Each group
uses an independent deterministic RNG seed. Model parameters are restored in a
`finally` block even when a perturbed evaluation fails. The complete job checks
exact model restoration, records artifact hashes, and can be verified/skipped.
Cached tensor hashes are checked before loading.

Development validation uses synthetic data and temporary CPU checkpoints:

```bash
python -m unittest tests.test_upgd_m1 -q
```

The tests check true per-example Fisher against explicit single-example gradients,
noise agreement with the existing CAP M1 helper, signed ranking, exact pause/resume
equivalence, interrupted-group recovery, immutable source checkpoints, full five-seed
report generation, and rejection of altered caches, manifests, and environments.
Synthetic outputs must not be used in the paper. Real L4 compatibility and timing
must be checked by the bounded pilot above.

On 20 November 2026, all 28 tests across `tests.test_upgd_m1`,
`tests.test_upgd_rebuttal`, and `tests.test_public_release` passed on the local
Python 3.14.6 / PyTorch 2.13.0+cpu environment. Nine of these tests exercise the
new M1 implementation. The release audit passed, and priority 1's frozen
protocol/source identity was verified unchanged. No production probes were run.
