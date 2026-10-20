# CAP-Adam mechanistic experiments

This package implements the MNIST reference and mechanisms M1–M6 under
`cap-mechanisms-final-v1`. It can run independently after installing its declared
dependencies. Commands below start in this directory.

## Install and validate

```bash
python -m pip install -e .
python -m pytest -q tests
python validation/run_smoke.py my-smoke
```

Python >=3.10, PyTorch, torchvision, NumPy, pandas, matplotlib, and pytest are
required. Spearman correlation uses average tied ranks and NumPy; SciPy and
statsmodels are unnecessary. MNIST is downloaded by torchvision only for real
runs. Synthetic smoke tests do not download data.

The isolation test copies the package into a temporary directory, clears inherited
`PYTHONPATH`, installs an import blocker for external project modules,
and runs the unit tests in a new process. A separate AST check rejects banned
imports. External dependencies remain available as installed dependencies.

## Validation scope

Unit, isolation, continuation, and smoke tests exercise implementation properties.
Scientific admission uses the five-seed historical reference gate or the separate
[bounded implementation-validation policy](diagnostics/README.md). The latter
preserves historical endpoint failures and conservative claim labels. Current
execution status is recorded in [VALIDATION.md](../docs/VALIDATION.md).

**No full scientific training is started by installation or tests.** Smoke mode writes
under `<output-dir>/smoke`, uses synthetic images, four reference tasks with three
updates each, and one-task branches. It retains MLP300 and all intervention types,
but reduces perturbation repetitions and Fisher samples. Smoke can never pass the
scientific gate, and every completed smoke analysis is labeled unverified.

## Commands

Run from this directory, or from anywhere after installing the package. Change the
data/output paths and device as appropriate. The frozen baseline is never tuned.

```bash
python -m cap_mech.cli reproduce --data-dir /path/to/data --output-dir results/cap_mechanistic_final --seeds 42 123 451 789 2026 --device cuda --resume
python -m cap_mech.cli m3-state-causality --data-dir /path/to/data --output-dir results/cap_mechanistic_final --device cuda --resume
python -m cap_mech.cli m2-k-dial --data-dir /path/to/data --output-dir results/cap_mechanistic_final --device cuda --resume
python -m cap_mech.cli m4-importance-assignment --data-dir /path/to/data --output-dir results/cap_mechanistic_final --device cuda --resume
python -m cap_mech.cli m1-faithfulness --data-dir /path/to/data --output-dir results/cap_mechanistic_final --device cuda --resume
python -m cap_mech.cli m5-predict --output-dir results/cap_mechanistic_final --resume
python -m cap_mech.cli m6-release --data-dir /path/to/data --output-dir results/cap_mechanistic_final --device cuda --resume
python -m cap_mech.cli analyze --output-dir results/cap_mechanistic_final --resume
```

M6 additionally requires completed M1–M5 outputs. It is secondary and can be
omitted from a primary report; unavailable evidence is explicitly identified.
The numbered scripts in `experiments/` and the script in `analysis/` are wrappers
around these same commands.

Every command supports `--dry-run`, which prints the protocol without training,
downloads, or output mutation. `--smoke` activates synthetic validation. Mechanisms
and analysis still require the explicit `--allow-unverified-reference` override
for smoke. This override is recorded in the manifest and prevents the resulting
report from labeling claims SUPPORTED.

```bash
python -m cap_mech.cli reproduce --smoke --seeds 42 --output-dir validation/one-seed
python -m cap_mech.cli m3-state-causality --smoke --seeds 42 --output-dir validation/one-seed --allow-unverified-reference
```

M5 needs at least three seeds for nested validation, and real M5 requires all
995 transitions from the five unmodified Stream-A trajectories. The all-command
smoke script generates all five synthetic trajectories.

## Exact optimizer

`cap_mech/optim/cap_final.py` implements ordinary bias-corrected Adam with the
frozen learning rate, betas, epsilon and zero weight decay. Matrix weights alone
are protected. Anchors initialize to the initial weights, and importance to zero.
For each update:

1. Compute Adam's proposal, then `r = k / (sqrt(imp) + eps)`.
2. Compute the strict outside-box contest mask before clipping.
3. Clamp the proposal unless the explicit `box_off` intervention is active.
4. Move the anchor toward the resulting parameter with rate `1/T_a`.
5. Compute the raw squared-gradient importance candidate with rate `1/T_s`.
6. Unbound coordinates ratchet upward; bound coordinates **ignore the candidate**
   and multiply their existing importance by `1 - 1/T_m`. Infinite `T_m` retains
   bound importance unchanged.

The arithmetic ordering of Adam and `lerp` matches the bundled reference optimizer.
`box_off` skips projection but keeps contest, anchor, and importance observers
active. Biases receive Adam updates and are excluded from protected diagnostics.

## Reference semantics and preregistration

The canonical packaged JSON files are in `cap_mech/configs/`. Identical top-level
copies are in `configs/`; tests enforce that the copies match. Do not edit a
configuration after result inspection. A correction requires a new experiment
version and a fresh output directory; old results remain intact.

Both streams use MLP300 (784→300→300→10, ReLU, no dropout), normalized MNIST
(`0.1307`, `0.3081`), batches of 128 including the short final epoch batch, 200
tasks, and 5,000 optimizer updates per task. Epochs continue across task boundaries.
Reporting seed order is 42, 123, 451, 789, 2026.

Stream A uses `randperm(784)` seeded by `100000 + task`, with task numbers 1–200.
Both streams evaluate the same 2,000 MNIST test indices drawn by `randperm` with
seed 999, in batches of 128.

**Stream B:** 40 fixed permutations seeded by
`200001 + pool_index` for indices 0–39. A single generator seeded with 555 produces
five successive `randperm(40)` blocks. Every domain occurs once per block and five
times overall. Evaluate all 40 domains at every task boundary; final R averages
the 39 non-current domains. M1, M4, and M6 use this same pool and schedule.

The explicit sampler reproduces the historical persistent-worker DataLoader's
index sequence: one iterator base-seed draw at initialization, the full epoch
permutation, and RandomSampler's unused exhaustion permutation. Persistent workers
do not consume another base seed at the next epoch. The default final runner used
eight persistent workers; the standalone implementation reproduces the index
sequence without worker processes. Tests compare multiple epochs against actual
loaders with both persistent and non-persistent workers.

**Plasticity definitions:**

- `P_historical_100step`: mean pre-update accuracy at global updates 500100,
  500200, …, 1000000; used exclusively for the historical reproduction comparison.
- `P_dense_all_updates`: mean pre-update accuracy at every update 500001–1000000;
  saved separately for mechanistic analyses, never compared with historical P.

The gate checks every seed against its historical target with absolute tolerance
`5e-4`. A missing seed, failed comparison, changed source, or incomplete/tampered
checkpoint bank blocks mechanisms and analysis by default. Full runs that fail the
gate stop after writing its diagnostics.

See [PROTOCOL_NOTES.md](PROTOCOL_NOTES.md) for metric and sampler details.

## Experiments and inference

- **M1:** frozen Stream-B snapshots; within-layer top/bottom 1%, 5%, and 20%;
  20 matched random groups; three Gaussian antithetic perturbations per group at
  0.1 times layer weight SD. Importance, per-example snapshot Fisher, and weight
  magnitude comparators share the same random reference. Weights are restored
  exactly even after evaluation errors. All group measurements are saved.
  Fisher uses 200 stratified examples per pool permutation: 8,000 per-example
  gradients with the pool-40 protocol, evaluated in microbatches of 16.
- **M2:** five finite k multipliers plus box off, five future Stream-A tasks;
  behavioral paired deltas and within-block finite-k rank responses.
- **M3:** baseline, box off, anchor recenter, importance reset, and full CAP reset;
  one new task, old probes before and after 1,000/5,000 updates, and constraint tax
  by stream age. Adam state and weights are preserved at intervention time.
- **M4:** true importance, eight within-layer shuffles, and reversed assignment;
  ten future Stream-B tasks. Shuffle repetitions are averaged within each
  seed/checkpoint before inference. Immediate prediction identity is asserted.
- **M5:** seven prespecified state features, baseline and combined ridge models,
  outer leave-one-seed-out validation and inner leave-one-training-seed-out alpha
  selection. Scaling uses training data only. All four outcomes are reported;
  early AUC is primary. The capped threshold target is descriptive, not survival
  inference. Threshold crossing uses a 200-update rolling accuracy window.
- **M6:** five release times including infinity, ten future Stream-B tasks, with
  bound-coordinate importance decay, retention gaps, and behavioral responses.

Inference uses 10,000 whole-seed bootstrap samples. Coordinates, group draws,
noise trials, shuffle repetitions, and checkpoints are not independent seeds.
All seed effects accompany intervals. No p-values are generated.

Claim labels use explicit conservative rules documented in the generated report:
all required directional confidence intervals must exclude zero, with five seeds
and verified outputs, for SUPPORTED. Uncertain or inconsistent evidence is MIXED.
Unavailable evidence is separately identified as unavailable, not a falsification.
The report never rewrites a failed contrast into a favorable claim.

## Outputs and recovery

See `OUTPUT_SCHEMA.md` for tables, columns, and checkpoint contents. Each run
directory has `run_config.json`, `raw/`, and `summary/`; source/config hashes and
the experiment version are locked. Reusing an existing directory requires
`--resume`. Different source, config, or source checkpoint hashes are rejected.

Training commits full checkpoints every 1,000 updates and at task boundaries.
Each checkpoint contains model, optimizer, sampler, Python/NumPy/CPU/CUDA RNG,
task position, per-task accumulators, initial retention probes, trailing bind
history, and journal row counts. Atomic temporary-file replacement commits the
checkpoint. Resume trims journal rows beyond that commit and restores exact
continuation state. Replayed bank checkpoints must match the immutable copy.

Frozen M1 resumes at completed group boundaries using independently fixed seeds;
it caches Fisher and ranks, and never trains or alters source checkpoints. M5 and
summary calculations are deterministic and can be recomputed on resume.

Checkpoints are trusted local PyTorch artifacts; load only checkpoints generated
by this suite. Exact continuation requires the recorded software/device/thread
configuration. Cross-device or cross-version numerical reproduction is not
claimed; the reference gate determines whether the scientific target is met.

## Compute scope

Full reference reproduction alone is ten million optimizer updates. Branches and
especially M1's repeated retained-loss probes add substantial compute. Raw online
logs retain every update and minibatch-index hash, so budget disk space accordingly.
Reference comparisons and bounded-check outcomes are written by each run;
precomputed validation outputs are not bundled. See
[VALIDATION.md](../docs/VALIDATION.md) for test commands and execution status.
