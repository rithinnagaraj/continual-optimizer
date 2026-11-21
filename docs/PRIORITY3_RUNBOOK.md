# Priority 3: CIFAR-100 learner baselines

**21 November update:** the pilot and all 40 initial LR trials are verified.
Both UPGD winners hit the upper grid boundary. Follow the
[bounded extension runbook](PRIORITY3_LR_EXTENSION.md) before evaluation, using its
separate launcher and merged selection. See [results and measured timings](PRIORITY3_RESULTS.md).
The original commands below document the initial ten-rate campaign.

Implementation prepared 21 November 2026. Addresses reviewer 2's request for
AdamW and UPGD results on the CIFAR MLP/CNN settings, so CAP's learning performance
can be assessed alongside its mechanistic findings. No scientific CIFAR baseline
runs were launched during implementation. Manuscript edits remain deferred.

## Scope and frozen decisions

Use the existing CIFAR-100 MLP and SmallCNN, 4×4-pixel patch permutations shared
across RGB channels, normalization, batch size 128, no augmentation, no batchnorm,
no dropout, and the existing persistent-worker RandomSampler semantics. Keep the
last short minibatch and carry the sampler across task boundaries. Optimizers
receive no task-boundary signals or resets.

The reporting horizon is the **first 80 tasks (400,000 updates)** of each original
stream, not the original 200-task endpoint. Stream A introduces a new permutation
per task; B uses 40 domains and two shuffled rounds, preserving the original
schedule prefix. Five reporting seeds: 42, 123, 451, 789, 2026.

| Stage | Jobs | Updates | Selection/reporting role |
|---|---:|---:|---|
| Bounded timing pilot | 4 | 4,000 total | Timing only; never eligible for scientific selection |
| LR search | 40 | 200,000 total | Ten rates × two optimizers × two models |
| Selected baseline evaluation | 40 | 16,000,000 total | Two optimizers × two models × five seeds × two streams |
| Archived CAP comparison | No new training | Already available | Matching 80-task prefixes, explicitly historical |
| Optional fresh CAP evaluation | 20 extra | 8,000,000 extra | Opt-in only if environment controls become necessary |

The prior 25–85 GPU-hour estimate is provisional. **Run the pilot and inspect its
measured projections before committing to the LR search or full evaluation.**

### LR selection

For each model/optimizer, test the same ten rates as the historical CAP pilot:
`1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 1/300, 1e-2`.
Each receives 5,000 updates on stream A task 1. Use a sorted stratified split of
the original training set: 45,000 training / 5,000 validation images (50/class),
split seed 777. Official test examples are not used for timing or LR selection.

Development seed **31415** is disjoint from reporting seeds. Within an architecture,
candidates and optimizers share the initial model and training-example sequence.
Select highest final validation accuracy, then lowest validation loss, then lowest
LR. Intermediate evaluations every 1,000 updates do not select an earlier model
or cause early stopping. All 40 trials must finish; numerical divergences are
recorded and excluded, while infrastructure failures block selection. A boundary
winner is flagged for review; no automatic grid expansion occurs.

This is deliberately a **single-seed, single-task LR-only search**, not a search
for optimal continual-learning retention. It matches CAP's historical search size,
split, and horizon, but not its development seed: CAP used seed 42. Historical
exploratory test access and unequal lifetime development budgets remain disclosed.

Other optimizer settings are fixed:

- AdamW: betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01; LR selected separately
  for each architecture.
- UPGD: pinned upstream `FirstOrderGlobalUPGD`, sigma 1e-4, utility beta 0.9999,
  weight decay 0; LR selected separately for each architecture. These non-LR values
  retain the priority-1 incumbent; no claim of full CIFAR-specific tuning is made.
- Optional fresh CAP: existing CIFAR optimizer/settings and historical selected
  LR 0.0002 (MLP), 0.0003 (CNN). It is not automatically scheduled.

### Reporting metrics and CAP reference

Selected baselines retrain from scratch on all 50,000 training examples. They
evaluate the fixed 2,000-example official-test subset, seed 999.

- P_80: stream A pre-update minibatch accuracy at updates 200100, 200200, …,
  400000, covering tasks 41–80.
- R_80: final stream B accuracy averaged over 39 non-current domains at task 80.
- B_80: min(P_80, R_80), computed per seed before averaging; report mean and sample SD.

Do not compare these values directly to 200-task endpoints or use dense all-update
accuracy in place of the sampled P metric.

The pinned historical CIFAR review archive contains task and pool summaries for
all 200 tasks, so exact matching CAP prefix metrics are recoverable without model
checkpoint recovery or retraining. They are bundled in
[prefix_80.json](cifar_cap_reference/prefix_80.json), with original configurations,
table hashes, archive hash, per-seed metrics, and limitations. Extraction checked
full task/pool coverage and provenance and independently reproduced the original
200-task gate values before computing the 80-task metrics.

The archive's integrity/completeness gate passed; it was not a historical-endpoint
reproduction test. Its recorded software was PyTorch 2.11.0+cu130; the historical
pilot records an A30 environment. These are archived CAP runs, **not new L4
controls or verified bitwise-equivalent trajectories**. Reporting adds them only
with `--include-archived-cap`, using a distinct `cap_archived` label and provenance.
Do not treat the archive as a fresh same-environment superiority test. If the
comparison requires fresh controls, schedule the optional CAP arm explicitly.

## L4 commands: timing pilot first

Use the existing clone and virtual environment from priorities 1–2. No additional
packages are needed. Do not reinstall/upgrade the working CUDA torch/torchvision
pair. The first pilot downloads CIFAR-100 if absent; that setup time is separate
from steady training/evaluation timing.

Update and initialize (no training):

```bash
cd ~/continual-optimizer-rebuttal
git fetch origin &&
git switch main &&
git pull --ff-only &&
.venv/bin/python tools/audit_release.py &&
.venv/bin/python experiment_cifar_baselines.py init
```

Launch all four **bounded pilot jobs only**:

```bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/l4_cifar_pilot
nohup .venv/bin/python -u experiment_cifar_baselines.py run \
  --stage pilot --all --device cuda --resume --checkpoint-every 250 \
  > results/l4_cifar_pilot/pilot.log 2>&1 < /dev/null &
echo "CIFAR pilot process ID: $!"
```

Each pilot job is 1,000 updates at a provisional LR (AdamW 0.0003, UPGD 0.001),
using the 45k training split and a fixed 2k-image subset of the train holdout.
It measures one full 40-domain evaluation sweep and one separate single-domain
sweep. The shape/batch/evaluation workload matches the production benchmark;
the pilot is short and deliberately cannot enter LR selection or the benchmark.
It is not a prefix to reuse as a selected-model training run. Completed pilot
jobs are verified/skipped if the command is repeated.

Progress:

```bash
tail -n 20 results/l4_cifar_pilot/pilot.log
.venv/bin/python experiment_cifar_baselines.py status --stage pilot
```

When all four jobs show `complete`, make the timing report and feedback archive:

```bash
.venv/bin/python experiment_cifar_baselines.py report --stage pilot \
  > results/l4_cifar_pilot/timing-report.json &&
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-cifar-pilot-feedback.tar.gz \
  -C results cifar_baselines l4_cifar_pilot
```

Send **`results/l4-cifar-pilot-feedback.tar.gz`** before starting the search or
evaluation. Keep `.pt` files on the machine. The feedback contains four optimizer/
architecture timings, memory peaks, data/environment identities, per-block timings,
single/pool evaluation times, and projected LR-search and five-seed benchmark
hours. The projection excludes the first training block when later blocks exist,
and excludes future setup/checkpoint overhead; allow headroom. It is not a guarantee
of runtime at every selected LR or stream age.

## Later commands, after timing review

Run all LR trials:

```bash
nohup .venv/bin/python -u experiment_cifar_baselines.py run \
  --stage search --all --device cuda --resume \
  > results/l4_cifar_pilot/lr-search.log 2>&1 < /dev/null &
```

Once the search is complete, freeze and review its selections before evaluation:

```bash
.venv/bin/python experiment_cifar_baselines.py select
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-cifar-search-feedback.tar.gz \
  -C results cifar_baselines l4_cifar_pilot
```

Only after selection review, run the 40 baseline jobs:

```bash
nohup .venv/bin/python -u experiment_cifar_baselines.py run \
  --stage eval --all --device cuda --resume \
  > results/l4_cifar_pilot/evaluation.log 2>&1 < /dev/null &
```

After completion:

```bash
.venv/bin/python experiment_cifar_baselines.py report \
  --stage eval --include-archived-cap &&
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-cifar-evaluation-feedback.tar.gz \
  -C results cifar_baselines l4_cifar_pilot
```

The optional fresh CAP arm uses `run --stage eval --arms cap --all`; it adds
20 jobs and is not part of the commands above. Do not launch it without reviewing
whether fresh controls are needed. It uses the existing CIFAR CAP implementation.

## Recovery, provenance, and validation

Each job has an OS lock and atomic checkpoints, by default every 1,000 updates
and at validation/task boundaries or deliberate pauses. Checkpoints preserve the
model, optimizer state (including UPGD utility), sampler, all RNG states, partial
task metrics, data indices, stream schedule, and committed records. Interrupted
evaluation is replayed from the last committed checkpoint. A complete checkpoint
can regenerate missing final exports. Altered source, configuration, environment,
data, or completed outputs are rejected. Use one worker per L4 initially.

`--job-ids` selects individual jobs, and `--stop-after-updates` pauses each selected
job after that many additional updates without redefining its scientific horizon.
Do not confuse this with the separately defined, timing-only pilot stage. `--all`
always applies to the explicit stage; no stage automatically launches another.

The optional `--smoke` initialization is for a separate synthetic campaign and
cannot be mixed with real results. Validation covers genuine numerical pause/resume
for both optimizers and both architectures; exact patch/schedule/split agreement;
train-only timing/selection data access; final-validation selection and missing-run
gates; numerical failures; corrupted exports; interrupted pool evaluation; and
the full synthetic search/selection/five-seed evaluation/report path. CPU tests
do not establish CUDA compatibility or real-data timing: that is the pilot's job.

The CAP prefix can be independently regenerated from the pinned historical review
archive with `tools/extract_cifar_cap_prefix.py ARCHIVE --output results/cap_prefix_check.json`.
No archive or scientific checkpoint download is required to run the baseline pilot.

Validation on 21 November 2026: all 42 tests across `tests.test_cifar_baselines`
(14 new tests), `tests.test_upgd_rebuttal`, `tests.test_upgd_m1`, and
`tests.test_public_release` passed on Python 3.14.6 / PyTorch 2.13.0+cpu. The
release audit passed. Both completed MNIST campaigns retain their exact frozen
source identities. These checks used synthetic CPU data and temporary runs;
real CIFAR/L4 compatibility, memory use, and timings await the bounded pilot.
