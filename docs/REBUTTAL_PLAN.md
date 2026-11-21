# TMLR rebuttal experiment plan

**20 November 2026 update:** The priority-1 UPGD search, confirmation, and
five-seed benchmark evaluation are complete. See [PRIORITY1_RESULTS.md](PRIORITY1_RESULTS.md)
for measured costs, verified results, compact evidence, and manuscript/reviewer
response text. The earlier estimates below are retained as the original plan.

Priority 2's five-seed checkpoint probes are also complete. See
[PRIORITY2_RESULTS.md](PRIORITY2_RESULTS.md) for results and
[PRIORITY2_RUNBOOK.md](PRIORITY2_RUNBOOK.md) for the frozen scope and commands.
Paper revisions are deferred until the planned experiments finish.

**21 November 2026 update:** Priority 3's resumable CIFAR baseline runners and
bounded timing pilot are implemented in [PRIORITY3_RUNBOOK.md](PRIORITY3_RUNBOOK.md).
Matching historical CAP 80-task prefixes have been recovered and bundled with
provenance; no new baseline scientific runs were launched during implementation.

Priority 1 uses a separate UPGD tuning/evaluation campaign; priority 2 uses the
checkpoint-probing runner linked above. Priority 3 has a separate CIFAR runner;
priority 4 remains a planned follow-up.
No scientific campaign was launched during implementation. Target: freeze experimental results during the
first week of an approximately two-week rebuttal window.

## Hardware and budget

One NVIDIA L4 is sufficient for the 784–300–300–10 MNIST MLP, batch size 128,
FP32 training, and optimizer checkpoints. The original benchmark also used L4.
Run one worker per L4 initially. Small tensor operations, CPU indexing, evaluation,
and durable checkpoint writes can dominate elapsed time; a larger GPU is not
automatically faster for this workload.

Provisional allocation for priority 1 is **18–36 GPU-hours**, plus the inexpensive
two-seed fixed CAP validation reference. This is an estimate, not an L4 timing
measurement of this implementation. Historical UPGD logs show roughly 2.5–3.2 ms
per update on earlier protocols, but hardware attribution and evaluation costs
do not support treating those as a current benchmark. Time a bounded pilot on
the actual L4, then replace the estimate. No deadline guarantee is implied.

| Work | Jobs | Updates |
|---|---:|---:|
| UPGD primary search | 64 | 12.8M |
| Confirmation, top three plus incumbent if different | 3–4 | 0.6–0.8M |
| Fixed CAP validation reference | 2 | 0.4M |
| Selected UPGD, five seeds and two full streams | 10 | 10M |
| Optional incumbent evaluation in the new environment | 10 | 10M additional |
| Optional fixed CAP evaluation in the new environment | 10 | 10M additional |

The default evaluation roster contains only selected UPGD. If it is identical to
the incumbent, the two labels represent the same configuration; do not spend
compute evaluating both. If a new environment changes the fixed incumbent's
behavior, use fresh paired controls before presenting a precise improvement over
the historical numbers. Optional incumbent or CAP evaluation is explicit, never
automatically launched.

## Priority 1 scientific protocol

Addresses reviewer 1 request 1 and reviewer 2 important request 3: unequal tuning
budgets and the reliability of the reported UPGD comparison.

The paper reports 216 CAP configurations versus 16 UPGD, 8 AdamW and 24 EWC,
plus later CAP development/retuning. This revision is an **expanded UPGD search**,
not a claim of equal lifetime development or tuning budgets. Do not merge old
validation scores with this new campaign: the data split and development seeds
are deliberately explicit and different.

The frozen grid is the Cartesian product:

| Parameter | Values |
|---|---|
| Learning rate | 0.0001, 0.001, 0.01, 0.03 |
| Noise sigma | 0, 0.0001, 0.01, 0.1 |
| Utility beta | 0.999, 0.9999 |
| Weight decay | 0, 0.01 |

The incumbent is lr=0.001, sigma=0.0001, beta=0.9999, weight decay=0
(`search-022`). Use the paper's vendored `FirstOrderGlobalUPGD`, with its source
hash checked. Its update is unchanged; the existing device-context adaptation
places its scalar factory on the parameter device. The unrelated
`upgd_corrected` historical diagnostic is not used.

Primary search uses seed **31415**. Confirmation uses **27182**. Neither is a
reporting seed. Each candidate gets 20 recurring permutations in two shuffled
rounds, 40 tasks of 5,000 updates, batch size 128 with short final batches kept.
Validation permutations use base seed 300001; schedule seed is 556.

Split 5,000 examples (500/class, split seed 777) out of MNIST's training set.
Train on the remaining 55,000. Score every validation domain on this holdout.
The official test set is not opened during tuning or confirmation. All candidates
share the split, schedule, model initialization and sampled data order within a
development seed.

- P_val: mean held-out current-domain task-end accuracy in the second round.
- R_val: final mean held-out accuracy on the 19 non-current domains.
- B_val: min(P_val, R_val).
- Shortlist: top three primary-search B_val values, then incumbent if absent.
  Ties use mean(P_val,R_val), then ascending frozen grid index.
- Winner: highest mean **per-seed B_val** across the search and confirmation
  seeds; ties use mean P/R across both seeds, then grid index. Do not take the
  minimum of separately averaged P and R instead.
- Missing jobs or infrastructure errors block selection. Numerical divergences
  are recorded and excluded; failure of the incumbent primary search blocks
  confirmation for integration inspection. All confirmation jobs must terminate
  before selecting. There is no automatic grid expansion or test-based selection.

The fixed final CAP reference (lr=1/300, k=7e-5, Ta=Ts=2000, Tm=1e6,
contest-blocked Release) runs on both development seeds. It is a contextual
reference, not another hyperparameter search and not part of UPGD selection.

Final evaluation uses all 60,000 training examples and the original fixed 2,000
test examples (subset seed 999), reporting seeds 42/123/451/789/2026, 200 tasks,
5,000 updates/task, and the original stream schedules. Stream A has permutation
seed 100000+task. Stream B has 40 permutations beginning at seed 200001 and five
shuffled rounds with schedule seed 555.

- P: pre-update incoming-batch accuracy every 100 updates in the second half of A.
- R: final mean accuracy over the 39 non-current B domains.
- B: min(P,R) within each reporting seed; then mean and sample SD over seeds.

The runner reuses the existing exact `IndexSampler` to preserve persistent-worker
RandomSampler RNG semantics without maintaining worker processes. It uses the
root MLP300 and the same normalization. This is a new deterministic harness;
synthetic equivalence checks do not establish million-update numerical agreement
with historical runs. The new campaign records software/device/data identities.

## Files and job lifecycle

| File | Responsibility |
|---|---|
| `experiment_upgd_tuning.py` | Plan/init, list/run jobs, shortlist, select, report/status |
| `experiment_upgd_evaluation.py` | List/run selected or explicit control evaluations and report/status |
| `upgd_rebuttal.py` | Frozen protocol, identities, locks, selection, aggregation |
| `upgd_rebuttal_runtime.py` | Data splits, training, evaluation, checkpoints and timing |
| `tests/test_upgd_rebuttal.py` | Protocol gates, interruptions, exact CPU resume and synthetic pipeline |

Default output root: `results/upgd_rebuttal`. Use a separate directory for smoke
and for any scientifically changed campaign. Initialization freezes source hashes,
the entire grid, split, seeds, schedules and metrics. Resume rejects a source,
protocol, data or numerical environment mismatch. Adding tests/docs does not
change the scientific source identity; changing worker code does.

Every job has its own directory under `jobs/`, exclusive OS worker lock,
`job.json`, `environment.json`, `data_identity.json`, `latest.pt`, `timing.jsonl`,
and terminal `result.json`. Terminal jobs export `scalars.csv`, `tasks.csv`, and
`pool.csv`. Hashes protect committed outputs. Only a complete coverage-checked
trajectory or an explicitly recorded numerical divergence is terminal.

Checkpoints are atomically replaced every 1,000 updates by default, at task
boundaries, and at deliberate pauses. They include parameters, native utility,
optimizer step counts, CPU/CUDA/Python/NumPy RNGs, sampler order/position/generator,
partial-task metrics, committed observations, data indices and the stream schedule.
An interrupted write leaves the previous committed checkpoint usable. A resume
replays only work since the last commit. A completed checkpoint can regenerate
exports/result publication if the process died immediately before those writes.

Stream-B evaluation also preserves `task_100.pt` and `task_200.pt` with utility
state and parameter names for priority 2. These are independent of later latest
checkpoint replacements. Smoke uses tasks 2 and 4 instead.

Timing events contain attempt ID, UTC Unix timestamp, elapsed wall time, completed
step, training/evaluation seconds, and checkpoint serialization time. The attempt
clock starts after dependency loading and admission/environment checks, so process
startup and those checks are outside its scope. Attempt durations include data/model
setup, evaluation and checkpointing; summaries exclude idle
time between processes and include recorded repeated work after recovery. A hard
kill can lose timing after the last event: this tail is marked unknown, never
reported as exact zero. Interrupted final JSONL appends are discarded on recovery.

Use distinct job IDs for concurrent workers; writing the same job is rejected.
OS locks release on process exit, so no stale-lock deletion is needed. Do not
delete lock files, move an active campaign, or edit frozen manifests. Model
checkpoints are trusted local pickle payloads; do not load untrusted checkpoints.

## Commands for the L4 machine

Run from the repository root (the local `public_repo` directory, or the root of
the GitHub clone). Use a CUDA-enabled PyTorch image and preserve its working
torch/torchvision pair. For priority 1, install only the lightweight requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-upgd.txt
python -c "import torch, torchvision, numpy; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, numpy.__version__, torch.cuda.get_device_name(0))"
python tools/audit_release.py
```

Do not install the full `requirements.txt` with CUDA-build constraints for this
campaign: that asks pip to resolve the GPU packages again and pulls dependencies
for unrelated experiments. The priority-1 runners need only the existing torch,
torchvision, NumPy, and the Python standard library. A normal checkout audit
ignores generated virtual-environment symlinks; `--strict` is for clean source
exports and intentionally rejects them. These commands do not start training.

Inspect the production plan without importing PyTorch, downloading data, or
creating output directories:

```bash
python experiment_upgd_tuning.py plan
```

Run the implementation tests and a separate small synthetic job:

```bash
python -m unittest tests.test_upgd_rebuttal -q
python experiment_upgd_tuning.py init --smoke --output-dir results/upgd_smoke
python experiment_upgd_tuning.py run --output-dir results/upgd_smoke --stage search --job-ids search-022 --device cuda --stop-after-updates 5
python experiment_upgd_tuning.py run --output-dir results/upgd_smoke --stage search --job-ids search-022 --device cuda --resume
```

Prepare a production campaign and inspect one pilot job:

```bash
python experiment_upgd_tuning.py init
python experiment_upgd_tuning.py run --stage search --job-ids search-022 --dry-run
```

When ready to spend compute, the bounded pilot runs only 10k additional updates,
including two task-boundary evaluations, then pauses. It does not redefine the
candidate's 200k horizon or make it eligible for selection:

```bash
python experiment_upgd_tuning.py run --stage search --job-ids search-022 --device cuda --stop-after-updates 10000 --resume
python experiment_upgd_tuning.py status --stage search
```

Continue at most one pending job, or explicitly authorize the complete stage:

```bash
python experiment_upgd_tuning.py run --stage search --all --max-jobs 1 --device cuda --resume
python experiment_upgd_tuning.py run --stage search --all --device cuda --resume
python experiment_upgd_tuning.py run --stage reference --all --device cuda --resume
python experiment_upgd_tuning.py report
python experiment_upgd_tuning.py shortlist
python experiment_upgd_tuning.py jobs --stage confirm
python experiment_upgd_tuning.py run --stage confirm --all --device cuda --resume
python experiment_upgd_tuning.py select
```

`--max-jobs` limits pending jobs, not wall time. Use `--job-ids` to schedule
specific jobs independently. `--stop-after-updates` applies separately to each
selected job, so pair it with one job ID for a bounded pilot.

Evaluate only the frozen winner, with individual jobs or the full explicit stage:

```bash
python experiment_upgd_evaluation.py jobs
python experiment_upgd_evaluation.py run --job-ids eval-selected-42-b --device cuda --resume
python experiment_upgd_evaluation.py run --all --device cuda --resume
python experiment_upgd_evaluation.py report
```

Optional controls are opt-in:

```bash
python experiment_upgd_evaluation.py run --arms incumbent cap --all --device cuda --resume
python experiment_upgd_evaluation.py report --arms selected incumbent cap
```

Selection and reports refuse partial evidence. The search budget curve is in
fixed grid order and is descriptive validation evidence; it is not a randomized
search-efficiency estimate. Numerical failures remain visible in the search CSV.
Reports from smoke campaigns are explicitly marked and must not enter the paper.

## Implementation validation

On 19 November 2026, 24 tests passed across `tests.test_upgd_rebuttal`,
`tests.test_paper_run_integrity`, and `tests.test_public_release`, using Python
3.14.6 and PyTorch 2.13.0+cpu. Coverage includes the complete tiny synthetic
search/confirmation/selection/evaluation workflow, exact mid-task resume,
recovery after an evaluation interruption and before result publication,
numerical failure handling, output corruption, native utility checkpoint retention,
selection arithmetic, job locks, and packaging a portable source copy.

These checks used synthetic inputs and temporary directories; no MNIST downloads,
production campaign initialization, or full scientific jobs were performed.
CUDA/L4 execution and real-data timing remain to be checked with the bounded
commands above. The inherited full scientific suites were not rerun.

## Priorities 2 to 4 after the first campaign

| Priority | Minimal follow-up | Reviewer concern | Incremental estimate |
|---|---|---|---|
| 2 | Five final UPGD B checkpoints; native utility/Fisher/random M1 at 1%, existing noise protocol; compare rankings within each host model | R2 important request 2: comparison with existing protection statistics | 1–6 GPU-hours, no new training |
| 3 | AdamW/UPGD on CIFAR MLP and CNN, five seeds, both streams, 80 tasks (400k updates); explicit LR pilot; recover matched CAP prefixes | R2: absent CIFAR learner baselines | L4 measured projection 16.52 hours; reserve 20–24 hours, plus 5–7 minutes for bounded LR extension |
| 4 | Recover CIFAR A checkpoints at 20/50/100/150; four M3 branches, five seeds, two architectures, one task each (800k updates); log proposed/clipped/applied update norms | R2: MLP–CNN difference and null recentering effect | 2–8 GPU-hours if checkpoint banks are recovered |

The local scientific review summaries do not include full mechanistic checkpoint
payloads. Recover original banks and their source/environment manifests before
committing to priorities 2–4. Existing `validation/artifacts` smoke outputs must
not be substituted for scientific runs. M1 already includes frozen empirical
Fisher, so do not duplicate that experiment. A large anchor-timescale sweep,
new benchmark, and broad baseline suite are outside this first-week scope.

Priority 3 status, 21 November: the four timing pilots and all 40 initial LR trials
are verified. Both UPGD winners reached the highest tested LR, so run the
[12-trial bounded extension](PRIORITY3_LR_EXTENSION.md) before evaluation.
[Verified evidence and timings](PRIORITY3_RESULTS.md) are saved. Matching historical
CAP 80-task prefixes have been recovered; priority 3 does not need new CAP training.
Priority 4 still requires the original mechanistic checkpoint banks.

Keep the paper framing narrow: inspectable protection studied with matched
interventions. Broaden related work and disclose the expanded but still unequal
historical tuning budgets. Bring the nontrivial M3/M4 findings forward, describe
the expected-by-design M2/reset behavior, and retain runtime/reproduction caveats.
