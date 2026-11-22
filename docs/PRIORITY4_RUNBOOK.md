# Priority 4: why constraints affect the two architectures differently

Frozen plan, 23 November 2026. Addresses Reviewer 2's instrument-use-case request,
MLP/CNN age difference, and null anchor-recentering explanation. Manuscript changes
remain deferred until the experimental program is complete.

## Design and scope

Use authenticated original CIFAR stream-A CAP checkpoints at tasks 20, 50, 100,
150, seeds 42/123/451/789/2026, both MLP and CNN. From each checkpoint restore
weights, Adam moments, protection state, sampler and RNG. Run baseline, box-off,
importance-reset and recenter-only for one complete 5,000-update new task:
160 jobs, 800,000 updates. No new reference training or hyperparameter search.
Original selected CAP learning rates are 0.0002 (MLP) and 0.0003 (CNN).

At updates 1–20 and every 100 thereafter, reconstruct the exact Adam proposal
on copies of optimizer tensors. Execute the unchanged frozen CAP step, asserting
that its result exactly equals the reconstructed proposal after projection.
Record per-weight-matrix proposed, applied and removed update L2 norms, binding
fraction, mean radius/importance, RMS anchor distance, mean distance/radius, and
fraction of coordinates at least 90% of their radius from the anchor. For layer
comparisons divide norms by sqrt(coordinate count). Removed/proposed is a ratio,
not necessarily a fraction bounded by one. Box-off binding is counterfactual
geometric binding; its actual removed update is zero.

Record every update's pre-update minibatch accuracy, loss, binding, and batch
hash. Primary learning response: mean accuracy over updates 1–1,000, as in M3;
secondary first-200 and whole-task accuracy. Retention uses the original old-task
probe list and fixed 2,000-example test subset at 0/1,000/5,000 updates. Inspect
both early transient and sustained projection cost: recentering may remove
initial anchor displacement without preventing later binding.

Primary summaries are paired branch-minus-baseline effects within seed and age.
Report baseline age trends and whether projection costs increase alongside reset
benefits in MLP but not CNN. Summarize uncertainty by resampling the five seeds,
keeping ages and layers together; do not count layers or steps as independent
replicates. Architecture comparisons are descriptive: different models and
selected learning rates prevent attributing differences solely to architecture.
Binding alone is not evidence of damage; relate it to removed update magnitude
and measured accuracy/retention effects. Importance-reset opening boxes is
expected by design. Do not claim unique inspectability or superiority over other
optimizers, and retain the narrower paper framing if the diagnostic use case is
inconclusive. Do not select additional interventions based on favorable results.

## Integrity and execution

`docs/priority4_reference.json` pins the original review archive and the exact
original run-config/gate documents. Initialization hashes all 90 bank files
against those documents. Each job rechecks its source hash before trusted
deserialization and verifies stream, age, seed, scientific status and provenance.
The public package renamed the original optimizer; compatibility is explicit
through pinned original config/checkpoint metadata and numerical parity tests.
The old frozen package is not edited. The new campaign separately pins source
and environment. Torchvision verifies the standard CIFAR archive; the runner
uses the original normalization, sampler, permutation and test-subset functions.

The entire campaign has an OS lock. Every 1,000 updates and pilot stop, an atomic
checkpoint commits all model/optimizer/sampler/RNG state and accumulated records.
Resume replays only work after the last committed checkpoint. Completed results
can be reconstructed from the final checkpoint if publication was interrupted.
Timing includes loading-independent training, diagnostics and evaluations;
process-level pilot wall time additionally captures startup/data load overhead.
No checkpoints are removed automatically.

Set BANK to the recovered lrsel directory containing mlp/ and cnn/ subdirectories.
Use the existing Python environment and data directory:

```bash
python experiment_cifar_diagnostics.py init --bank-root "$BANK"
python experiment_cifar_diagnostics.py run --bank-root "$BANK" --resume \
  --job-ids mlp-42-20-baseline mlp-42-150-baseline cnn-42-20-baseline cnn-42-150-baseline \
  --stop-after-updates 1000
python experiment_cifar_diagnostics.py run --bank-root "$BANK" --all --resume
python experiment_cifar_diagnostics.py status --bank-root "$BANK"
```

Pilot prefixes remain part of the frozen full jobs, not separate shortened
scientific runs. Inspect exact proposal checks, numerical finiteness, elapsed
time, GPU memory, and free disk before full launch. Initial budget is 2–8 L4 GPU
hours; use measured pilot timings to refine it. Preserve the original banks.
After completion verify all 160 result/checkpoint pairs, matched batch hashes,
expected trace/retention counts and recomputed metrics before reporting.

```bash
python report_cifar_diagnostics.py
```

The four L4 pilot prefixes passed exact sampled proposal checks. Measured job
times for 1,000 updates were 3.76/3.42 seconds (MLP ages 20/150) and 4.94/4.71
seconds (CNN). Linear scaling gives 0.935 job-hours for the campaign; actual
wall time also includes loading and checkpoint commits. The full 160-job run
was authorized and launched after these checks, with 36.7 GiB disk available.
The original source-tree digest also matched the archived source version, and
rename-normalized original/public optimizer implementations were identical.
Data, sampler, architecture, evaluation and protocol functions also matched.
The released seed helper adds an explicit matmul-TF32 disable that the original
left at the framework default. New matched branches all use the same recorded
settings; exact historical cross-GPU trajectory reproduction is not claimed.
