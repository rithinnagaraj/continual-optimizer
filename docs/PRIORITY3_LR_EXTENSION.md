# CIFAR bounded LR extension

**23 November update:** all 40 scientific evaluations have finished and passed
verification. Recorded evaluation cost was 16.862 hours. See the
[final benchmark and evidence](PRIORITY3_RESULTS.md). The commands below remain
as reproduction/recovery instructions; no additional priority-3 training is planned.

**Verified completion, 21 November:** all 12 extra trials finished in 4.28 minutes.
The merged selection is AdamW LR 0.0003 and UPGD LR 0.1 for both architectures;
all winners are interior. No further extension is planned. Continue with
**Evaluation after extension review** below. The full evaluation projects to
16.58 hours; reserve 20–24 hours. See [verified results](PRIORITY3_RESULTS.md).

The initial search completed successfully, but both UPGD winners were the largest
tested LR (0.01). Before benchmark evaluation, test 0.03, 0.1, and 0.3 for both
AdamW and UPGD on both architectures. This is exactly 12 new 5,000-update jobs;
allow **5–7 minutes on an otherwise idle L4**. The original 40 jobs are reused,
not retrained. See [verified initial results](PRIORITY3_RESULTS.md).

The numerical worker and existing frozen campaigns are unchanged. A separate
extension manifest pins the original campaign and selection plus the new source.
Keep both output directories and all checkpoints. The runner verifies original
checkpoint/export hashes, refuses an extension after parent evaluation starts,
and checks the union of 52 trials before freezing the new winner. Trial ranking
is final train-holdout accuracy, then loss, then LR; no test-based selection.
Numerically diverged trials are visible and excluded; missing/changed trials
block selection. If an extended winner is still a boundary, evaluation is blocked
pending review. No stage launches another stage automatically.

## Update and initialize without training

```bash
cd ~/continual-optimizer-rebuttal
git fetch origin &&
git switch main &&
git pull --ff-only &&
.venv/bin/python tools/audit_release.py &&
.venv/bin/python experiment_cifar_lr_extension.py init
```

No dependencies need changing. Defaults reference `results/cifar_baselines` and
write the extension to `results/cifar_baselines_extended`. Custom locations use
`init --parent-dir PATH --output-dir PATH`; use the same `--output-dir` on all
later commands. The parent pointer is relative so a move preserving the two
directories' relationship remains portable.

## Run the 12 extra search trials only

```bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/l4_cifar_pilot
nohup .venv/bin/python -u experiment_cifar_lr_extension.py run \
  --stage search --all --device cuda --resume \
  > results/l4_cifar_pilot/lr-extension.log 2>&1 < /dev/null &
echo "CIFAR LR extension process ID: $!"
```

Progress:

```bash
tail -n 20 results/l4_cifar_pilot/lr-extension.log
.venv/bin/python experiment_cifar_lr_extension.py status --stage search
```

Repeat the same run command to resume after an interruption. Completed jobs are
verified and skipped; checkpoints preserve optimizer/sampler/RNG state. As in the
original runner, `--job-ids`, `--max-jobs`, and `--stop-after-updates` can bound work
further. The numerical horizon and eligible selection metrics remain fixed.

After all 12 jobs are terminal (`complete` or recorded numerical `diverged`):

```bash
.venv/bin/python experiment_cifar_lr_extension.py select &&
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-cifar-extended-search-feedback.tar.gz \
  -C results cifar_baselines cifar_baselines_extended l4_cifar_pilot
```

Send **`results/l4-cifar-extended-search-feedback.tar.gz`** for selection and timing
review before starting the long evaluation. Keep `.pt` files on the L4 machine.

## Evaluation after extension review

Use the **extension** launcher for evaluation so it uses the merged selection.
The older launcher still refers to the original ten-rate winner.

```bash
nohup .venv/bin/python -u experiment_cifar_lr_extension.py run \
  --stage eval --all --device cuda --resume \
  > results/l4_cifar_pilot/evaluation.log 2>&1 < /dev/null &
```

The unchanged scope is 40 jobs: two models × two baselines × five seeds × two
80-task streams, 16M updates. Budget 20–24 hours, pending extended-rate timing.
`status --stage eval` verifies completed outputs and lists pending work.

After completion:

```bash
.venv/bin/python experiment_cifar_lr_extension.py report \
  --stage eval --include-archived-cap &&
tar --exclude='*.pt' --exclude='*.tmp' \
  -czf results/l4-cifar-evaluation-feedback.tar.gz \
  -C results cifar_baselines cifar_baselines_extended l4_cifar_pilot
```

Archived CAP entries remain labelled historical; they are not new L4 controls.
The optional `--arms cap` evaluation is not included in these commands or budgets.

## Implementation checks

On 21 November 2026, all 26 tests passed: five extension tests, 14 existing CIFAR
tests, and seven release tests. Extension coverage includes merged old/new
selection, missing/corrupted/diverged evidence, boundary and prior-test-access
gates, exact numerical pause/resume for both UPGD architectures, protocol-provider
restoration, and a complete synthetic five-seed evaluation/report. The release
audit passed for 252 files. Tests used synthetic CPU inputs; no scientific L4
training was launched during implementation. Original campaign source identity
was independently rechecked against the returned search archive.
