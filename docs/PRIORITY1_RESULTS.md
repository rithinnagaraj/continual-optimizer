# Priority 1: expanded UPGD tuning and benchmark evaluation

Recorded 20 November 2026. The primary search, confirmation, and selected-model
benchmark evaluation are complete. No further UPGD training is required for this
priority. Manuscript edits remain to be applied.

## Main finding

A 64-configuration UPGD search followed by confirmation of the top three on a
second development seed retained the original configuration. Its five-seed
benchmark balanced score is **0.912714 ± 0.000486** (mean ± sample standard
deviation). Plasticity, retention, and balanced-score means and standard deviations
match the historical UPGD summary to all six recorded decimal places. This is
reproduction of the reported result, not an improvement over it.

This addresses reviewer 1 requested change 1 and reviewer 2 important requested
change 3: the unequal tuning budgets and reliability of the UPGD comparison.
It is an **expanded UPGD search**, not equal historical or lifetime tuning effort.
Historical CAP tuning included 216 configurations and subsequent development;
the previously reported baseline searches included 16 UPGD, 8 AdamW, and 24 EWC
configurations. Do not combine old and new validation scores or present these
counts as directly equivalent compute budgets.

## Frozen protocol

The complete machine-readable protocol and source hashes are in
[campaign.json](rebuttal_priority1/campaign.json). The implementation and runbook
are described in [REBUTTAL_PLAN.md](REBUTTAL_PLAN.md).

| Parameter | Search values |
|---|---|
| Learning rate | 0.0001, 0.001, 0.01, 0.03 |
| Noise sigma | 0, 0.0001, 0.01, 0.1 |
| Utility beta | 0.999, 0.9999 |
| Weight decay | 0, 0.01 |

All 64 Cartesian-product configurations used the pinned upstream
`FirstOrderGlobalUPGD`. Primary-search seed: 31415. Confirmation seed: 27182.
These are disjoint from the reporting seeds 42, 123, 451, 789, and 2026.

Each tuning job trained for 200,000 updates: 20 permuted-MNIST domains, two
shuffled rounds, 40 tasks of 5,000 updates, batch size 128. A stratified 5,000-example
holdout (500/class, split seed 777) came from the MNIST training set; training used
the other 55,000 examples. Official test data were not used for selection.
Candidates shared initialization and sampled data order within a development seed.

- P_val: mean held-out current-domain task-end accuracy over tasks 21–40.
- R_val: final held-out accuracy averaged over the 19 non-current domains.
- B_val: min(P_val, R_val), computed separately for each seed.
- Primary ranking: descending B_val, then descending mean(P_val, R_val), then
  ascending grid index. Confirm the top three, adding the incumbent if absent.
- Final selection: descending mean of the two per-seed B_val values, then mean
  P/R across both seeds, then ascending grid index. No test-based selection.

All 64 primary jobs and all three confirmation jobs completed; none diverged.
The incumbent was already in the top three, so no fourth confirmation was needed.

## Selection results

All three finalists used learning rate 0.001 and weight decay 0.

| Grid index | Sigma | Utility beta | Primary B_val | Confirmation B_val | Two-seed mean B_val |
|---|---:|---:|---:|---:|---:|
| **022 (incumbent, selected)** | 0.0001 | 0.9999 | 0.861000000 | 0.861221053 | **0.861110526** |
| 018 | 0 | 0.9999 | 0.860936842 | 0.861221053 | 0.861078947 |
| 020 | 0.0001 | 0.999 | 0.860568421 | 0.860200000 | 0.860384211 |

The selected candidate's mean advantage over zero noise is only 0.00316 percentage
points. Do not interpret this as evidence that nonzero noise is superior. The
predeclared rule retained the incumbent among practically similar candidates.

Full primary-search results: [search.csv](rebuttal_priority1/search.csv).
Frozen selection, including evidence hashes:
[selected_config.json](rebuttal_priority1/selected_config.json).

## Five-seed benchmark results

Evaluation used the selected configuration on both original benchmark streams:
five seeds × two streams × 1,000,000 updates, or ten jobs and 10 million updates.
Each job trained on all 60,000 MNIST training examples for 200 tasks. Test evaluation
used the fixed 2,000-example official-test subset (seed 999).

P is pre-update incoming-batch accuracy sampled every 100 updates in the second
half of stream A. R is final test accuracy averaged over the 39 non-current
domains in stream B. B = min(P, R) is computed per seed before aggregation.

| Seed | P | R | B |
|---|---:|---:|---:|
| 42 | 0.912208854 | 0.924871795 | 0.912208854 |
| 123 | 0.912920313 | 0.927012821 | 0.912920313 |
| 451 | 0.912738021 | 0.923602564 | 0.912738021 |
| 789 | 0.912298438 | 0.919884615 | 0.912298438 |
| 2026 | 0.913402604 | 0.924653846 | 0.913402604 |
| **Mean** | **0.912713646** | **0.924005128** | **0.912713646** |
| **Sample SD** | **0.000486102** | **0.002615284** | **0.000486102** |

P is lower than R for all five seeds, so B equals P in every case. SD is across
seeds, not a standard error or confidence interval. Validation B_val and benchmark
B have different protocols and must not be compared directly.

Full-precision exports: [benchmark.csv](rebuttal_priority1/benchmark.csv) and
[per_seed_metrics.csv](rebuttal_priority1/per_seed_metrics.csv).

The historical report used for comparison is
`results/main_res/safe_retune_and_horizon_findings_report.md` in the research
workspace (outside this public source repository). Its UPGD row is
P = 0.912714 ± 0.000486, R = 0.924005 ± 0.002615, B = 0.912714 ± 0.000486.
Matching summary statistics is not a claim of checkpoint or trajectory equality.
No CAP, AdamW, or EWC benchmark reruns were performed in this campaign.

## Measured compute and environment

One NVIDIA L4, one worker at a time; Python 3.10.22, PyTorch 2.11.0+cu130,
torchvision package metadata 0.26.0, CUDA 13.0, cuDNN 91900, one CPU thread,
deterministic algorithms enabled, TF32 disabled, CUBLAS workspace `:4096:8`.

| Stage | Jobs | Updates | Recorded job-hours |
|---|---:|---:|---:|
| Primary search (includes resumed pilot/incumbent) | 64 | 12,800,000 | 9.1004 |
| Confirmation | 3 | 600,000 | 0.4255 |
| Selected UPGD benchmark | 10 | 10,000,000 | 6.9182 |
| **Total** | **77** | **23,400,000** | **16.4441** |

Confirmation took 25.53 minutes. Mean evaluation job time was 40.66 minutes for
stream A and 42.36 minutes for stream B. The pilot's work is included in the
incumbent search job, not charged twice. Smoke tests and human analysis are outside
this table. Recorded times sum attempt durations, include training/evaluation/
checkpoint work, and exclude idle time and process startup/admission checks before
the attempt clock. They are not device-utilization measurements or rental billing.

The optional fixed CAP validation reference and optional CAP benchmark arm were
not run. A separate incumbent evaluation would duplicate the selected arm and was
therefore unnecessary.

## Evidence and verification

[verification.json](rebuttal_priority1/verification.json) records verification
outcomes, archive SHA-256 identifiers, timings, and checkpoint hash manifests.
The compact evidence files alongside it were copied from the evaluation feedback
archive; they are not manually reconstructed results.

Verified all 64 search and three confirmation results, the shortlist, frozen
selection, all ten evaluation results, and all 231 job CSV hashes across the
three stages. Independently recomputed selection and benchmark scores, checked
task/scalar/pool coverage, and checked protocol/source/environment identities.
The evaluation archive preserved 813 prior campaign files byte-for-byte.
All evaluation jobs completed with no logged errors or numerical divergence.

The feedback archives omit `.pt` files. Their actual checkpoint bytes were not
verified locally; only result manifest entries were checked. Preserve the L4
campaign directory, especially `jobs/eval-selected-<seed>-b/task_100.pt` and
`task_200.pt`, for native-utility M1 analysis. Full logs remain in the supplied
feedback archive and on the L4 machine; this repository contains compact evidence.

## Text for the paper and reviewer response

Suggested appendix paragraph:

> To assess sensitivity to the original UPGD tuning budget, we conducted an
> expanded 64-configuration search over learning rate, perturbation noise,
> utility decay, and weight decay. Each configuration received 200,000 updates
> on a shared training/validation split and development seed. We confirmed the
> three leading candidates on a second development seed and selected by the
> mean per-seed validation balanced score. This procedure retained the original
> UPGD configuration. Evaluation on five reporting seeds and both one-million-
> update benchmark streams yielded P = 0.912714 ± 0.000486,
> R = 0.924005 ± 0.002615, and B = 0.912714 ± 0.000486 (mean ± sample SD),
> reproducing the previously reported summary at its recorded precision.
> Search, confirmation, and evaluation required 16.44 recorded job-hours on one
> NVIDIA L4. This expands UPGD tuning but does not equalize historical development
> effort across methods.

Suggested response to reviewer 1 request 1 and reviewer 2 important request 3
(use after the corresponding manuscript edits have been made):

> We expanded UPGD tuning from the previously reported 16-configuration search
> to a new 64-configuration search, confirmed the three leading candidates on
> a second development seed, and evaluated the selected configuration on five
> reporting seeds. The selection retained the original configuration and
> reproduced its benchmark result. We now disclose the search spaces, training
> horizons, selection protocol, compute cost, and remaining historical tuning
> disparity. The evidence supports robustness of the reported UPGD result to
> this expanded search; we do not claim equal lifetime tuning budgets.

Paper-edit checklist:

- Update Appendix B/Table 5 with separate historical and new campaign budgets,
  horizons, seeds, and selection rules; do not silently replace historical counts.
- Add the confirmation table and five-seed result above to the appendix.
- Add a main-text pointer to the expanded-tuning check. The headline UPGD score
  remains 0.913 at three decimals; do not describe a new performance gain.
- Keep the existing reproduction and optimizer-overhead disclosures. This check
  does not resolve unrelated CAP reproduction differences or compute overhead.
- Retain the historical-tuning limitation; no global optimality or statistical
  superiority among the shortlisted settings has been established.

## Next experiment

Priority 2 is native UPGD utility versus empirical Fisher and random ranking in
the M1 perturbation test, evaluated within the same UPGD checkpoint. It addresses
reviewer 2 important request 2. Reuse the five final stream-B checkpoints; no new
training is planned. The separate runner and bounded pilot commands are now in
[PRIORITY2_RUNBOOK.md](PRIORITY2_RUNBOOK.md). Time the pilot before launching all
seeds. The previous 1–6 GPU-hour estimate is provisional.
