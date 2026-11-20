# Priority 2: native UPGD utility faithfulness results

Recorded 20 November 2026. All five final stream-B checkpoint probes completed.
The planned minimal priority-2 experiment is complete. Manuscript edits remain
deferred until results from the other planned experiments are available.

## Main finding and reviewer concern

Native signed UPGD utility contains information about weight sensitivity: its
top-ranked 1% groups cause more loss damage than random groups. It is not a clean
monotonic sensitivity ranking under this probe: low-utility groups also cause
substantial damage, and top-versus-bottom ordering reverses in several matrices.
The frozen empirical Fisher diagnostic gives substantially stronger separation.

This addresses reviewer 2 important request 2 by applying M1 to UPGD's own stored
utility, with Fisher and random comparators **within the same UPGD model**.
It also demonstrates that matched inspection is applicable beyond CAP. It does
not establish a unique inspection capability for CAP or compare optimizer quality
through cross-model raw damage magnitudes. The conceptual explanation of CAP's
explicit constraint state relative to EWC/UPGD remains a manuscript task.

## Protocol and coverage

The frozen procedure is in [PRIORITY2_RUNBOOK.md](PRIORITY2_RUNBOOK.md), with the
exact manifest in [campaign.json](rebuttal_priority2/campaign.json).

- Source: selected UPGD configuration from priority 1; stream B, task 200,
  one million training updates; seeds 42, 123, 451, 789, 2026.
- Three weight matrices per checkpoint, excluding biases; 1% of coordinates
  per group (2,352 / 900 / 30 coordinates).
- Top and bottom groups for signed bias-corrected native utility and empirical
  Fisher, plus 20 random groups per matrix.
- Three Gaussian draws per group, each evaluated with positive and negative sign;
  noise standard deviation = 0.1 × the original matrix sample standard deviation.
- Primary damage: antithetic cross-entropy increase averaged over all 40 pool
  domains and the fixed 2,000-example test subset. Secondary damage excludes
  the current domain and averages the remaining 39.
- Fisher: mean squared **per-example** gradients, 200 stratified examples/domain.
  It uses labels from subsets of the evaluation examples and is a retrospective
  diagnostic, not online Fisher, an EWC run, or independent generalization evidence.
- All 360 groups completed: 72 per seed, 1,080 antithetic draws, and 2,160 perturbed
  full-pool evaluations. No training, checkpoint selection, or scope changes occurred.

## Primary results

All values in the following table are **cross-entropy loss changes multiplied by
100,000** (units of 10^-5), not accuracy percentage points. Contrasts first average
the three matrices within each seed. Intervals are percentile 95% bootstrap
intervals from 10,000 whole-seed resamples, using the frozen bootstrap seed.

| Statistic | Native utility | Frozen Fisher |
|---|---:|---:|
| Mean top-group damage | 5.8848 | 12.2262 |
| Mean bottom-group damage | 5.7248 | 0.8353 |
| Mean random-group damage | 4.4086 | 4.4086 |
| Top minus random, mean [95% CI] | **1.4762 [1.1594, 1.7930]** | **7.8176 [7.0632, 8.4658]** |
| Top minus bottom, mean [95% CI] | **0.1600 [0.0396, 0.2804]** | **11.3910 [10.6213, 12.1283]** |
| Mean top-group percentile among 20 random groups | 97 [95, 99] | 100 [100, 100] |

The paired Fisher-minus-utility difference in top-minus-random damage is
6.3414 × 10^-5, with 95% CI [5.3121, 7.0510] × 10^-5. This uses paired seed
contrasts, not overlap/non-overlap of separate confidence intervals.

All five seed-averaged utility top-minus-random contrasts are positive. The
seed-averaged top-minus-bottom contrasts are also positive, but two are nearly
zero. Only **7 of 15** individual seed–matrix utility top-minus-bottom contrasts
are positive. Utility top-minus-random is positive in **15 of 15**. Fisher has
positive top-minus-random and top-minus-bottom contrasts in **15 of 15** each.
These 15 repeated matrix measurements are not 15 independent seeds.

The Fisher percentile interval [100, 100] reflects saturation against the sampled
random controls; it is not proof of perfect coordinate ranking. Effect sizes and
paired contrasts are the substantive evidence.

## Layer-level qualification

Five-seed means below also use units of 10^-5. Layer labels correspond to the
784→300, 300→300, and 300→10 weight matrices, respectively.

| Matrix | Utility top − random | Utility top − bottom | Fisher top − random | Fisher top − bottom |
|---|---:|---:|---:|---:|
| `network.1.weight` | 2.3913 | **−0.5255** | 15.0565 | 22.6388 |
| `network.3.weight` | 0.4625 | **−0.0440** | 1.3344 | 2.0401 |
| `network.5.weight` | 1.5747 | 1.0496 | 7.0618 | 9.4940 |

The output matrix supplies the positive aggregate utility top-minus-bottom gap;
the two hidden matrices have negative mean gaps. Do not summarize this experiment
as universally showing utility top > random > bottom. Aggregate utility bottom
damage also exceeds random damage. Native signed utility and retrospective loss
sensitivity are related but different quantities; no absolute-utility ranking or
additional post-hoc comparator was substituted.

## Secondary non-current-domain result

Excluding the current domain preserves the qualitative findings. In units of
10^-5, utility top-minus-random is 1.4333 [1.1112, 1.7555], and Fisher is
7.9274 [7.1779, 8.5737]. Utility top-minus-bottom is 0.1613 [0.0488, 0.2775],
and Fisher is 11.5413 [10.7837, 12.2921]. This remains a secondary outcome;
the original all-domain result is primary.

## Measured time

One NVIDIA L4 in the unchanged priority-1 environment. Total recorded job time,
including the reused pilot, is **436.79 seconds = 7.28 minutes**. Seed 42 used two
attempts; each other seed used one. There were no logged errors or numerical
failures. The earlier 1–6-hour planning estimate was overly conservative and is
superseded by this measurement for this exact protocol.

| Seed | Recorded seconds |
|---|---:|
| 42, including pilot | 88.842 |
| 123 | 86.890 |
| 451 | 86.986 |
| 789 | 87.045 |
| 2026 | 87.029 |

Times sum recorded attempt durations and exclude idle time, process admission
before the attempt clock, and final reporting. They are not GPU utilization or
cloud billing measurements.

## Evidence and verification

- [summary.csv](rebuttal_priority2/summary.csv): full-precision effects and intervals.
- [per_seed.csv](rebuttal_priority2/per_seed.csv): seed-level contrasts.
- [group_damage.csv](rebuttal_priority2/group_damage.csv): all 360 groups for both
  domain scopes (720 rows).
- [layer_results.csv](rebuttal_priority2/layer_results.csv): independently
  reconstructed seed–matrix contrasts.
- [report.json](rebuttal_priority2/report.json): original aggregate report and
  result/artifact hashes.
- [verification.json](rebuttal_priority2/verification.json): independent audit,
  source archive SHA-256, per-seed timings, and checks performed.

Independently verified protocol/source identity against the frozen implementation,
parent evaluation result identities, and exact unperturbed accuracies against the
priority-1 pool records. Verified all 360 group hashes, coordinate selections,
noise norms, 225 nonbinary job artifacts, and report hashes. Recomputed both
domain scopes, seed aggregation, paired contrasts, and all bootstrap intervals.
The largest numerical difference from the original JSON report was 1.7 × 10^-21.
The original pilot evidence was preserved; its timing log was extended on resume.

Checkpoint, Fisher, and score `.pt` files were excluded from the feedback archive.
Their actual bytes were not reverified locally; only their hash manifests were
available. Keep the complete L4 campaign and checkpoints. The compact repository
exports do not replace the original archive's per-domain draw measurements.

## Scope limits and next work

These results apply to the final checkpoint, the 1% fraction, this MLP, and five
seeds. They do not establish behavior at other ages, architectures, or fractions.
Bootstrap intervals describe this five-seed sample; they do not establish broad
generality. Fisher's access to retrospective labeled evaluation examples must
be disclosed alongside its stronger discrimination.

No further priority-2 runs are needed for the planned minimal comparison. Preserve
these results for the joint paper revision. Priority 3 (CIFAR learner baselines)
is next; its implementation and bounded timing pilot still need preparation.
