# Priority 3: verified CIFAR baseline results

Updated 23 November 2026. **Priority 3 is complete:** all 40 scientific baseline
jobs finished and passed verification. Manuscript changes remain deferred until
the remaining planned experiments finish. Earlier pilot/search records follow
the final benchmark below.

Addresses reviewer 2's request for AdamW and UPGD on the CIFAR MLP/CNN settings.
The follow-up search also addresses the general concern about under-tuned baselines.

## Final 80-task benchmark

Two architectures × two baselines × five seeds × two streams; each job ran
400,000 updates, for **16 million new updates**. AdamW LR 0.0003 and UPGD LR 0.1
were frozen for both architectures before evaluation. All other optimizer,
data, model, and stream settings retain the recorded protocol.

Values are means ± sample standard deviations across five seeds, in accuracy
fractions. P_80 is stream A pre-update minibatch accuracy sampled every 100 updates
over tasks 41–80. R_80 is final stream B held-out accuracy on the 39 non-current
domains. B_80 is min(P_80,R_80), calculated per seed before aggregation.

| Architecture | Method | P_80 | R_80 | B_80 |
|---|---|---:|---:|---:|
| MLP | AdamW, new L4 | 0.3373 ± 0.0040 | 0.0554 ± 0.0019 | 0.0554 ± 0.0019 |
| MLP | UPGD, new L4 | 0.3525 ± 0.0021 | 0.0555 ± 0.0018 | 0.0555 ± 0.0018 |
| MLP | CAP, archived | 0.3556 ± 0.0018 | 0.0702 ± 0.0013 | 0.0702 ± 0.0013 |
| CNN | AdamW, new L4 | 0.5874 ± 0.0066 | 0.2318 ± 0.0046 | 0.2318 ± 0.0046 |
| CNN | UPGD, new L4 | 0.5285 ± 0.0033 | 0.1953 ± 0.0024 | 0.1953 ± 0.0024 |
| CNN | CAP, archived | 0.3566 ± 0.0044 | 0.2091 ± 0.0065 | 0.2091 ± 0.0065 |

All 30 reported seed/method/architecture rows have R_80 < P_80, so B_80 equals
R_80 throughout this table. The absolute MLP retention values remain low for all
methods. P and R measure different quantities; P is not final held-out accuracy.

**Interpretation:** historical CAP has the highest mean balanced score for the
MLP, while AdamW has the highest for the CNN. On the CNN, AdamW exceeds archived
CAP on both mean P and R. CAP retains more than UPGD on the CNN but has much lower
P. CAP's MLP mean P is only slightly above UPGD's; do not inflate this small gap.
These results support an architecture-dependent performance assessment, not a
claim that CAP consistently outperforms these baselines.

CAP entries are matching **historical 80-task prefixes**, not fresh L4 controls.
The source/configuration checks and historical environment limitations remain in
[the pinned reference](cifar_cap_reference/prefix_80.json). New baseline tuning
used 13 LRs per architecture versus CAP's historical ten, and a disjoint
development seed rather than CAP's seed 42. UPGD's non-LR settings were inherited
from the MNIST incumbent. Do not describe this as equal lifetime tuning effort,
a matched environment superiority test, or a comparison at the 200-task endpoint.
No cross-method significance claim is made here.

The new baseline table adds learner context to the mechanistic experiments. It
does not itself identify why the architecture-dependent outcomes arise. Priority
4 remains the targeted matched-intervention follow-up; no additional priority-3
training is needed for the planned minimal comparison.

## Evaluation verification and measured cost

Training ran from **21 November 2026, 11:00:56 IST** to **22 November 2026,
03:52:54 IST**. The SSH monitoring connection dropped during the CNN runs, but
the evaluation continued independently: 40 start events, 40 completion events,
and no recorded errors, divergences, or job restarts.

| Architecture / baseline | Jobs | Recorded job hours |
|---|---:|---:|
| MLP / AdamW | 10 | 3.1775 |
| MLP / UPGD | 10 | 3.9270 |
| CNN / AdamW | 10 | 4.2126 |
| CNN / UPGD | 10 | 5.5448 |
| Total | 40 | **16.8620** |

Elapsed campaign time including gaps between jobs was 16.8662 hours. The measured
cost was close to the 16.58-hour projection and below the 20–24-hour reserve.
Initial and extended tuning together added 0.3081 recorded hours. These figures
are logged wall durations, not GPU utilization or cloud billing measurements.

Verification covered all 40 job specifications and source hashes, unchanged
search evidence, shared initialization within architecture/seed, consistent data
and environment identities, stream schedule agreement, 3,200 task records,
160,000 sampled training records, and 64,000 pool records. All 240 nonbinary
evaluation artifacts passed local hash checks. Per-seed P/R/B and aggregate
means/SD were independently recomputed from CSVs. Remote and local reports agree
to floating-point precision; the largest numerical difference was 2.78e-17.

The remote release audit and report gate passed, including hashes of all required
checkpoint payloads and exports. The checkpoint bytes remain on the L4; they
were not downloaded or deserialized locally. The archive transfer SHA-256 matched.

Compact evidence:

- [Benchmark table](rebuttal_priority3/benchmark.csv) and [per-seed metrics](rebuttal_priority3/per_seed.csv).
- [Independent verification, timings, and limitations](rebuttal_priority3/evaluation_verification.json).
- [Evaluation manifest](rebuttal_priority3/evaluation_manifest.json), [frozen campaign](rebuttal_priority3/evaluation_campaign.json), and [final selection](rebuttal_priority3/selected_lrs.json).
- [Environment](rebuttal_priority3/evaluation_environment.json), [data identity](rebuttal_priority3/evaluation_data_identity.json), and [remote verification](rebuttal_priority3/remote_verification.json).

Evaluation archive SHA-256:
`14f9a96b7dc360527354bb9eb727620fa9261f2cde37ff2f76ae6b80e3557b0d`.
Scientific source commit: `3f155575b4d9b5c7359eafdf8c8e2b10b518fc49`.
The existing extension campaign hash below is unchanged.

## Completed extension and frozen final selections

All 12 extension trials completed at 5,000 updates with no numerical divergences
or logged infrastructure errors. Their 72 nonbinary result artifacts passed hash
checks; source, environment, data, initialization, CSV coverage, and timing records
were verified. All original campaign evidence was byte-identical to the previously
audited archive. Selection was independently recomputed over **all 52 trials**.
There were no evaluation jobs in the returned feedback.

| Architecture | Baseline | Final selected LR | Train-holdout accuracy | Boundary winner |
|---|---|---:|---:|---|
| MLP | AdamW | 0.0003 | 26.40% | No |
| MLP | UPGD | 0.1 | 26.00% | No |
| CNN | AdamW | 0.0003 | 29.72% | No |
| CNN | UPGD | 0.1 | 27.10% | No |

UPGD's MLP validation accuracies at LR 0.03 / 0.1 / 0.3 were 21.40% / 26.00% /
25.36%; CNN accuracies were 20.74% / 27.10% / 26.48%. The interior winners resolve
the original boundary concern. The high-rate AdamW trials all achieved 1% accuracy
with finite losses; these are poor completed trials, not numerical divergences.
The extension took **4.279 minutes**, making total recorded tuning time
**18.485 minutes**. No further tuning extension was needed. The 40-job evaluation
used `experiment_cifar_lr_extension.py` and its merged frozen selection, rather
than the original ten-rate launcher's selection.

These are single-seed, single-task validation results. They do not establish
relative continual-learning performance or significance of the LR differences.
UPGD's validation improvement over its former boundary choice is 9.68 percentage
points for MLP and 15.92 points for CNN. The completed full-stream evaluation is
reported above.

Evidence: [extension audit, all 12 trials, and merged selections](rebuttal_priority3/extended_search_review.json).
Extension feedback SHA-256:
`f2ea911846206918d5d37a9876d50c5dafab90f5fb9e9f5d3fda65d29806b111`.
Extension campaign SHA-256:
`17b6026d363861c3b0798525fa48e49b78ed732817beceda57fcb326e18f70c3`.
Checkpoint bytes remain on the L4 and were not present for local audit.

## Verified evidence

The four timing pilots completed, followed by all 40 initial search trials:
two architectures × two baselines × ten learning rates, 5,000 updates each.
No numerical divergences or logged infrastructure errors occurred in the search.
The search took **14.206 minutes** summed over recorded job durations, excluding
any gaps between jobs. All 240 nonbinary result artifacts passed SHA-256 checks;
the campaign/source identity, job specifications, final-validation selections,
CSV coverage, environment/data identities, and timing arithmetic were verified.
All original pilot evidence in the search archive was unchanged. Checkpoint
payloads were omitted from feedback; their recorded hashes could not be verified
against bytes locally. The L4 runner validates them before reuse.

Development seed 31415; 45,000 training images and 5,000 stratified train-holdout
images; one patch-permuted task. No official-test examples were used. Within
each architecture all candidates shared the initial model and prescribed sampler.

| Architecture | Baseline | Best initial LR | Final validation accuracy | Boundary winner |
|---|---|---:|---:|---|
| MLP | AdamW | 0.0003 | 26.40% | No |
| MLP | UPGD | 0.01 | 16.32% | Upper |
| CNN | AdamW | 0.0003 | 29.72% | No |
| CNN | UPGD | 0.01 | 11.18% | Upper |

UPGD validation accuracy was still rising toward the upper end of the grid in
both models. Its initial selected LR is therefore provisional. These accuracies
must not be interpreted as evidence of comparative continual-learning performance.

Evidence: [machine-readable audit and all 40 trials](rebuttal_priority3/initial_search_review.json).
Search archive SHA-256:
`a76a056fa69a27f277f1ca899d88185a6b6eeceb4db670bfe46ba2443be9428d`.
Parent campaign SHA-256:
`523b31bb34f746ac92ff3634fa26c1e7a914ee0b0d162d2a3acbd5b11be10ae9`.
Pilot archive SHA-256:
`d3542fd7314905ded8a1595ecc1965688c3aa8c6dd21fe394f851d41ecdd5fad`.

## Bounded follow-up design (now completed)

Add rates **0.03, 0.1, 0.3** for both baselines and both architectures: 12 new
jobs, 60,000 total updates. Keep every other setting, seed, train/holdout split,
and final-validation selection rule unchanged. Reuse the original 40 trials
as immutable evidence; select over all 13 rates per baseline/architecture.
Equal extra budgets for both baselines simplify disclosure, although AdamW's
initial winners were interior. Historical CAP had ten rates, so do not claim
equal tuning budgets across all three methods after this extension.

The extension uses separate output `results/cifar_baselines_extended`, a pinned
parent selection/campaign, and the unchanged tested numerical worker. Original
source hashes, results, and selections are preserved. Initialization verifies the
parent trials and refuses if parent benchmark jobs have already started. Missing
trials or changed artifacts block selection; numerical divergence is recorded and
excluded. A new boundary winner blocks evaluation for review. There is no automatic
further grid expansion or benchmark launch. Commands are in the
[extension runbook](PRIORITY3_LR_EXTENSION.md).

## Pre-evaluation runtime estimates on one L4

Using the final selected LR's steady training speed (last four 1,000-update
blocks) and measured pilot single-domain / 40-domain evaluation costs:

| Architecture / baseline | ms/update | Five seeds × both 80-task streams |
|---|---:|---:|
| MLP / AdamW | 2.702 | 3.13 hours |
| MLP / UPGD | 3.369 | 3.86 hours |
| CNN / AdamW | 3.582 | 4.12 hours |
| CNN / UPGD | 4.791 | 5.46 hours |
| Total | | **16.58 hours projected** |

The planning reserve was **20–24 GPU-hours** for the 40 baseline evaluations. The extension is
complete and its selected-rate timings are included above. Projections extrapolate from
one initial task, omit some setup/checkpoint overhead, and assume an otherwise idle
L4. Evaluation uses 50k training images rather than the search's 45k. The pilots
allocated at most about 134 MiB of PyTorch GPU tensor memory; this excludes device
context and allocator reservations. The L4 is sufficient for this workload.

The first pilot spent about 28.5 minutes before its ready event; the feedback
does not establish the cause. Subsequent cached setup was about 2.6 seconds per
job. Do not charge that first setup delay to every training trial.

The completed benchmark reports P_80, R_80, and per-seed min(P_80,R_80),
with five-seed means/sample SD, alongside explicitly historical CAP prefixes.
No fresh CAP training is included in the default budget.
