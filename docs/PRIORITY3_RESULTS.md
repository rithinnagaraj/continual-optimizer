# Priority 3: verified CIFAR pilot and initial LR search

Updated 21 November 2026. Scientific baseline evaluation is **pending**. These
are timing and train-holdout selection results, not continual-learning benchmark
results. Manuscript changes remain deferred until the planned experiments finish.

Addresses reviewer 2's request for AdamW and UPGD on the CIFAR MLP/CNN settings.
The follow-up search also addresses the general concern about under-tuned baselines.

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

## Bounded follow-up decided before benchmark evaluation

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

## Runtime estimates on one L4

Using the selected initial LR's steady training speed (last four 1,000-update
blocks) and measured pilot single-domain / 40-domain evaluation costs:

| Architecture / baseline | ms/update | Five seeds × both 80-task streams |
|---|---:|---:|
| MLP / AdamW | 2.702 | 3.13 hours |
| MLP / UPGD | 3.312 | 3.80 hours |
| CNN / AdamW | 3.582 | 4.12 hours |
| CNN / UPGD | 4.796 | 5.47 hours |
| Total | | **16.52 hours projected** |

Reserve **20–24 GPU-hours** for the 40 baseline evaluations and **5–7 minutes**
for the 12-trial extension (about 4.3 minutes from original average trial durations).
Update the estimate if extended-rate timings differ. Projections extrapolate from
one initial task, omit some setup/checkpoint overhead, and assume an otherwise idle
L4. Evaluation uses 50k training images rather than the search's 45k. The pilots
allocated at most about 134 MiB of PyTorch GPU tensor memory; this excludes device
context and allocator reservations. The L4 is sufficient for this workload.

The first pilot spent about 28.5 minutes before its ready event; the feedback
does not establish the cause. Subsequent cached setup was about 2.6 seconds per
job. Do not charge that first setup delay to every training trial.

The eventual benchmark still reports P_80, R_80, and per-seed min(P_80,R_80),
with five-seed means/sample SD, alongside explicitly historical CAP prefixes.
No fresh CAP training is included in the default budget.
