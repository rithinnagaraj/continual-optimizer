# Priority 4: verified CIFAR update diagnostics

Reviewed 23 November 2026. All 160 matched continuations completed and passed
server-side checkpoint verification and independent local result recomputation.
This completes the four planned experimental priorities. Manuscript edits remain
deferred; this file records evidence for the later paper and rebuttal revision.

## What the additional measurements establish

The new evidence distinguishes persistent importance-dependent restriction from
the transient effect of an anchor's location. Recentring does change the update
geometry immediately, but clipping returns while importance remains intact.
Consequently, describing recentering as having **limited benefit** is more
accurate than saying it does nothing. Its small MLP benefit is measurable.

The CNN is not unconstrained: it has larger absolute projection cost and larger
early-accuracy gains from disabling/resetting protection than the MLP. Its
projection cost largely plateaus after age 50, whereas the MLP's continues to
grow. This separates the magnitude of a constraint cost from its growth with
stream age. It does not establish that architecture alone causes the difference:
models and selected learning rates differ.

These are useful additional mechanism measurements, not proof that CAP is
uniquely inspectable or "more mechanistic" than EWC or UPGD. Keep the paper's
contribution framed as inspectable protection studied with matched interventions.

## Fixed design and metrics

Two architectures × five seeds × four checkpoint ages × four branches = 160
jobs, each continuing for 5,000 updates; 800,000 updates total. Ages are
20/50/100/150, seeds 42/123/451/789/2026. Branches are baseline, box-off,
importance-reset, and recenter-only. All branches restore the same original
weights, Adam moments, sampler and RNG for their seed/age/architecture.
The original selected learning rates are 0.0002 for MLP and 0.0003 for CNN.

Early accuracy is mean pre-update minibatch accuracy over the first 1,000
updates. All accuracy differences below are percentage points, intervention
minus matched baseline. Projection cost is the L2 norm of the update removed
by the box divided by the norm of Adam's proposed update, aggregated across
protected weight matrices at each sampled step, then averaged over steps
100, 200, ..., 1,000 and five seeds. It is a norm ratio, not a percentage of
coordinates or a direct measure of loss damage. Binding fraction is reported
separately. Full-layer diagnostics and early transients are retained in the raw
archive. See [the frozen runbook](PRIORITY4_RUNBOOK.md) for all definitions.

## Learning response and age dependence

| Model | Age | Baseline early accuracy (%) | Projection ratio | Box-off gain (pp) | Importance-reset gain (pp) | Recenter gain (pp) |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 20 | 21.691 | 0.1855 | 1.055 | 0.988 | 0.082 |
| MLP | 50 | 23.078 | 0.2177 | 1.167 | 1.051 | 0.212 |
| MLP | 100 | 24.018 | 0.2593 | 1.436 | 1.310 | 0.250 |
| MLP | 150 | 24.293 | 0.2897 | 1.407 | 1.305 | 0.333 |
| CNN | 20 | 25.728 | 0.4534 | 3.158 | 3.031 | 0.075 |
| CNN | 50 | 28.649 | 0.4883 | 3.786 | 3.715 | 0.116 |
| CNN | 100 | 32.489 | 0.4897 | 3.485 | 3.467 | 0.041 |
| CNN | 150 | 35.087 | 0.4868 | 3.197 | 3.213 | 0.063 |

Importance-reset and box-off benefits are positive in every seed/age/model
combination. Their near agreement is expected from the intervention design;
the added evidence concerns the measured geometry and its persistence.

At age 150, paired seed-bootstrap 95% intervals for importance-reset gains are
MLP **1.305 [1.206, 1.399] pp** and CNN **3.213 [3.009, 3.427] pp**. Recenter
gains are MLP **0.333 [0.301, 0.357] pp** and CNN **0.063 [0.006, 0.115] pp**.
CNN recentering has mixed seed signs at ages 100/150; the age-100 interval
includes zero. These are descriptive percentile intervals over five seeds,
10,000 resamples, without multiplicity adjustment, not strong significance claims.

MLP baseline projection ratio increases by 0.1042 [0.0993, 0.1078] between ages
20 and 150. CNN also increases over those endpoints, by 0.0334
[0.0290, 0.0384], mostly between ages 20 and 50. Therefore do not call the CNN
projection ratio completely age-independent. Its box-off early-accuracy benefit
changes by only +0.039 pp [-0.145, 0.192] over the endpoints, compared with
+0.353 pp [0.139, 0.522] in MLP. Importance-reset endpoint benefit changes are
+0.183 pp [0.051, 0.311] in CNN and +0.318 pp [0.105, 0.475] in MLP; the CNN
benefit is nonmonotonic across all four ages. Do not replace this with a blanket
claim of exactly zero CNN age effects.

Baseline early accuracy itself rises with age in both architectures. The claim
concerns intervention-measured constraint cost, not a decline in absolute
observed accuracy. Baseline binding fractions grow from 2.50% to 5.89% in MLP
and 7.67% to 9.19% in CNN; binding counts alone omit the size of removed updates.

## Why recentering has limited sustained benefit

At age 150, the ratio of removed to proposed update norm evolves as follows
(five-seed means; these are individual sampled updates, not window averages):

| Model / branch | Update 1 | Update 20 | Update 100 | Update 500 | Update 1,000 |
|---|---:|---:|---:|---:|---:|
| MLP baseline | 0.1686 | 0.2894 | 0.2954 | 0.2918 | 0.2818 |
| MLP recentered | 0.0000 | 0.2434 | 0.2376 | 0.2694 | 0.2819 |
| CNN baseline | 0.3914 | 0.4280 | 0.4713 | 0.5013 | 0.4746 |
| CNN recentered | 0.0038 | 0.1972 | 0.4170 | 0.5052 | 0.4771 |

Before the first continuation step, mean absolute anchor distance divided by
radius is 0.4228 in MLP and 0.3628 in CNN; recentering makes it zero by design.
The immediate projection reduction is therefore substantial. By around 500
updates in CNN and 1,000 in MLP, the projection ratios are close to baseline
again. This supports **rapid return of binding under retained importance**, rather
than explaining the limited accuracy response solely by anchors having already
tracked the weights closely. We did not vary the anchor time constant, so its
causal contribution to this time course remains untested.

## Retention cost

At age 150, old-domain accuracy differences after the full continuation task
are:

| Model | Box off (pp) | Importance reset (pp) | Recenter (pp) |
|---|---:|---:|---:|
| MLP | -1.117 | -0.556 | -0.060 |
| CNN | -7.901 | -5.516 | -0.483 |

These are averages over the original ten old-task probes, with five paired
seeds. The plasticity rescue is not free retention. Full seed-level outcomes,
1,000-update retention, and paired intervals are in the machine-readable report.

## Reviewer mapping and appropriate claim

- **R2, important request 1:** a targeted use case connects architecture-dependent
  learning responses to actual proposal/projection measurements and transient
  anchor dynamics. It strengthens the mechanism account within CAP; retain the
  narrower title/abstract framing rather than claiming broad optimizer-independent
  discoveries from this experiment alone.
- **R2, recentering question:** immediate changes followed by rapid rebinding
  support an explanation beyond the unsupported assumption that a 2,000-update
  anchor timescale simply makes anchors irrelevant. Replace categorical null
  wording with the observed small, architecture-dependent benefit.
- **R2, expected-by-design concern:** the importance-reset/box-off comparison
  remains a control. Lead with the transient-versus-persistent distinction and
  the difference between absolute constraint magnitude and its age trend.
- **R2, inspectability comparison:** these results do not establish superiority
  over EWC/online EWC or UPGD as instruments; retain the priority-2 ranking
  evidence and explicit scope limitations.

## Integrity, timing, and artifacts

All 90 original bank checkpoint hashes matched pinned archived manifests;
the original source-tree digest also matched. Renamed optimizer arithmetic is
identical. Data/model/sampler/evaluation functions match. The released seed
helper additionally explicitly disables matmul TF32; the original left the
framework default. Every new branch uses the same recorded environment.
Exact cross-GPU reproduction of historical trajectories is not claimed.

Optimizer parity passed for all branches and both architectures, including RNG
invariance. The mid-task resume test matched model, optimizer, sampler, RNG,
learning records and diagnostics exactly. Every sampled CUDA proposal was also
checked against the actual frozen CAP step during production execution.

The server verified 160 result/checkpoint pairs, finite model and optimizer
states, 800,000 updates, 44,800 trace rows and 4,800 retention rows. Matched
branches had identical batch hashes and initial retention. Local recomputation
matched every nonbinary result hash and all report statistics within 1e-12;
maximum numeric difference was 2.22e-16. Artifact path separators were normalized
between Windows and Linux. Binary checkpoints remain on the instance; their
verified hashes are preserved in the report.

Recorded job time was **0.807932 L4 hours (48.476 minutes)**: 0.336963 MLP hours
and 0.470969 CNN hours. This includes committed pilot prefixes, training,
diagnostics and evaluations; it excludes some startup/final-save overhead and
the later report/download work. The original estimate was 2–8 GPU hours; the
bounded pilot projected 0.935 hours after the banks were recovered.

- [Verified report and per-job summaries](rebuttal_priority4/verified_report.json)
- [Independent local verification](rebuttal_priority4/local_verification.json)
- [Paired age contrasts and transient estimates](rebuttal_priority4/supplementary_contrasts.json)
- [Campaign/source/environment identity](rebuttal_priority4/campaign.json)
- [Pilot review](rebuttal_priority4/pilot_review.json)
- [Source compatibility check](rebuttal_priority4/source_compatibility.json)
- [Scientific figure (SVG)](rebuttal_priority4/diagnostics.svg)

Raw archive: `l4-cifar-diagnostics-feedback.tar.gz`, 56,794,210 bytes.
SHA256: `1997cb94240f56fbfa7d79d053f0ba3e0a57cb0cfcf62497429b12976a719c4e`.
It contains nonbinary job records and provenance, excluding model checkpoints.
Training code was committed as `6030235`; the verified reporter is from
`fda6e1f`. The scientific campaign pins its actual source bytes independently
of subsequent documentation commits. PNG/PDF exports were also generated
locally from `plot_cifar_diagnostics.py`; the SVG is included in the repository.
