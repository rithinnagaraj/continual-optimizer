# Interpretation to preserve for the joint revision

Updated 23 November 2026 at the author's request. This is a results/synthesis
record, not a manuscript edit. Paper revisions remain deferred until the planned
experiments are complete.

## Contribution framing

> CAP-Adam provides explicit protection controls that support matched
> interventions and reveal distinct sources of stability and plasticity loss,
> while achieving a competitive stability–plasticity balance without replay or
> task boundaries.

The distinctive contribution is explicit constraint geometry (radius, anchor
displacement, and binding), separable interventions on protection state, and the
nontrivial M3/M4 findings. Ground the interpretation in the studied settings and
within-method interventions; do not imply a general explanation of all continual
learning optimizers. Describe expected-by-design dial/reset effects accordingly.

Do not claim that CAP is the best optimizer, has the best importance estimator,
or is uniquely inspectable. The UPGD probes demonstrate that matched inspection
extends to another optimizer. They complement the case for useful explicit
controls, rather than establishing an exclusive capability.

## Performance context

Historical five-seed benchmark means from the research workspace's
`results/main_res/safe_retune_and_horizon_findings_report.md`:

| Method | Plasticity P | Retention R | Balanced B |
|---|---:|---:|---:|
| CAP, final tuned configuration | 0.902670 | 0.907367 | 0.902670 |
| UPGD | 0.912714 | 0.924005 | 0.912714 |
| AdamW | 0.964505 | 0.796954 | 0.796954 |
| EWC-oracle | 0.786380 | 0.933626 | 0.786380 |
| Adam with replay | 0.985264 | 0.929059 | 0.929059 |

CAP has better balanced performance than AdamW and EWC-oracle here, while UPGD
exceeds CAP on both P and R. Replay has a higher balanced score as well. The
expanded UPGD search retained its incumbent and reproduced its historical summary;
it leaves this performance conclusion unchanged. Only UPGD was rerun in priority 1;
the other rows remain historical results. "Competitive" must be read in this
qualified context, not as a claim of leading performance or equal resource access.

## Qualitative M1 comparison at final checkpoint and 1% fraction

This is a descriptive comparison assembled after the priority-2 results, not a
new matched cross-method experiment. The CAP subset is checkpoint task 200,
fraction 0.01, importance/Fisher rows from the historical mechanistic M1 table.
UPGD uses its own final task-200 models from priority 1.

| Host model and ranking | Top > random mean | Top > bottom |
|---|---:|---:|
| CAP importance | 15/15 | 15/15 |
| Fisher on CAP | 15/15 | 15/15 |
| Native signed UPGD utility | 15/15 | 7/15 |
| Fisher on UPGD | 15/15 | 15/15 |

CAP importance and Fisher on CAP also have random mean > bottom in all 15 cells.
Thus CAP importance shows a cleaner qualitative high-to-low sensitivity ordering
in these **within-model probes**, while native signed UPGD utility is not a clean
monotonic sensitivity ranking. The UPGD aggregate positive top-minus-bottom gap
must be accompanied by its hidden-layer reversals.

Evidence: [historical CAP subset](rebuttal_priority2/cap_final_1pct_subset.csv),
[subset provenance and counts](rebuttal_priority2/cap_comparison_provenance.json),
and [UPGD layer results](rebuttal_priority2/layer_results.csv). The CAP source
file's hash and original checkpoint/configuration provenance are retained.

Required qualifications alongside this comparison:

- Fifteen cells are repeated measurements from **five independent seeds**, not
  fifteen independent replications or a separate superiority significance test.
- CAP and UPGD are different trained hosts and environments: historical CAP
  mechanistic A30 runs versus the current L4 UPGD runs. This is not a matched
  test of superiority between methods. Scope is the final checkpoint/1% fraction.
- Do not compare raw damage magnitudes across hosts as evidence that one
  optimizer is better. Greater damage may simply reflect greater model sensitivity.
- Fisher is a retrospective diagnostic using labeled evaluation examples, not
  online EWC or an independently evaluated replacement optimizer.
- The historical CAP mechanistic endpoint-reproduction gate failed. Its separate
  same-environment bounded validation does not establish million-update endpoint
  equality or erase the original reproduction caveat.

The defensible synthesis is usefulness and distinctive explicit controls, with
qualified empirical support for importance ordering—not general superiority.

## CIFAR baseline context after priority 3

All 40 new baseline jobs are complete and verified. At the matching 80-task
horizon, mean balanced scores for MLP are CAP archived 0.07024, UPGD 0.05548,
and AdamW 0.05537. CNN scores are AdamW 0.23178, CAP archived 0.20909, and
UPGD 0.19532. All reported per-seed balanced scores equal retention in this setting.

The CNN result limits any broad performance claim: AdamW exceeds historical CAP
on both mean plasticity (0.58742 versus 0.35661) and retention (0.23178 versus
0.20909). CAP has greater CNN retention than UPGD but lower plasticity. Historical
CAP has the strongest MLP retention/balanced means, although all MLP retention
values are low. These comparisons use archived CAP prefixes, not fresh matched
L4 controls; tuning scope/development seeds and environments have disclosed
differences. Treat them as qualified benchmark context, not a superiority test.

The contribution should remain explicit protection controls and matched causal
interventions, with architecture-dependent empirical performance. Do not turn
the CIFAR ranking into a causal explanation for the architecture contrast. The
priority-4 diagnostics below directly measure constraint geometry. See
[full CIFAR results, uncertainties, and provenance](PRIORITY3_RESULTS.md).

## Mechanism measurements after priority 4

All 160 matched continuations are complete. Early projection cost (removed /
proposed update norm) increases from 0.185 to 0.290 across ages 20–150 in MLP.
CNN has larger absolute costs, 0.453 at age 20 and approximately 0.487–0.490
at ages 50–150. Its weak learning-response age trend does not imply absence of
constraints. Absolute baseline accuracy rises with age in both models; discuss
the intervention-measured constraint cost rather than absolute learning decline.

Recentring sharply reduces projection immediately, but it returns near baseline
within about 500 updates in CNN and 1,000 in MLP. Importance remains intact.
This is stronger evidence for transient relief followed by rebinding than for
the speculation that anchors were already so close that recentering does nothing.
Recenter gains at age 150 are small, not zero: +0.333 pp in MLP and +0.063 pp
in CNN, versus importance-reset gains of +1.305 and +3.213 pp. There are retention
costs, especially in CNN. See [verified diagnostics](PRIORITY4_RESULTS.md).

These measurements strengthen the within-CAP mechanism account and provide a
concrete diagnostic use case. They do not show that CAP is a uniquely superior
instrument, that architecture alone causes the differences, or that every CNN
age contrast is exactly zero. Retain the narrower instrument framing and
acknowledge the different learning rates, five-seed uncertainty, and historical
source/environment compatibility boundary.

## Results records

- [Priority 1](PRIORITY1_RESULTS.md): expanded UPGD tuning and benchmark reproduction.
- [Priority 2](PRIORITY2_RESULTS.md): native utility/Fisher/random probes, effect
  sizes, uncertainty, and layer-level qualifications.
- [Priority 3](PRIORITY3_RESULTS.md): completed CIFAR baselines and qualified
  comparison with archived CAP prefixes.
- [Priority 4](PRIORITY4_RESULTS.md): completed proposal/projection diagnostics,
  transient recentering effects, age trends and paired retention costs.
