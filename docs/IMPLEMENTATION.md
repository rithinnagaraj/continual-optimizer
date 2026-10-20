# Implementation reference

This guide connects the optimizer and experiment entry points to Section 3.4,
Equation 4, and Appendix A. See [METHOD.md](METHOD.md) for the update equations
and [the README](../README.md#tables-and-figures) for table and figure commands.
`paper_targets.json` defines 24 table targets, nine figure targets, and the
overhead experiment, including dependencies and expected artifacts.

## Optimizer state

`cap_adam.CAPAdam` initializes importance to zero and each anchor to an
independent clone of its parameter's initial value. Parameters added later with
`add_param_group` receive the same initialization and protection inference.
The standalone mechanistic optimizers use the same initial state; the historical
implementation initializes its state before its first gradient update.

The default rule is contest-blocked Release: `k=7e-5`, `T_a=T_s=2000`,
`T_m=1e6`, zero weight decay, and learning rate `1/300`. Bound coordinates ignore
the squared-gradient candidate; infinite `T_m` holds their importance constant.
Matrices and kernels receive protection, while biases use plain Adam. Only
protected coordinates contribute to the bind-rate denominator. With the box
disabled, strict proposal crossings and state observers remain active, while
clipping is skipped.

## Experiment implementations

| Entry point | Protocol |
|---|---|
| `patched_e1.py`, `experiment_e3.py` | Centered importance; E3 also records an independent raw-gradient tracker. |
| `experiment_e2.py` | Proximal spring with importance and anchors frozen at the task switch. Explicit-spring results are incompatible with this protocol. |
| `experiment_paper_development.py` | Original Release, gated anchors, min-candidate Release, final Release, selective boxes, Adam-v evidence, row slack, and momentum reset. Both streams supply Tables 20–23. |
| `experiment_paper_benchmark.py` | Five methods with fixed configurations, P/R/B, prespecified horizons, and retention by revisit gap. Retuning is a separate workflow. |
| `mechanistic-mnist-paper` launcher stage | Complete MNIST reference diagnostics, bounded implementation validation, then M1–M6 and analysis. |
| `mechanistic-cifar-paper` launcher stage | S2 MLP learning rate `2e-4` and S3 CNN learning rate `3e-4`; the Table 6 pilot runs separately. |

The development runner refreshes the global protected-weight importance median
`c` every 100 updates, skipping zero burn-in. Gated anchors use importance from
before the current update. Both mechanistic packages disable matrix TF32, use
deterministic cuDNN settings, and use one CPU thread. S1 benchmark, development,
final-stream, and MNIST mechanism evaluation use batches of 128. Earlier E1–E6
protocols retain their separately declared settings.

Replay plasticity observes the full incoming current-task batch before replay
mixing; the mixed batch supplies the training loss. P uses the prescribed
100-step samples, R excludes the current domain, and B takes the minimum within
seed before aggregation. See [run integrity](RUN_INTEGRITY.md) for checkpoint
and resume contracts.

## Outputs and checks

Table and plot commands use generated measurements. The update schematic is
generated as SVG; runner plots and data need not share the camera-ready layout.
End-to-end wall times are unrecorded; README budgets are update counts.

Numerical tests cover initialization, defaults, plain-Adam biases, bound Release,
infinite timescales, box-off observations, centered evidence, proximal protection,
and development equations. Reporting tests cover aggregation and incomplete-data
rejection. See [validation](VALIDATION.md) for commands and execution status.
Static checks do not establish agreement with published numerical results.
