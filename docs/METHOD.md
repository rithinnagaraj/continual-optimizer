# Optimizer and experiment variants

CAP-Adam augments Adam with an anchor and an importance value for each protected
weight coordinate. Importance controls a box around the moving anchor: greater
importance gives a smaller allowed displacement. The release rule lets a
persistently contested coordinate become less constrained over time.

## Final contest-blocked Release

At update t, let `g` be the current gradient, `a` the previous anchor, `s` the
previous nonnegative importance, and `z` the ordinary bias-corrected Adam proposal.
All expressions below are coordinate-wise, using the state from before its update:

```text
radius       = k / (sqrt(s) + epsilon)
contested    = (z < a - radius) or (z > a + radius)
theta_new    = clip(z, a - radius, a + radius)
anchor_new   = a + (theta_new - a) / T_a
candidate    = s + (g^2 - s) / T_s
importance_new = s * (1 - 1/T_m)       if contested
                 max(s, candidate)   otherwise
```

The contest mask uses strict inequalities before clipping. A contested coordinate
ignores the squared-gradient candidate; finite release time decays its existing
importance. For M6's infinite `T_m`, contested importance stays unchanged.
Importance starts at zero and the anchor starts at the initial weight.

The final MNIST reference uses learning rate 1/300, Adam betas (0.9, 0.999),
epsilon 1e-8, zero decay, `k=7e-5`, `T_a=T_s=2000`, and `T_m=1000000`.
Dense weight matrices are protected; biases retain ordinary Adam updates.
CIFAR additionally protects convolution kernels. Its MLP and CNN do not use
BatchNorm or dropout, keeping frozen functional probes deterministic.

The `box_off` intervention skips clipping but keeps anchor/importance observers
and the pre-projection contest mask active. Other interventions change only their
declared optimizer-state component while keeping paired initialization, sample
order and Adam state as specified in each experiment.

## Which implementation to read

| File | Role |
|---|---|
| `cap_adam.py` | Public optimizer API: defaults to final contest-blocked Release, zero decay, k=7e-5, Ta=Ts=2000, Tm=1e6. Infers protected matrices/kernels and leaves biases on Adam. |
| `cap_adam_experimental.py` | Explicit historical implementation used by the older experiment constructors. Its defaults are historical; it is not the default optimizer API. |
| `cap_mechanistic/cap_mech/optim/cap_final.py` | Minimal final optimizer plus explicit mechanistic intervention switches, with frozen MNIST configuration. |
| `cap_mechanistic_cifar/cifar_mech/optim/cap_final.py` | Standalone CIFAR version with the same final equations. |
| `phase1_arms.py` | Comparator constructors, replay and EWC state, and pinned upstream UPGD loader. |
| `dn4il_upgd_corrected.py` | Separately labeled corrected-UPGD scale-study variant. |

E1–E5 preserve their historical protocol settings and their reports disclose
changes/deviations. Applying the final settings retroactively to them would
change the experiments. E6 and the mechanistic studies explicitly select the
final optimizer. Final CAP retuning changes only its registered k and release
timescale grid, while the mechanistic center stays frozen.

## Development variants

E1/E3 explicitly select centered importance; their raw tracker remains
independent. E2 uses the proximal spring
with frozen importance and anchors. The development runner explicitly implements
original Release, slower importance-gated anchors, bound min(I,C) Release,
final contest-blocked Release, selective radius I²/(I+c), Adam-v evidence,
Q4 row gating and conflict momentum reset. Anchor gating uses pre-update I.
Both importance-dependent variants use the refreshed global protected-weight
median c. The fixed main benchmark and CIFAR paper-rate launcher preserve the
paper's reported hyperparameters; retuning remains a separate historical study.
