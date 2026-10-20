# Final Post-Phase-1 experiment

This study is separate from the earlier Phase-1 analysis. All baseline grids,
selection artifacts and final trajectories are generated afresh under
`results/post_phase1_v2/`. It reuses the local protocol engines, models, optimizer
constructors and the bundled UPGD source. No older result tree is required.

## Baselines and retention selection

```bash
python run_experiments.py post-phase1 --device cuda --resume
```

This expands to the following dependency order:

1. `post_phase1_grid.py preflight`: deterministic knob-liveness checks.
2. `post_phase1_pool_preflight.py`: compare the registered pool sizes 20 and 40
   and freeze the headroom decision. A completed decision is immutable; the
   launcher skips this step on resume when it is already present.
3. `post_phase1_grid.py corrected-upgd` then `post_upgd_corrected.py`: the
   separately labeled corrected-UPGD robustness diagnostic.
4. `post_phase1_grid.py retention-grid`: the 114-cell registered six-method grid.
5. `post_phase1_select.py plan-extension`, `post_phase1_grid.py edge-extension`,
   then `post_phase1_select.py freeze-winners`: one registered edge-extension
   round followed by immutable winner selection.
6. `post_phase1_streams.py`: all six selected methods on both final streams and
   five reporting seeds.
7. `post_phase1_report.py`: baseline report and frontier.

The six methods are AdamW, Adam with replay, vendored UPGD, EWC-oracle, CAP
Release, and CAP Ratchet. Replay stores data; EWC-oracle receives task-boundary
information. Their labels and limitations remain explicit in reports.

Grid campaigns and final-stream runners recover completed compatible cells
internally; not all accept an explicit `--resume` flag. The launcher passes flags
only where supported. Resume verifies the frozen decisions downstream. If source
or protocol changes, preserve the old outputs and use a fresh checkout/output tree.

## Final CAP retuning, training and report

After the baseline pipeline finishes:

```bash
python run_experiments.py final --device cuda --resume
```

Equivalent raw commands:

```bash
python cap_release_final.py --retune --data-dir data --device cuda --resume
python cap_release_final.py --train --all-seeds --data-dir data --device cuda --resume
python cap_release_final.py --analyze
```

The bounded retune varies only `k` in `{5e-5, 7e-5, 1e-4}` and `T_m` in
`{500000, 1000000, 2000000}`. The center is `k=7e-5, T_m=1000000`. It uses the
fixed development campaign (primary seed 42; confirmation seeds 31415 and
27182), then trains all five reporting seeds. Seed 42 is shared between primary
development and reporting in the inherited protocol; the two confirmation seeds
are disjoint from reporting. Selection and reporting are therefore not fully
held out from each other.

Outputs are in `results/cap_release_final/`: `retune/`,
`selected_config.json`, `train/tuned/`, an optional center control and `analysis/`.
Analysis requires the freshly generated AdamW/UPGD/EWC baseline tables and run
configurations under `results/post_phase1_v2/`. It checks their scientific identity
and completeness. Copying the final CAP script alone cannot produce this report.

The final experiment requires the registered pool-40 protocol. If the baseline
headroom decision is absent, fails, or selects a different pool, inspect the
diagnostics and stop; do not fabricate a pool decision to pass the guard.

## Frozen final reference

The mechanistic suites freeze the final center configuration: learning rate
1/300, betas (0.9, 0.999), epsilon 1e-8, zero weight decay, `k=7e-5`,
`T_a=T_s=2000`, `T_m=1000000`, matrix-weight protection, moving anchors,
raw squared-gradient importance, coordinate boxes and contest-blocked Release.
They do not dynamically adopt a newly selected retune winner.

An optional historical horizon audit is available with
`python run_experiments.py horizon-audit`. It verifies supplied historical
endpoint means within fixed tolerances; it can reject a new environment's
trajectories. It is separate from the primary final-report command for that reason.
