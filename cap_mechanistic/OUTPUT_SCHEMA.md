# Output schema

All scientific runs are identified by `cap-mechanisms-final-v1` in immutable
`run_config.json` manifests. Use a new output root for any protocol/source revision.
Synthetic runs are always nested under `smoke/` and are unverified.

## Common provenance

Raw observations include `seed`, `checkpoint_task`, `branch`, `branch_id`,
`source_checkpoint_sha256`, `baseline_config_sha256`, `experiment_config_sha256`,
and `source_version` (SHA-256 of packaged Python/config sources). `intervention`
records the actual changed variable and deterministic shuffle seed.

Individual observations identify their actual immutable input checkpoint. Reference
observations identify their initial checkpoint; branch observations identify their
bank checkpoint. Aggregate rows carry the digest of the input manifest in the
source field; that manifest enumerates the constituent checkpoint/file hashes.
This distinction is required when interpreting aggregate provenance. Training
seed order follows the fixed reporting order in runs; summary grouping keys are
explicit and must not be interpreted as independent samples.

## Reference

`reproduction/raw/<cell>/` stores newline-delimited JSON journals:

| Journal | Observation | Main columns |
|---|---|---|
| `online.jsonl` | Every optimizer update | task, within_task_update, global_update, accuracy (pre-update), loss, bind_rate, batch_indices_sha256 |
| `tasks.jsonl` | Every completed task | early_200, early_auc, plasticity (dense mean), historical_sampled_plasticity, task_end_accuracy, bind_rate_mean, steps_to_85_percent, threshold_reached |
| `pool.jsonl` | Each Stream-B domain at every task boundary | task, domain, accuracy, loss, is_current, revisit_gap |
| `task_features.jsonl` | Stream-A transitions 2–200 | seven pre-task features, next-task outcomes, task index |
| `mechanism_traces.jsonl` | Every 100 updates | bind_rate, median_importance, median_radius, displacement summaries, boundary_fraction, parameter_norm, bound_importance_decay |
| `learning_curves.jsonl` | Steps 100/200/500/1000/2000/5000 | task, within_task_update, pre-update accuracy |

`summary/tasks.csv`, `summary/pool.csv`, and `summary/task_features.csv` export
the corresponding records. `stream_a/` and `stream_b/` also provide separate
task/pool exports. `reproduction_gate.json` contains each expected/observed value,
absolute error, pass flag, source hash, immutable checkpoint hashes, and the dense
P metric separately. `P_historical_100step` alone is compared to historical P.

`checkpoints/a_seed<seed>_task<task>.pt` saves tasks 20/50/100/150/200;
Stream B saves tasks 50/100/150/200. Each checkpoint has:

- model state; Adam first/second moments and step; protected anchor, importance,
  and last proposal contest mask;
- global_update, completed task, within_task_update, stream metadata and probe IDs;
- epoch permutation, cursor, generator state, worker-lifetime semantics, batch size;
- Python, NumPy, CPU and CUDA RNG;
- source/config hashes, branch/intervention identity;
- journal row counts, current-task accuracies/bind sum, previous accuracy,
  trailing 500 bind values, initial old-domain accuracies, last-visited domain map;
- completion marker and pending pre-task features.

At a partial task, `task` counts completed tasks; the active task is `task + 1`.
No optimizer task-boundary signal is applied.

## M1

`raw/m1_group_damage.csv`: seed/checkpoint/layer/fraction/comparator/group/replicate,
selected coordinate count, baseline loss, mean antithetic damage, individual noise
draw damages, and the fixed perturbation seed. Random groups are shared across
comparators of the same layer/fraction, never pooled as independent replicates.

`raw/m1_fisher_rankings.csv`: all protected coordinate indices and ascending
importance/Fisher/magnitude ranks (0 = lowest). Per-block rank files and frozen
Fisher tensors are cached under `raw/<block>/`.

`summary/m1_summary.csv`: top/bottom/random mean damage, top percentile among random
groups (midrank tie convention), top-minus-bottom and top-minus-random effects.
`m1_cluster_intervals.csv` includes mean/median block quantities and seed-cluster CIs.
`m1_importance_faithfulness.png` overlays seed effects for the three fractions.

## Causal continuations (M2/M3/M4/M6)

Each branch has the same journals and `resume.pt` as the reference runner, plus
`retention.jsonl`. Initial retention has `within_task_update=0`; later rows include
`before_accuracy`, `forgetting = before - current accuracy`, `is_current`, and
`revisit_gap` (null until previously seen). Positive forgetting means memory loss.

`raw/mN_branch_metrics.csv` includes mean early AUC/plasticity, held-out accuracy,
retention, forgetting, worst final old-domain forgetting, mechanism summaries,
and paired `delta_*` values relative to the predefined baseline.
M2/M3 old retention is measured at the final continuation boundary. M4/M6 primary
retention averages non-current domains across all continuation boundaries;
final forgetting is separately retained.

`raw/mN_learning_curves.csv`, `mN_mechanism_traces.csv`, and
`mN_old_task_retention.csv` preserve detailed measurements.

- M2: `summary/m2_k_response_summary.csv`, `m2_rank_responses.csv`,
  `m2_k_stability_plasticity.png`, and `m2_k_bind_response.png`.
- M3: `summary/m3_state_reset_summary.csv`, `m3_constraint_tax_by_age.png`,
  `m3_branch_learning_curves.png`, and `m3_retention_cost.png`.
- M4: `raw/m4_shuffle_replicates.csv`, `raw/m4_gap_retention.csv`,
  `summary/m4_summary.csv`, and `m4_true_vs_shuffle.png`.
- M6: `raw/m6_gap_retention.csv`, `summary/m6_release_summary.csv`,
  `m6_rank_responses.csv`, and `m6_release_response.png`.

Every causal experiment also saves `summary/mN_block_effects.csv`. M4 averages
eight shuffle replicates within seed/checkpoint in this table before bootstrapping.
M2 excludes box-off from finite-k Spearman correlations. M6 uses `1/T_m` (zero for
infinity) so increasing x means faster release. No undefined correlations are
converted to favorable numbers; CSVs represent them as missing.

## M5

- `raw/m5_task_features.csv`: reference transitions and next-task outcomes.
- `raw/m5_fold_predictions.csv`: held-out seed/task, model, outcome, observed,
  predicted, alpha, and provenance.
- `summary/m5_model_metrics.csv`: MAE, R², Spearman, alpha per held-out seed and
  overall. Overall metrics use concatenated held-out predictions.
- `summary/m5_coefficients.csv`: standardized coefficients, intercept, training
  means/scales, alpha, seed/model/outcome.
- `summary/m5_prediction_plot.png`: observed versus held-out predicted early AUC.

## Report

`analysis/claim_evidence_table.csv` records claim, evidence experiment, status,
reason, and falsifying outcome. `mechanistic_summary.csv` includes all seed effects,
expected directions, bootstrap means/CIs, and independent seed counts.
`mechanistic_report.md` states the decision rules and verification status.
`figures/` collects the experiment plots. A missing optional M6 is recorded as no
evidence; it is never converted into a successful claim.

Completed data are protected by manifests and explicit `--resume`. Derived CSVs
and plots can be reconstructed from committed journals. An interrupted append is
trimmed to the last atomic checkpoint before further observations are recorded.
