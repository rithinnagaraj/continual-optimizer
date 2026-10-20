# Run integrity and recovery

Scientific outputs are tied to source, configuration, and checkpoint identities.
Use a fresh output tree when those identities change. Source hashes cover the
declared files, including comments and diagnostic text; an editorial source edit
can therefore invalidate resume even when numerical equations are unchanged.

## Configuration commits

E2 atomically records scientific settings, the seed list, and source hashes in
`run_config.json` before data preparation or training. Resume validates that
identity before accepting partial branch results. Unidentified or incompatible
results are rejected. The final grade is also written atomically.

## Checkpoints and journal coverage

The shared paper stream runner resumes from a validated committed checkpoint.
A row for task 200 alone is insufficient evidence of completion. A terminal
checkpoint requires exact scalar, task, and pool coverage, one current domain per
pool evaluation, and agreement with committed row counts.

After an interruption, rows beyond the last checkpoint commit are trimmed and
training continues from that commit. This also handles a final-task CSV write
that completed before its checkpoint was committed. See the mechanistic
[output schema](../cap_mechanistic/OUTPUT_SCHEMA.md) for journal and checkpoint
fields used by that suite.

## Measurement and state contracts

- Replay accuracy is measured without gradients on the complete incoming
  current-task minibatch before replay mixing. Training uses the mixed batch.
  Non-replay methods use their existing forward pass.
- S1 evaluation uses batches of 128; the MNIST mechanism evaluator reads its
  packaged configuration. Earlier experiment protocols declare their own defaults.
- `CAPAdam.add_param_group` validates candidate settings before adding groups,
  infers protection per tensor, and initializes zero importance and independent
  initial-weight anchors. Added biases receive Adam moments without CAP state.

## MNIST reference admission

`mechanistic-mnist-paper` runs complete reference diagnostics, the pinned bounded
same-environment comparison, M1–M6, and analysis, in that order. Any bounded
comparison failure stops intervention execution. Historical endpoint failures
remain recorded, and reports retain their conservative verification labels.
`mechanistic-mnist` instead enforces the strict historical endpoint gate.

Table 15 and Figure 7 collect complete reference endpoint diagnostics, including
failed tolerances, without requiring unrelated interventions. Collecting those
diagnostics does not admit a reference for mechanistic inference.

Source pins cover the MNIST core, bundled historical optimizer/model snapshots,
and upstream UPGD files. A bounded-validation certificate is reusable only when
its source, data, configuration, environment, and checkpoint identities match.

## Regression coverage

`tests/test_cap_paper_contract.py`, `tests/test_paper_run_integrity.py`,
`tests/test_cap_v2_postp1.py`, and `tests/test_public_release.py` cover added
parameters, incoming-batch replay accuracy, E2 configuration commits, terminal
coverage, final-write rollback, and launcher dependencies. These cases are
provided as executable checks; their current execution status is recorded in
[VALIDATION.md](VALIDATION.md).
