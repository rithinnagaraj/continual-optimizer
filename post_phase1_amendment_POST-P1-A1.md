# POST-P1-A1 — Replay count on the loader's short final batch

**Timing:** recorded after `adam_replay[1]` completed and the first
`replay_count = 96` validation cell stopped before producing a result row.
No `replay_count = 96` Post-Phase-1 cell had completed.

**Mis-specification:** the registered upper bound for `replay_count` assumed
the nominal batch size of 128, but the unchanged Phase-1 MNIST loader preserves
the final short batch of each epoch. That batch contains 96 examples. A
configured `replay_count = 96` therefore attempted to replace all 96 examples
with replay data and violated the registered requirement that at least one
fresh example enter every optimizer step.

**Correction:** the configured replay count remains unchanged and is used
exactly on every full batch. On a short batch only, the effective replay count
is:

```text
min(configured replay_count, actual batch size - 1)
```

Thus a configured count of 96 uses 95 replay examples and one fresh example on
the 96-example final batch. The optimizer records how many such adjustments
occurred in its checkpointed replay state.

**What does not change:** no grid value, buffer capacity, seed, schedule,
threshold, selection rule, tie-break, stream protocol, or previously completed
cell changes. The completed `replay_count` 32 and 64 cells are unaffected
because both counts are already below the 96-example short batch. The stopped
cell had not written a `grid_results.csv` row and is restarted normally by the
existing resume behavior.
