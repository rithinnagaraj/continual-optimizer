# MNIST reference protocol

## Plasticity

`P_historical_100step` averages Stream-A pre-update accuracy at updates 500100,
500200, ..., 1000000. The reference gate compares this metric with the five-seed
historical targets. Dense accuracy is recorded separately as
`P_dense_all_updates` over updates 500001–1000000; it is not compared with those
targets. Mechanistic early/full-task learning measures use every update.

## Recurrent Stream B

Stream B uses 40 fixed permutations in five shuffled blocks:

```python
pool = [torch.randperm(784, generator=torch.Generator().manual_seed(200001+p))
        for p in range(40)]
g = torch.Generator().manual_seed(555)
schedule = torch.cat([torch.randperm(40, generator=g) for _ in range(5)])
```

There are 200 tasks and five visits per domain. All 40 domains are evaluated at
each boundary; final R averages the 39 non-current domains. M1/M4/M6 use the
same pool-40 schedule. Retained-loss probes average all 40 domains. Fisher uses
200 examples per domain, giving 8,000 per-example gradients, evaluated in
microbatches to limit memory. Both streams evaluate the fixed 2,000-example
test subset in batches of 128.

The corresponding project-level implementations are:

- `cap_release_final.py`: final settings, P/R aggregation, and eight-worker default.
- `cap_v2_postp1_common.py`: `POST_P1_POOL_SIZE = 40` and protocol validation.
- `post_phase1_common.py`: `block_stream_b_schedule` with one generator seeded 555.
- `post_phase1_report.py`: 100-step P aggregation.
- `cap_mech/constants.py`: the five-seed historical P/R targets.

The first four files live at the repository root. They document the shared
protocol; `cap_mech` has no runtime imports from them.

## Persistent-worker sampler

The final stream runner defaults to eight workers and sets
`persistent_workers = num_workers > 0`. Reusing a persistent iterator does not
draw an additional base seed at each epoch; a fresh non-persistent iterator does.
The standalone sampler reproduces the persistent sequence without worker
processes. Tests compare complete multi-epoch sequences, including short batches,
with an actual persistent-worker PyTorch DataLoader.

Changing worker lifetime changes the declared sampling protocol and requires a
new output directory. Matching sampler semantics alone does not establish
agreement with the historical endpoint targets.

## Synthetic validation

Smoke runs use the same pool-40 and persistent-sampler semantics with synthetic
data and reduced horizons. They are labeled unverified. Test commands and
execution status are in [VALIDATION.md](../docs/VALIDATION.md).
