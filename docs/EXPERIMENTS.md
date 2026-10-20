# E1–E6 reproduction

Start in the repository root after installing `requirements.txt`. Use the full
default seed sets for scientific runs. Reduced probes/epochs/seeds are debugging
runs and must be labeled accordingly. All outputs below are local to this repo.

## Run order

```bash
python run_experiments.py e1 e1-extra e2 e3 e4a e4a-discriminator phase1-prepare e4b e5 e6 --device cuda --resume
```

The launcher stops at the first failed command. E1 and E3 start fresh even when
the launcher receives `--resume`; their original interfaces have no resume flag.
To run one experiment, pass just its stage. Inspect every underlying command
with the same invocation plus `--dry-run`.

| Experiment | Entry point | Scientific purpose and outputs |
|---|---|---|
| E1 | `patched_e1.py` | Five-seed MNIST/Fashion-MNIST importance and forgetting damage probes; six signals, saved models, individual/group statistics and report in `results/e1/`. |
| E1 extended | `extra_e1.py` | Additional matched-group damage probes and centered-importance autopsy, reading E1 artifacts. Run after E1. |
| E2 | `experiment_e2.py` | Paired box/proximal-spring protection from identical frozen Task-A state, five seeds and separate nine-value k grids; `results/e2/`. |
| E3 | `experiment_e3.py` | State contamination, burn-in, and functional recovery under task changes including EMNIST-Balanced; `results/e3/`. |
| E4a | `experiment_e4a.py` | Long-silence memory and burn-in retest, full 120-cell matrix; `results/e4a/`. |
| E4a discriminator | `experiment_e4a_discriminator.py` | Adam-moment staleness and onset-kick diagnosis; consumes E4a results and writes the regrade used by E4b. |
| E4b | `experiment_e4b.py` | 200-task, 5,000-update/task permuted-MNIST lifetime rigidity stream; `results/stream_a/`. Includes the original reduced four-arm core plus the selected Phase-1 comparators. |
| E5 | `experiment_e5.py` | Drain and single-task decay sanity checks, plus norm grading from E4b scalars; `results/e5/`. |
| E6 | `experiment_e6.py` | CUDA timing/state-memory benchmark: MLP, ResNet-18, GPT-2; AdamW and final CAP with diagnostics enabled/disabled; `results/e6/`. |

Historical experiments retain their original optimizer settings and explicit
deviation disclosures. They do not all use the final contest-blocked Release
configuration. E6 explicitly uses that final configuration. The comprehensive
historical optimizer implementation is `cap_adam_experimental.py`; the public
paper API is `cap_adam.py`. The standalone mechanistic versions
implement only their frozen final configuration and interventions.

## E4b's selection prerequisites

The current E4b runner includes Phase-1 comparisons. From a clean repo these
must be generated before E4b:

```bash
python phase1_tuning.py --data-dir data --device cuda --resume
python phase1_retention_validation.py --data-dir data --device cuda --resume
python experiment_e4b.py --data-dir data --device cuda --resume
```

The first command runs the 41-cell symmetric tuning grid and the UPGD integration
gate. The second preserves the completed v4 tuning evidence, runs disjoint
recurrent validation, and writes the artifact consumed by the P1-A7 stream
runner. A failed learning gate stops the pipeline; no precomputed winner is
silently substituted.

Optional Phase-1 recurrent Stream B and grading:

```bash
python run_experiments.py phase1-stream-b --device cuda --resume
```

Run this after E4b. If `phase1_grade.py` produces a conditional-retune request,
follow its frozen branch explicitly; it is not part of the unconditional run:

```bash
python experiment_e4b.py --device cuda --resume --conditional-retune
python experiment_stream_b.py --device cuda --resume --conditional-retune
python phase1_grade.py
```

This optional historical Phase-1 analysis does not supply results to the fresh
Post-Phase-1 pipeline.

## E5 and E6 details

E5 expects `results/stream_a/stream_scalars.csv` for norm grading; E4b must finish
first. `--drain-only` and `--sanity-only` expose its component runs. Full E5
uses both, with its unchanged final norm check.

E6 creates random synthetic inputs and models, including GPT-2 from configuration;
it needs no pretrained model or dataset download. Install `transformers` to cover
all three workloads. The runner records a missing optional dependency or memory
failure as such rather than treating it as a complete workload. Use an idle GPU,
and inspect every workload's status before interpreting aggregate overhead.

Each raw script supports `--help` for exact operational flags. Use a fresh output
tree after changing scientific source/configuration. Resume never legitimizes
combining incompatible experiment versions.

## Fixed-paper protocols

E1 and E3 explicitly select centered evidence `(g - m_hat)^2`; E3's external
raw-gradient tracker remains a distinct control. E2 selects the proximal spring
`a + (z-a)/(1+lr*I/k^2)` and freezes anchors and importance at the task switch.
Old explicit-spring output files cannot be resumed as proximal results.

`experiment_paper_benchmark.py` runs all five main-table methods with their fixed
paper configurations, without reselecting hyperparameters. It computes P from
100-step observations in the second half of Stream A, R from the final 39
non-current Stream-B domains, and B as the per-seed minimum before averaging.
It also exports the five prespecified horizon scores and revisit-gap retention.

`experiment_paper_development.py` runs all eight development variants on seed 42
by default, on both streams, and exports Tables 20–23. Its global importance
median c is refreshed every 100 updates by the shared stream engine, skipping
zero burn-in. Use the README's table commands for complete dependencies.

E2 commits and validates its configuration/source identity before training, so
partial branch results can be resumed safely. The paper stream runner requires
validated checkpoint commits and full output coverage before accepting a
completed cell. Resume completed paper streams with `--resume`. Replay training
uses the mixed batch, while its reported accuracy observes the incoming batch.
