# CAP-Adam

CAP-Adam is an Adam-based optimizer for continual learning. It tracks coordinate
importance and moving anchors, constrains updates with coordinate boxes, and
uses contest-blocked Release to relax protection as tasks change.

This repository contains the optimizer, continual-learning benchmarks,
mechanistic interventions, and analysis scripts. It includes E1–E6, the final
Post-Phase-1 experiment, MNIST mechanisms M1–M6, CIFAR-100 mechanisms M1–M4,
and additional frozen ResNet-50 scale studies. All project code is contained
here; no parent directory or separate project checkout is required.

## Installation

Start in the repository root and create a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
```

Install a compatible PyTorch/torchvision pair for your CPU or CUDA environment,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The recorded scientific environment used Python 3.10.20 and PyTorch
2.11.0+cu130. The requirements specify minimum package versions, rather than an
exact environment lock. Keep the environment consistent within a campaign;
runners record versions, source hashes, seeds, and configuration identities.
See [environment details](docs/ENVIRONMENT.md) for setup and numerical limitations.

MNIST, Fashion-MNIST, EMNIST, and CIFAR-100 download on their first relevant
full run. Data live under `data/`, and results under `results/`. The optional
CIFAR learning-rate pilot needs CIFAR-100 downloaded first; its table command
handles this. Scale studies need separately supplied datasets and pretrained
weights. See [data preparation](docs/DATA.md) for layouts and dependencies.

Full experiments are intended for a CUDA GPU. E6 requires an otherwise idle
CUDA GPU; other runners also accept CPU. Synthetic smoke runs need no dataset
downloads.

## Using the optimizer

```python
from cap_adam import CAPAdam

optimizer = CAPAdam(model.parameters(), lr=1 / 300)

for inputs, targets in loader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

The default implementation uses contest-blocked Release (Equation 4). Defaults
are `k=7e-5`, `T_a=T_s=2000`, `T_m=1_000_000`, Adam betas `(0.9, 0.999)`,
`eps=1e-8`, and zero weight decay. Importance starts at zero, and each anchor
starts at its parameter's initial value.

Matrix weights and convolution kernels are protected by default.
One-dimensional parameters, including biases, use plain Adam and are excluded
from the bind-rate denominator. Parameters added with `add_param_group` follow
the same initialization and protection rules. Setting `box_enabled=False`
disables clipping while retaining crossing counts and state updates.

See [method details](docs/METHOD.md) for equations and state semantics.
`cap_adam_experimental.py` contains the earlier variants used by the development
experiments; those runners select their variants explicitly.

## Running the experiments

To list stages or inspect commands without training:

```bash
python run_experiments.py --list
python run_experiments.py mechanistic-mnist-paper --dry-run --device cuda
```

`--dry-run` needs only the Python standard library. It creates no outputs and
downloads no data. The launcher resolves paths from the repository root, so it
also works when invoked from another working directory.

### Main benchmark and mechanisms

```bash
# Five-method, five-seed MNIST benchmark (Table 3 and Figure 3)
python experiment_paper_benchmark.py --device cuda --resume

# MNIST reference and mechanisms M1–M6
python run_experiments.py mechanistic-mnist-paper --device cuda --resume

# CIFAR-100 MLP and CNN references and mechanisms M1–M4
python run_experiments.py mechanistic-cifar-paper --device cuda --resume
```

S1 uses an MLP on two MNIST streams with learning rate `1/300`. S2 and S3 use
CIFAR-100 with an MLP at `2e-4` and a CNN at `3e-4`, respectively. Each reference
uses five seeds, two streams, 200 tasks per stream, and 5,000 updates per task.
The benchmark compares CAP-Adam, AdamW, UPGD, EWC, and replay. Its metrics are:

- **P:** Stream-A accuracy sampled every 100 updates in the second half of training.
  Replay is evaluated on the incoming batch before replay mixing.
- **R:** Final Stream-B accuracy averaged over the 39 non-current domains.
- **B:** The minimum of P and R within each seed, then averaged across seeds.

Mechanistic experiments use paired branches from reference checkpoints. MNIST
checks complete reference banks and performs bounded comparisons against bundled
reference implementations before running interventions. Historical endpoint
discrepancies remain reported, and bounded checks do not certify agreement over
the entire training trajectory. The separate `mechanistic-mnist` stage enforces
the strict historical endpoint gate. See [mechanistic protocols](docs/MECHANISTIC.md)
for admission checks, intervention definitions, and individual-stage commands.

### E1–E6 and Post-Phase-1

```bash
# Earlier importance, protection, state, lifetime, and overhead experiments
python run_experiments.py e1 e1-extra e2 e3 e4a e4a-discriminator phase1-prepare e4b e5 e6 --device cuda --resume

# Baseline selection, final CAP-Adam retuning, training, and reporting
python run_experiments.py post-phase1 final --device cuda --resume

# Eight development variants on both streams (Tables 20–23)
python experiment_paper_development.py --device cuda --resume
```

Stages run in the order given and stop on failure. `phase1-prepare` generates
the selection artifacts needed by E4b; E5 reads E4b outputs. The final
Post-Phase-1 report needs the baselines generated by `post-phase1`. This search
and retuning workflow is separate from the fixed-configuration main benchmark.

`--resume` requests recovery where supported. E1 and E3 start fresh because
their interfaces do not support resume. Scientific runners reject incompatible
source/configuration identities; use a fresh output tree after changing them.
The historical stream loaders default to eight persistent workers. Changing
the worker count can change the sampled trajectory.

For the five-seed comparison of the original and final Release rules:

```bash
python experiment_paper_development.py --variants original final --seeds 42 123 451 789 2026 --device cuda --resume
```

Detailed guides: [E1–E6](docs/EXPERIMENTS.md),
[Post-Phase-1](docs/POST_PHASE1.md), and
[mini-ImageNet / DN4IL scale studies](docs/SCALE_STUDIES.md).

## Tables and figures

Each command below runs the required experiments and analysis, then records
the generated source artifacts and their hashes in
`results/paper/<target>/source_artifacts.json`. Tables are supplied as source
data and figures as runner plots; camera-ready composite layout is separate.
Several targets share a campaign and reuse compatible completed outputs where
the underlying runner supports resume.

```bash
python reproduce_paper.py --list
python reproduce_paper.py table-03                 # Inspect dependencies only
python reproduce_paper.py table-03 --run           # Generate results
python reproduce_paper.py table-03 --collect-only  # Index existing results
```

End-to-end wall times were not recorded. The tables give update budgets where
available, rather than estimates in hours. The recorded hardware was NVIDIA
L4 for the S1 benchmark, development experiments, and E6 timing; NVIDIA A30 for
S1 mechanisms and the CIFAR learning-rate pilot. The full CIFAR mechanism GPU
model and earlier E1–E4 GPU models were not separately recorded.

A reference campaign costs 10 million updates per setting before interventions.
The three settings therefore require 30 million reference updates in total,
plus branch training and probes. The main benchmark costs 50 million updates.
The eight-variant development campaign costs 16 million updates, with another
2 million for the UPGD retention-gap reference. These are campaign budgets;
shared targets should not be counted as independent full campaigns.

In the tables, **S1–S3** means A30 for S1 and an unrecorded GPU model for S2/S3.
**Unrecorded** in the runtime column always means no measured end-to-end wall time.
Metadata-only exports involve no training and are expected to take seconds
(an unmeasured estimate).

| Table | Command | Hardware | Runtime / compute budget |
|---|---|---|---|
| 1: Settings | `python reproduce_paper.py table-01 --run` | CPU | Seconds estimated; metadata only |
| 2: Headline M1–M4 contrasts | `python reproduce_paper.py table-02 --run` | S1–S3 | Unrecorded; references + mechanisms |
| 3: S1 benchmark P/R/B | `python reproduce_paper.py table-03 --run` | L4 | Unrecorded; 50M updates |
| 4: Baseline hyperparameters | `python reproduce_paper.py table-04 --run` | CPU | Seconds estimated; metadata only |
| 5: Tuning budgets | `python reproduce_paper.py table-05 --run` | L4 (S1 environment) | Unrecorded; registered grids and extensions |
| 6: CIFAR learning-rate pilot | `python reproduce_paper.py table-06 --run` | A30 | Unrecorded; 100k updates |
| 7: Intervention design | `python reproduce_paper.py table-07 --run` | CPU | Seconds estimated; metadata only |
| 8: Per-seed effect signs | `python reproduce_paper.py table-08 --run` | S1–S3 | Unrecorded; references + mechanisms |
| 9: Importance/Fisher/magnitude damage | `python reproduce_paper.py table-09 --run` | S1–S3 | Unrecorded; shared mechanism campaigns |
| 10: Box-scale responses | `python reproduce_paper.py table-10 --run` | S1–S3 | Unrecorded; shared mechanism campaigns |
| 11: State-reset responses | `python reproduce_paper.py table-11 --run` | S1–S3 | Unrecorded; shared mechanism campaigns |
| 12: Importance-assignment responses | `python reproduce_paper.py table-12 --run` | S1–S3 | Unrecorded; shared mechanism campaigns |
| 13: Prospective prediction | `python reproduce_paper.py table-13 --run` | A30 | Unrecorded; S1 reference + mechanisms |
| 14: Release-timescale response | `python reproduce_paper.py table-14 --run` | A30 | Unrecorded; S1 reference + mechanisms |
| 15: Historical endpoint comparison | `python reproduce_paper.py table-15 --run` | A30 | Unrecorded; 10M reference updates |
| 16: CIFAR reference P/R | `python reproduce_paper.py table-16 --run` | CIFAR GPU unrecorded | Unrecorded; 10M reference updates per architecture + mechanisms |
| 17: CIFAR absolute baseline branches | `python reproduce_paper.py table-17 --run` | CIFAR GPU unrecorded | Unrecorded; references + mechanisms |
| 18: Earlier importance validation | `python reproduce_paper.py table-18 --run` | GPU unrecorded | Unrecorded; E1 training + probes |
| 19: Centered/raw contamination | `python reproduce_paper.py table-19 --run` | GPU unrecorded | Unrecorded; E3 training + diagnostics |
| 20: Development variants | `python reproduce_paper.py table-20 --run` | L4 | Unrecorded; 16M updates |
| 21: Late Stream-A windows | `python reproduce_paper.py table-21 --run` | L4 | Unrecorded; shared development campaign |
| 22: Retention by revisit gap | `python reproduce_paper.py table-22 --run` | L4 | Unrecorded; development + 2M UPGD updates |
| 23: Within-task accuracy | `python reproduce_paper.py table-23 --run` | L4 | Unrecorded; shared development campaign |
| 24: Optimizer-version ledger | `python reproduce_paper.py table-24 --run` | CPU | Seconds estimated; metadata only |

| Figure / timing result | Command | Hardware | Runtime / compute budget |
|---|---|---|---|
| 1: Coordinate-update schematic | `python reproduce_paper.py figure-01 --run` | CPU | Seconds estimated; no training |
| 2: M2 trade-off and M3 age panels | `python reproduce_paper.py figure-02 --run` | S1–S3 | Unrecorded; references + mechanisms |
| 3: S1 horizon plasticity and retention gaps | `python reproduce_paper.py figure-03 --run` | L4 | Unrecorded; shared 50M-update benchmark |
| 4: Importance damage | `python reproduce_paper.py figure-04 --run` | A30 | Unrecorded; S1 reference + mechanisms |
| 5: Prospective prediction | `python reproduce_paper.py figure-05 --run` | A30 | Unrecorded; S1 reference + mechanisms |
| 6: Release response | `python reproduce_paper.py figure-06 --run` | A30 | Unrecorded; S1 reference + mechanisms |
| 7: Endpoint-error diagnostic | `python reproduce_paper.py figure-07 --run` | A30 | Unrecorded; 10M reference updates |
| 8: Box versus proximal spring | `python reproduce_paper.py figure-08 --run` | GPU unrecorded | Unrecorded; 475k updates |
| 9: Retention-rule freezing | `python reproduce_paper.py figure-09 --run` | GPU unrecorded | Unrecorded; 20M core updates + E4a, selection, and comparators |
| §3.4 / Appendix A: Optimizer overhead | `python reproduce_paper.py overhead --run` | L4 | Unrecorded; 27 full-workload timing repeats + optimizer-only timing |

## Tests and smoke runs

```bash
python -m pytest -q
python run_experiments.py mechanistic-mnist --smoke --device cpu
python run_experiments.py mechanistic-cifar --smoke --device cpu
```

The two mechanistic packages also have their own test suites; commands are in
the [mechanistic guide](docs/MECHANISTIC.md). Smoke runs use synthetic data and
short horizons to exercise the pipeline. They do not establish reproduction of
the scientific results. This source revision has received static checks;
numerical tests and complete experiment campaigns have not been rerun for it.

## Repository layout

| Path | Contents |
|---|---|
| `cap_adam.py` | Final CAP-Adam optimizer |
| `cap_adam_experimental.py` | Earlier optimizer variants |
| `experiment_*.py`, `phase1_*.py`, `post_phase1_*.py` | Benchmark and development runners |
| `cap_release_final.py` | Final Post-Phase-1 retune, training, and analysis |
| `cap_mechanistic/` | MNIST mechanisms and reference diagnostics |
| `cap_mechanistic_cifar/` | CIFAR-100 mechanisms and learning-rate selection |
| `reproduce_paper.py`, `paper_targets.json` | Table/figure dependencies and artifact collection |
| `tests/` | Optimizer, protocol, and source-distribution checks |
| `docs/` | Method, data, and experiment guides |
| `external/upgd/` | Bundled UPGD dependency and upstream license |

For a source-only archive, `python tools/package_release.py` verifies
`MANIFEST.sha256` and writes `dist/anonymous-source.zip`. Generated results,
checkpoints, datasets, and environments are excluded. See the
[source-distribution guide](docs/SOURCE_DISTRIBUTION.md) for integrity and metadata
checks. Third-party attribution
and licensing information are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
