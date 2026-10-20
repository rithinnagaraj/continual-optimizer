# CAP-Adam M1–M4 on patch-permuted CIFAR-100

This package runs MLP and CNN references and mechanisms M1–M4 on patch-permuted
CIFAR-100. It has no runtime imports from the MNIST suite or other project
experiment modules.

## Protocol

Both architectures run the same final contest-blocked Release CAP and central
M1–M4 interventions. The MLP retains hidden widths 300/300, adapting input/output
to 3072/100. The small CNN uses three padded 3x3 convolutions (32/64/128 channels),
ReLU and 2x2 max pooling after each, followed by 2048→256→100 fully connected
layers. No BatchNorm, dropout, augmentation, pretrained weights or global pooling.
This keeps frozen probes deterministic and avoids erasing patch arrangement by
design. Conv kernels and dense matrices receive coordinate-wise CAP protection;
biases use the same Adam updates without boxes.

Default patch size is **4x4 pixels**, forming an 8x8 grid. Every task permutation
reorders the 64 spatial patches, preserving every pixel's position within its
patch and applying the same mapping to all RGB channels. It is not independent
pixel or channel shuffling. `--patch-size` supports 2/4/8/16 and is recorded in
manifests; use a new output root when changing it.

Dataset: all 50,000 CIFAR-100 training images, 100 fine classes. Normalize RGB
with mean (0.5071,0.4867,0.4408), std (0.2675,0.2565,0.2761). No random transforms.
Evaluation uses the same deterministic 2,000-example test subset for both models
(seed 999). The images are flattened in storage and reshaped to NCHW by the CNN.

The shared MNIST/CIFAR M1–M4 protocol uses:

- Seeds 42/123/451/789/2026, 200 tasks, 5,000 updates/task, batch 128 including
  final short batches; sampler state continues across task boundaries.
- Stream A: new permutation each task, permutation seed 100000 + task.
- Stream B: 40 fixed permutations (seeds 200001 + domain), five shuffled blocks
  of all 40 domains, generated sequentially with schedule seed 555.
- CAP: lr=1/300, betas=(0.9,0.999), eps=1e-8, weight decay=0, k=7e-5,
  Ta=Ts=2000, Tm=1,000,000; raw-g² importance, moving anchors, coordinate boxes,
  final contest-blocked Release. The optimizer equations are unchanged.
- M1: Stream B checkpoints 100/200, each protected tensor, fractions 1%/5%/20%,
  importance/Fisher/magnitude comparisons, 20 random groups, 3 antithetic noise
  pairs at 0.1 tensor standard deviation; Fisher 200 examples/domain. The CNN
  includes its three convolution kernels and two dense matrices.
- M2: Stream A checkpoints 20/50/100/150, five-task continuations; k multipliers
  0.5/0.75/1/1.5/2 plus box-off.
- M3: same Stream A checkpoints, one-task continuations; baseline, box-off,
  recenter-only, importance-reset, full CAP-state reset. Adam moments preserved.
- M4: Stream B checkpoints 100/150, ten-task continuations; original, reversed,
  and eight shuffled importance assignments within each protected tensor.
- M1 averages retained loss across all 40 domains. M4 averages non-current
  domain retention over the continuation. M2/M3 probe recent and older tasks
  using the original deterministic old-task selection.
- Paired branch effects, whole-seed bootstrap (10,000 draws), original M1–M4
  directional claim criteria. Architectures are analyzed separately, not pooled.

Evaluation microbatches are 128 for both CIFAR and MNIST.
The 85% adaptation threshold is retained for protocol fidelity; on CIFAR-100 it
may often be unreached. The capped time-to-threshold is descriptive; M1–M4's
central claims use loss damage, early accuracy, forgetting and binding instead.
The default `lr=1/300` is a fixed-configuration transfer setting, rather than an
architecture-specific selection. For S2/S3 use `--paper-settings`, which selects
MLP `lr=2e-4` and CNN `lr=3e-4` under the `paper_fixed/` output directory.
The optional pilot and selected-rate suite are described in [LR_SWEEP.md](LR_SWEEP.md).

## Reference admission

There are no supplied historical CIFAR endpoints. `reference` builds a new,
separate 45-checkpoint bank per architecture and records descriptive sampled/dense
plasticity and final Stream-B R. **It never compares against MNIST targets.**
Admission checks five-seed completeness, finite accuracy metrics, frozen source,
CAP configuration, model/patch protocol identity, and all checkpoint hashes.
This is reference integrity, not historical reproduction or an accuracy-quality
threshold. Reports explicitly label that distinction. Non-smoke mechanisms
require the full bank; seed subsets can be used for debugging but will not pass
the full reference gate. There is no scientific bypass flag.

Every model gets its own bank, manifests, raw journals, summaries and report:

```text
OUTPUT_DIR/
  mlp/scientific/
    reproduction/  # local CIFAR reference, name retained for schema compatibility
    m1_importance_faithfulness/
    m2_k_causal_dial/
    m3_stream_age_state/
    m4_importance_assignment/
    analysis/mechanistic_report.md
  cnn/scientific/
    ...
```

Synthetic smoke runs use `mlp/smoke` and `cnn/smoke`; they are explicitly not
scientific evidence. They use 32 synthetic training images, 10 evaluation images,
four three-update reference tasks, one-task branches, two random probe groups
and one antithetic pair. Output dimensions still cover 100 classes. The CLI automatically appends model and run type, so pass
the common output root rather than a model-specific path.

## Bash commands

Run these commands from the `cap_mechanistic_cifar` directory:

```bash
# Preserve your working PyTorch/CUDA installation; install this package without upgrading it.
python -m pip install -e . --no-deps
python -c 'import torch, torchvision, numpy, pandas, matplotlib; print(torch.__version__)'
python -m pytest tests -q

# Inspect the exact protocol without training or downloading data.
python -m cifar_mech.cli all --model both --patch-size 4 --dry-run

# Quick implementation exercise: synthetic data, one seed, reduced horizons/probes.
python -u -m cifar_mech.cli all --model both --patch-size 4 \
  --device cuda --smoke --seeds 42 --output-dir ../results/cifar_mechanistic_check --resume

# Full experiment: reference + M1-M4 + analysis, MLP followed by CNN.
mkdir -p ../results/mechanistic_cifar
set -o pipefail
python -u -m cifar_mech.cli all --model both --patch-size 4 \
  --device cuda --data-dir ../data \
  --output-dir ../results/mechanistic_cifar --resume \
  2>&1 | tee -a ../results/mechanistic_cifar/launch.log
```

Use `--model mlp` or `--model cnn` to run/resume one architecture independently.
Commands `reference`, `m1-faithfulness`, `m2-k-dial`, `m3-state-causality`,
`m4-importance-assignment`, and `analyze` can also be run separately with the same
options. Resume checks source/configuration/checkpoint identity and trims any
uncommitted journal tail. Do not modify the package during a scientific run.
Different architectures may use separate processes/devices with the same common
output root, but do not run two writers for the same architecture simultaneously.

This is substantial compute: reference training alone is 10 million updates per
architecture; M1 requires many frozen loss/Fisher evaluations. Full raw logs and
checkpoints also consume significant disk space. Datasets, checkpoints, and
precomputed scientific results are not bundled.

After the optional learning-rate sweep, `bash run_all.sh` runs both
selected-rate scientific pipelines. For the fixed-rate study use the CLI above. Override `MODEL=cnn`, `OUTPUT_DIR=...`, `DATA_DIR=...`, `DEVICE=...`, or
`PATCH_SIZE=...` as environment variables when needed. See
[VALIDATION.md](../docs/VALIDATION.md) for test commands and execution status.
