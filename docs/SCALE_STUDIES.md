# Additional frozen-backbone scale studies

These studies train classifier heads on frozen ResNet-50 features. They are
separate from E1–E6 and the MNIST/CIFAR intervention studies. Prepare the
datasets described in [DATA.md](DATA.md) and run commands from the repository
root. All code, manifests, caches and baseline runs can be generated here.

## DN4IL recurrent natural domains

Frozen `ResNet50_Weights.IMAGENET1K_V2` features feed a trainable 2048→300→100
MLP head. The six domains recur in five cycles: 30 segments, 200 updates/segment,
batch 128. All official test banks are evaluated. This is a recurrent experiment
built from DN4IL splits, not a reproduction of the original paper's schedule.

```bash
python experiment_frozen_resnet_head.py --dataset dn4il --prepare
python experiment_frozen_resnet_head.py --dataset dn4il --build-feature-cache --device cuda
python experiment_frozen_resnet_head.py --dataset dn4il --smoke --all-methods --device cuda --checkpoint-every-updates 1
python experiment_frozen_resnet_head.py --dataset dn4il --qualify --method adamw --device cuda --resume
python experiment_frozen_resnet_head.py --dataset dn4il --calibrate --method cap --device cuda
python experiment_frozen_resnet_head.py --dataset dn4il --tune --method adamw --device cuda --resume
python experiment_frozen_resnet_head.py --dataset dn4il --tune --method cap --device cuda --resume
python experiment_frozen_resnet_head.py --dataset dn4il --tune --method upgd --device cuda --resume
python experiment_frozen_resnet_head.py --dataset dn4il --tune --method ewc --device cuda --resume
python experiment_frozen_resnet_head.py --dataset dn4il --train --all-methods --all-seeds --device cuda --resume
python analyze_frozen_resnet_head.py --dataset dn4il --results-dir results/dn4il_frozen_resnet50_head
```

Qualification and calibration are real gates. Qualification requires forgetting
and within-visit learning gain of at least one percentage point on the development
stream. Calibration is accuracy-blind and checks whether transferred CAP boxes
are nondegenerate. If it fails, inspect its diagnostics before establishing a new
center with `--calibrate --cap-k-center VALUE`. Do not silently change the center.

Development seeds are 31415 and 27182; reporting seeds are 42, 123, 2026, 451, 789.
P averages pre-update accuracy in segments 16–30, R averages pre-visit full-test
revisit accuracy in those segments, and B is `min(P,R)` within seed before
aggregation. Reports include signed final forgetting and recurrent BWT.

## DN4IL separate CAP/corrected-UPGD retune

First complete the original DN4IL study above, including the original caches and
AdamW/EWC reporting trajectories. The separate study freezes and reuses those
baselines read-only within this repository:

```bash
python experiment_dn4il_retune.py --prepare
python experiment_dn4il_retune.py --smoke --all-methods --device cuda --resume
python experiment_dn4il_retune.py --tune --method cap --device cuda --resume
python experiment_dn4il_retune.py --tune --method upgd --device cuda --resume
python experiment_dn4il_retune.py --train --all-methods --all-seeds --device cuda --resume
python experiment_dn4il_retune.py --analyze
```

Each method has 36 candidates evaluated on both development seeds (144 tuning
runs). The combined analysis uses the newly tuned CAP and corrected-UPGD plus
the original AdamW/EWC runs. It is a post-hoc study with unequal search budgets;
it cannot establish superiority under equal tuning budgets. The separate
`dn4il_upgd_corrected.py` keeps the utility numerator and normalization on the
same bias-corrected scale and records its learning-gate diagnostics.

## mini-ImageNet visual domains

```bash
python prepare_miniimagenet_scale.py --data-root data/miniimagenet
python experiment_miniimagenet_resnet.py --smoke --all-methods --data-root data/miniimagenet --device cuda
```

Inspect the smoke pilot's importance/radius/binding diagnostics and choose an
accuracy-blind CAP k center before CAP tuning. The command below uses the
transferred `7e-5` center only if those diagnostics support it:

```bash
python experiment_miniimagenet_resnet.py --tune --method adamw --data-root data/miniimagenet --device cuda --resume
python experiment_miniimagenet_resnet.py --tune --method cap --cap-k-center 7e-5 --data-root data/miniimagenet --device cuda --resume
python experiment_miniimagenet_resnet.py --tune --method upgd --data-root data/miniimagenet --device cuda --resume
python experiment_miniimagenet_resnet.py --tune --method ewc --data-root data/miniimagenet --device cuda --resume
python experiment_miniimagenet_resnet.py --train --all-methods --all-seeds --data-root data/miniimagenet --device cuda --resume
python analyze_miniimagenet_resnet.py --results-dir results/miniimagenet_frozen_resnet50_head
```

The nine deterministic domains include clean images and visual corruptions.
Feature caches can be generated lazily with a local `--feature-cache-dir` passed
consistently. `--help` documents alternate manifest/cache/output paths.
Source hashes and fixed seed roles prevent incompatible checkpoints being reused.
