# Data and external dependencies

All default data paths are inside this repository. Downloads require network
access; after preparation the experiments use local datasets and installed Python
packages. No original project directory or separately installed project module
is needed. Set `TORCH_HOME` to a local `data/torch_cache` when invoking scale
scripts directly if pretrained weight downloads should also stay in this tree.

| Data | Used by | Acquisition |
|---|---|---|
| MNIST | E1–E5, Phase-1, Post-Phase-1, MNIST mechanisms | torchvision downloads on the first real-data invocation into `data/MNIST/`. |
| Fashion-MNIST | E1, E5 | torchvision downloads into `data/FashionMNIST/`. |
| EMNIST Balanced | E3 | torchvision downloads into `data/EMNIST/`; allow time/storage for the source archive. |
| CIFAR-100 | CIFAR mechanisms | Reference command downloads via torchvision into `data/cifar-100-python/`. The optional LR pilot requires this download to exist already. |
| mini-ImageNet | Optional scale study | Supply all 100 classes with 600 RGB 84×84 images per class inside `data/miniimagenet/`. CSV image/label layouts or class folders are supported. |
| DomainNet + official DN4IL splits | Optional scale study | Supply images and all twelve train/test split files in the layout below. No automatic download. |
| ImageNet ResNet-50 V2 weights | Optional scale studies | torchvision can download pretrained weights, or use its existing local cache. The same declared transforms are applied deterministically. |

```text
data/
  DomainNet/
    real/  clipart/  infograph/  painting/  quickdraw/  sketch/
  DN4IL-dataset/
    real_train.txt       real_test.txt
    clipart_train.txt    clipart_test.txt
    infograph_train.txt  infograph_test.txt
    painting_train.txt   painting_test.txt
    quickdraw_train.txt  quickdraw_test.txt
    sketch_train.txt     sketch_test.txt
```

Obtain data from the original dataset providers under their terms. The repository
does not redistribute dataset images or trained weights. DN4IL preparation checks
class mappings, the complete label space, path traversal, missing files, duplicate
images and train/test overlap before feature extraction. mini-ImageNet preparation
builds a deterministic 500/100-per-class split; it does not adopt few-shot train/
validation/test class partitions as separate continual tasks.

Synthetic mechanistic smoke runs need no external data. E6's GPT-2 and ResNet-18
are created from configuration/random initialization and do not need weights.

PyTorch/torchvision must be a compatible pair. `requirements.txt` expresses minimum
APIs, not a claim of bitwise reproduction across versions. Use one frozen
environment and device for a scientific run; runners record runtime versions,
source hashes, seeds and protocol settings. Preserve the environment's package
freeze with the resulting experiment for later reproduction. GPU timing is
hardware dependent; the original historical gate remains the numerical check.
