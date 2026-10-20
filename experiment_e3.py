"""E3: test whether centered importance stays clean across distribution shifts.

The passive part of this experiment tracks CAP-Adam's centered estimator,
``(g - m_hat)^2``, and a raw ``g^2`` estimator on the same optimization run.
Consequently every paired comparison sees the same initialization, minibatches,
gradients, and AdamW updates.  Four streams are measured:

* coherent: MNIST switches to one fixed input permutation;
* incoherent: EMNIST-Balanced switches to one fixed label permutation;
* gradual: MNIST starts rotating by one additional degree every 50 steps.
* coherent_hard: EMNIST-Balanced switches from digits to unseen letters.

The functional comparison is a separate, active-protection rerun of the hard
coherent shift.  It compares AdamW, centered CAP-Adam, and a raw-importance
leash.  The raw arm copies its external raw tracker into CAP-Adam's importance
state after each update, so the next update's box is based only on raw
importance.

Default settings implement the preregistered E3 experiment (five seeds, 5,000
pre-shift steps, and 4,000 measured post-shift steps).  Reduced settings are
useful for smoke tests but are not valid E3 evidence.  Example:

    python experiment_e3.py --data-dir data --output-dir results/e3

Outputs include per-seed and summarized contamination CSVs, one contamination
plot per shift type, a burn-in alignment plot, functional learning-speed CSVs,
the fixed permutations, a machine-readable config, and a preregistered verdict
report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from model import MLP300
from cap_adam_experimental import RawImportanceTracker, CAPAdam


SEEDS = (42, 123, 2026, 451, 789)
SHIFT_TYPES = ("coherent", "incoherent", "gradual", "coherent_hard")
ESTIMATORS = ("centered", "raw")
SUBSETS = ("all", "top5_new_gradient")

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
T_S = 2_000.0
T_A = 2_000.0
T_M = 10_000.0
TOP_GRADIENT_FRACTION = 0.05

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
EMNIST_MEAN = 0.1736
EMNIST_STD = 0.3317


class PermutedDataset(Dataset):
    """Apply a fixed flattened-pixel permutation to an image dataset."""

    def __init__(self, dataset: Dataset, permutation: Tensor) -> None:
        if permutation.dtype != torch.long or permutation.ndim != 1:
            raise TypeError("permutation must be a one-dimensional long tensor")
        if permutation.numel() != 28 * 28:
            raise ValueError("input permutation must contain 784 indices")
        expected = torch.arange(28 * 28)
        if not torch.equal(torch.sort(permutation.cpu()).values, expected):
            raise ValueError("input permutation must contain every pixel exactly once")
        self.dataset = dataset
        self.permutation = permutation.cpu().clone()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, target = self.dataset[index]
        return image.reshape(-1)[self.permutation].reshape_as(image), int(target)


class LabelPermutedDataset(Dataset):
    """Map every class label through one fixed class permutation."""

    def __init__(
        self, dataset: Dataset, permutation: Tensor, class_count: int
    ) -> None:
        if permutation.dtype != torch.long or permutation.ndim != 1:
            raise TypeError("permutation must be a one-dimensional long tensor")
        if permutation.numel() != class_count:
            raise ValueError("label permutation has the wrong number of classes")
        expected = torch.arange(class_count)
        if not torch.equal(torch.sort(permutation.cpu()).values, expected):
            raise ValueError("label permutation must contain every class exactly once")
        self.dataset = dataset
        self.permutation = permutation.cpu().clone()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, target = self.dataset[index]
        return image, int(self.permutation[int(target)].item())


class ClassSubset(Dataset):
    """Select dataset examples by stored label without applying transforms."""

    def __init__(self, dataset: Dataset, allowed_labels: Sequence[int]) -> None:
        if not hasattr(dataset, "targets"):
            raise TypeError("ClassSubset requires a dataset with a targets tensor")
        allowed = list(allowed_labels)
        if not allowed:
            raise ValueError("allowed_labels cannot be empty")
        targets = torch.as_tensor(dataset.targets)
        allowed_tensor = torch.as_tensor(allowed, dtype=targets.dtype)
        self.dataset = dataset
        self.indices = torch.nonzero(
            torch.isin(targets, allowed_tensor), as_tuple=False
        ).flatten().tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, target = self.dataset[self.indices[index]]
        return image, int(target)


def upright_emnist(image):
    """Undo the transposed orientation used by the raw EMNIST files."""
    image = transform_functional.rotate(image, -90)
    return transform_functional.hflip(image)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and choose deterministic CUDA paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fixed_permutation(size: int, seed: int) -> Tensor:
    """Return a deterministic permutation without changing the global RNG."""
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed))


def load_mnist(root: Path) -> Tuple[Dataset, Dataset]:
    """Load normalized MNIST train and test datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train, test


def load_emnist(root: Path) -> Tuple[Dataset, Dataset]:
    """Load upright, normalized EMNIST-Balanced train and test datasets."""
    transform = transforms.Compose(
        [
            transforms.Lambda(upright_emnist),
            transforms.ToTensor(),
            transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
        ]
    )
    train = datasets.EMNIST(
        root=root,
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.EMNIST(
        root=root,
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )
    return train, test


def cap_parameter_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Name every tensor and exempt biases from protection and importance use."""
    groups: List[dict] = []
    for name, parameter in model.named_parameters():
        is_weight = parameter.ndim >= 2
        groups.append(
            {
                "params": [parameter],
                "param_name": name,
                "protect": is_weight,
                "weight_decay": weight_decay if is_weight else 0.0,
            }
        )
    return groups


def adamw_parameter_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Match CAP's matrix-only weight-decay convention in plain AdamW."""
    return [
        {
            "params": [parameter],
            "weight_decay": weight_decay if parameter.ndim >= 2 else 0.0,
        }
        for parameter in model.parameters()
    ]


def weight_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return matrix-valued weights; biases are excluded from E3 statistics."""
    return [parameter for parameter in model.parameters() if parameter.ndim >= 2]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    *,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """Construct a reproducibly shuffled data loader."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def repeat_batches(loader: DataLoader) -> Iterator[Tuple[Tensor, Tensor]]:
    """Yield batches forever, retaining the loader generator across epochs."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Return full-loader mean cross-entropy and accuracy."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += targets.numel()
    if count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return loss_sum / count, correct / count


def rotate_normalized_mnist(inputs: Tensor, angle: float) -> Tensor:
    """Rotate normalized MNIST tensors while filling with normalized black."""
    normalized_black = (0.0 - MNIST_MEAN) / MNIST_STD
    return transform_functional.rotate(
        inputs,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=[normalized_black],
    )


def post_shift_inputs(inputs: Tensor, shift_type: str, step: int) -> Tuple[Tensor, float]:
    """Apply the step-dependent part of a stream and return its current angle."""
    if shift_type != "gradual":
        return inputs, 0.0
    angle = float(1 + (step - 1) // 50)
    return rotate_normalized_mnist(inputs, angle), angle


def centered_importance_vector(
    optimizer: CAPAdam, parameters: Sequence[nn.Parameter]
) -> Tensor:
    """Flatten CAP's centered importance in a stable weight-only order."""
    values = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state or "imp" not in state:
            raise RuntimeError("centered importance state is not initialized")
        values.append(state["imp"].detach().reshape(-1))
    return torch.cat(values)


def raw_importance_vector(
    tracker: RawImportanceTracker, parameters: Sequence[nn.Parameter]
) -> Tensor:
    """Flatten raw importance in the same weight-only parameter order."""
    values = []
    for parameter in parameters:
        if parameter not in tracker.state:
            raise RuntimeError("raw importance state is not initialized")
        values.append(tracker.state[parameter].detach().reshape(-1))
    return torch.cat(values)


def gradient_top_indices(
    parameters: Sequence[nn.Parameter], fraction: float
) -> Tensor:
    """Select weights with the largest absolute first-new-task gradients."""
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("a weight has no gradient at the task switch")
        gradients.append(parameter.grad.detach().abs().reshape(-1))
    flat = torch.cat(gradients)
    count = max(1, math.ceil(flat.numel() * fraction))
    return torch.topk(flat, count, largest=True, sorted=False).indices


def relative_floor(baseline: Tensor, ratio_floor: float) -> float:
    """Choose a scale-relative denominator floor for zero-importance weights."""
    positive = baseline[baseline > 0]
    scale = float(positive.median().item()) if positive.numel() else 1.0
    return max(scale * ratio_floor, torch.finfo(baseline.dtype).tiny)


def contamination_value(
    current: Tensor,
    baseline: Tensor,
    indices: Optional[Tensor],
    ratio_floor: float,
) -> Tuple[float, float, float, float]:
    """Return median per-weight ratio plus raw medians and denominator floor.

    A weight that is numerically zero both before and after the shift has ratio
    one.  If it acquires evidence from a zero baseline, its baseline is replaced
    by a tiny, scale-relative floor so contamination remains finite and visible.
    """
    if indices is not None:
        current = current[indices]
        baseline = baseline[indices]
    floor = relative_floor(baseline, ratio_floor)
    denominator = baseline.clamp_min(floor)
    ratio = current / denominator
    unchanged_zero = (baseline <= floor) & (current <= floor)
    ratio = torch.where(unchanged_zero, torch.ones_like(ratio), ratio)
    return (
        float(ratio.median().item()),
        float(current.median().item()),
        float(baseline.median().item()),
        floor,
    )


def append_contamination_rows(
    rows: List[dict],
    *,
    shift_type: str,
    seed: int,
    step: int,
    angle: float,
    centered: Tensor,
    raw: Tensor,
    centered_baseline: Tensor,
    raw_baseline: Tensor,
    top_indices: Tensor,
    ratio_floor: float,
    b_test_accuracy: float,
) -> None:
    """Append the four paired estimator/subset measurements at one time."""
    for estimator, current, baseline in (
        ("centered", centered, centered_baseline),
        ("raw", raw, raw_baseline),
    ):
        for subset, indices in (
            ("all", None),
            ("top5_new_gradient", top_indices),
        ):
            value, current_median, baseline_median, floor = contamination_value(
                current, baseline, indices, ratio_floor
            )
            rows.append(
                {
                    "shift_type": shift_type,
                    "seed": seed,
                    "post_shift_step": step,
                    "angle_degrees": angle,
                    "estimator": estimator,
                    "subset": subset,
                    "contamination": value,
                    "median_importance": current_median,
                    "baseline_median_importance": baseline_median,
                    "denominator_floor": floor,
                    "top_gradient_fraction": TOP_GRADIENT_FRACTION,
                    "b_test_accuracy": b_test_accuracy,
                }
            )


def stream_datasets(
    shift_type: str,
    mnist_train: Optional[Dataset],
    mnist_test: Optional[Dataset],
    emnist_train: Optional[Dataset],
    emnist_test: Optional[Dataset],
    input_permutation: Tensor,
    label_permutation: Tensor,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """Resolve Task-A train, Task-B train/test, and output width."""
    if shift_type == "coherent":
        if mnist_train is None or mnist_test is None:
            raise RuntimeError("MNIST was not loaded")
        return (
            mnist_train,
            PermutedDataset(mnist_train, input_permutation),
            PermutedDataset(mnist_test, input_permutation),
            10,
        )
    if shift_type == "incoherent":
        if emnist_train is None or emnist_test is None:
            raise RuntimeError("EMNIST was not loaded")
        return (
            emnist_train,
            LabelPermutedDataset(emnist_train, label_permutation, 47),
            LabelPermutedDataset(emnist_test, label_permutation, 47),
            47,
        )
    if shift_type == "gradual":
        if mnist_train is None or mnist_test is None:
            raise RuntimeError("MNIST was not loaded")
        return mnist_train, mnist_train, mnist_test, 10
    if shift_type == "coherent_hard":
        if emnist_train is None or emnist_test is None:
            raise RuntimeError("EMNIST was not loaded")
        return (
            ClassSubset(emnist_train, range(0, 10)),
            ClassSubset(emnist_train, range(10, 47)),
            ClassSubset(emnist_test, range(10, 47)),
            47,
        )
    raise ValueError(f"unknown shift type: {shift_type}")


def run_passive_stream(
    shift_type: str,
    task_a: Dataset,
    task_b: Dataset,
    task_b_test: Dataset,
    output_units: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    collect_burn_in: bool,
) -> Tuple[List[dict], List[dict]]:
    """Run one perfectly paired passive stream and return its measurements."""
    seed_everything(seed)
    model = MLP300(input_units=28 * 28, output_units=output_units).to(device)
    optimizer = CAPAdam(
        cap_parameter_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        T_s=args.T_s,
        T_a=args.T_a,
        T_m=args.T_m,
        k=args.k,
        passive=True,
        retention=args.measurement_retention,
        importance_evidence="centered",
    )
    raw_tracker = RawImportanceTracker(optimizer)
    weights = weight_parameters(model)
    a_batches = repeat_batches(
        make_loader(
            task_a,
            args.train_batch_size,
            device,
            args.num_workers,
            shuffle=True,
            seed=seed + 10_000,
        )
    )

    burn_rows: List[dict] = []
    model.train()
    for step in range(1, args.pre_shift_steps + 1):
        inputs, targets = next(a_batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        raw_tracker.update()

        if collect_burn_in and (
            step == 1
            or step == args.pre_shift_steps
            or step % args.burn_in_log_every == 0
        ):
            centered = centered_importance_vector(optimizer, weights)
            raw = raw_importance_vector(raw_tracker, weights)
            burn_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "batch_loss": float(loss.item()),
                    "batch_accuracy": float(
                        (logits.argmax(dim=1) == targets).float().mean().item()
                    ),
                    "centered_median": float(centered.median().item()),
                    "centered_mean": float(centered.mean().item()),
                    "raw_median": float(raw.median().item()),
                    "raw_mean": float(raw.mean().item()),
                }
            )

    centered_baseline = centered_importance_vector(optimizer, weights).clone()
    raw_baseline = raw_importance_vector(raw_tracker, weights).clone()
    b_batches = repeat_batches(
        make_loader(
            task_b,
            args.train_batch_size,
            device,
            args.num_workers,
            shuffle=True,
            seed=seed + 20_000,
        )
    )
    b_test_loader = make_loader(
        task_b_test,
        args.evaluation_batch_size,
        device,
        args.num_workers,
        shuffle=False,
    )
    _, initial_b_accuracy = evaluate(model, b_test_loader, device)
    model.train()

    contamination_rows: List[dict] = []
    top_indices: Optional[Tensor] = None
    for post_step in range(1, args.post_shift_steps + 1):
        inputs, targets = next(b_batches)
        inputs, angle = post_shift_inputs(inputs, shift_type, post_step)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        if post_step == 1:
            top_indices = gradient_top_indices(weights, TOP_GRADIENT_FRACTION)
            append_contamination_rows(
                contamination_rows,
                shift_type=shift_type,
                seed=seed,
                step=0,
                angle=0.0,
                centered=centered_baseline,
                raw=raw_baseline,
                centered_baseline=centered_baseline,
                raw_baseline=raw_baseline,
                top_indices=top_indices,
                ratio_floor=args.ratio_floor,
                b_test_accuracy=initial_b_accuracy,
            )

        optimizer.step()
        raw_tracker.update()
        should_log = (
            post_step <= 50
            or post_step % args.contamination_log_every == 0
            or post_step == args.post_shift_steps
        )
        if should_log:
            if top_indices is None:
                raise AssertionError("top-gradient subset was not initialized")
            _, b_test_accuracy = evaluate(model, b_test_loader, device)
            model.train()
            append_contamination_rows(
                contamination_rows,
                shift_type=shift_type,
                seed=seed,
                step=post_step,
                angle=angle,
                centered=centered_importance_vector(optimizer, weights),
                raw=raw_importance_vector(raw_tracker, weights),
                centered_baseline=centered_baseline,
                raw_baseline=raw_baseline,
                top_indices=top_indices,
                ratio_floor=args.ratio_floor,
                b_test_accuracy=b_test_accuracy,
            )

    return contamination_rows, burn_rows


@torch.no_grad()
def sync_raw_importance(
    optimizer: CAPAdam, tracker: RawImportanceTracker, model: nn.Module
) -> None:
    """Make the raw arm's next box use its external raw importance exactly."""
    for parameter in model.parameters():
        if parameter not in tracker.state:
            continue
        optimizer.state[parameter]["imp"].copy_(tracker.state[parameter])


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    raw_tracker: Optional[RawImportanceTracker],
    model: nn.Module,
) -> None:
    """Take one arm update and synchronize raw importance when requested."""
    optimizer.step()
    if raw_tracker is not None:
        raw_tracker.update()
        if not isinstance(optimizer, CAPAdam):
            raise TypeError("a raw tracker can only accompany CAPAdam")
        sync_raw_importance(optimizer, raw_tracker, model)


def train_functional_arm(
    arm: str,
    initial_state: Mapping[str, Tensor],
    output_units: int,
    task_a_train: Dataset,
    task_a_test: Dataset,
    task_b_train: Dataset,
    task_b_test: Dataset,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[dict, List[dict]]:
    """Run one full hard-coherent-shift arm from the paired initialization."""
    if arm not in ("adamw", "centered_protected", "raw_protected"):
        raise ValueError(f"unknown functional arm: {arm}")
    seed_everything(seed)
    model = MLP300(input_units=28 * 28, output_units=output_units).to(device)
    model.load_state_dict(initial_state)
    raw_tracker: Optional[RawImportanceTracker] = None
    if arm == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            adamw_parameter_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
        )
    else:
        optimizer = CAPAdam(
            cap_parameter_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            T_s=args.T_s,
            T_a=args.T_a,
            T_m=args.T_m,
            k=args.k,
            passive=False,
            protection="box",
            retention=args.functional_retention,
            importance_evidence="centered",
        )
        if arm == "raw_protected":
            raw_tracker = RawImportanceTracker(optimizer)

    a_batches = repeat_batches(
        make_loader(
            task_a_train,
            args.train_batch_size,
            device,
            args.num_workers,
            shuffle=True,
            seed=seed + 30_000,
        )
    )
    model.train()
    for _ in range(args.pre_shift_steps):
        inputs, targets = next(a_batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets).backward()
        optimizer_step(optimizer, raw_tracker, model)

    task_a_test_loader = make_loader(
        task_a_test,
        args.evaluation_batch_size,
        device,
        args.num_workers,
        shuffle=False,
    )
    task_b_test_loader = make_loader(
        task_b_test,
        args.evaluation_batch_size,
        device,
        args.num_workers,
        shuffle=False,
    )
    _, task_a_accuracy = evaluate(model, task_a_test_loader, device)
    initial_b_loss, initial_b_accuracy = evaluate(model, task_b_test_loader, device)
    trajectory = [
        {
            "arm": arm,
            "seed": seed,
            "b_step": 0,
            "b_test_loss": initial_b_loss,
            "b_test_accuracy": initial_b_accuracy,
            "target_accuracy": args.target_accuracy,
        }
    ]
    reached_step: Optional[int] = 0 if initial_b_accuracy >= args.target_accuracy else None
    final_b_loss = initial_b_loss
    final_b_accuracy = initial_b_accuracy
    early_bind_count = 0
    early_bind_numel = 0

    if reached_step is None:
        b_batches = repeat_batches(
            make_loader(
                task_b_train,
                args.train_batch_size,
                device,
                args.num_workers,
                shuffle=True,
                seed=seed + 40_000,
            )
        )
        model.train()
        for b_step in range(1, args.functional_max_b_steps + 1):
            inputs, targets = next(b_batches)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(inputs), targets).backward()
            optimizer_step(optimizer, raw_tracker, model)

            if arm != "adamw" and b_step <= 500:
                for group in optimizer.param_groups:
                    if group.get("protect", True):
                        early_bind_count += int(group["last_bind_count"])
                        early_bind_numel += int(group["last_bind_numel"])

            if (
                b_step % args.functional_eval_every == 0
                or b_step == args.functional_max_b_steps
            ):
                final_b_loss, final_b_accuracy = evaluate(
                    model, task_b_test_loader, device
                )
                trajectory.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "b_step": b_step,
                        "b_test_loss": final_b_loss,
                        "b_test_accuracy": final_b_accuracy,
                        "target_accuracy": args.target_accuracy,
                    }
                )
                if final_b_accuracy >= args.target_accuracy:
                    reached_step = b_step
                    break
                model.train()

    _, task_a_accuracy_after_b = evaluate(model, task_a_test_loader, device)
    reached = reached_step is not None
    result = {
        "arm": arm,
        "seed": seed,
        "target_accuracy": args.target_accuracy,
        "reached_target": reached,
        "steps_to_target": reached_step if reached else "",
        "capped_steps_to_target": (
            reached_step if reached else args.functional_max_b_steps
        ),
        "task_a_accuracy_at_shift": task_a_accuracy,
        "task_a_accuracy_after_b": task_a_accuracy_after_b,
        "initial_b_accuracy": initial_b_accuracy,
        "final_b_accuracy": final_b_accuracy,
        "final_b_loss": final_b_loss,
        "max_b_steps": args.functional_max_b_steps,
        "early_bind_rate": (
            ""
            if arm == "adamw"
            else early_bind_count / max(early_bind_numel, 1)
        ),
    }
    return result, trajectory


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    """Write homogeneous dictionaries to CSV, creating parent directories."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    """Read a CSV emitted by this experiment, or return empty if absent."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return arithmetic mean and sample standard deviation."""
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def summarize_contamination(
    rows: Sequence[Mapping], post_shift_steps: Optional[int] = None
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Summarize curves plus per-seed transient and steady-state statistics.

    Transient boundaries are determined independently within each seed before
    peaks are aggregated.  This preserves the preregistered order of
    operations: per-seed maximum first, median across seeds second.
    """
    grouped: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    angles: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    accuracies: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    accuracy_by_stream: Dict[Tuple[str, int], Dict[int, float]] = defaultdict(dict)
    for row in rows:
        if "b_test_accuracy" not in row:
            raise ValueError(
                "contamination rows lack b_test_accuracy; rerun E3 with the "
                "accuracy-aware transient measurement"
            )
        shift_type = str(row["shift_type"])
        seed = int(float(row["seed"]))
        step = int(float(row["post_shift_step"]))
        accuracy = float(row["b_test_accuracy"])
        key = (
            shift_type,
            step,
            str(row["estimator"]),
            str(row["subset"]),
        )
        grouped[key].append(float(row["contamination"]))
        angles[key].append(float(row["angle_degrees"]))
        accuracies[key].append(accuracy)
        accuracy_by_stream[(shift_type, seed)][step] = accuracy

    curve_summary = []
    for key in sorted(grouped):
        shift_type, step, estimator, subset = key
        values = grouped[key]
        value_mean, value_std = mean_std(values)
        accuracy_mean, accuracy_std = mean_std(accuracies[key])
        curve_summary.append(
            {
                "shift_type": shift_type,
                "post_shift_step": step,
                "angle_degrees": mean(angles[key]),
                "estimator": estimator,
                "subset": subset,
                "contamination_mean": value_mean,
                "contamination_std": value_std,
                "contamination_median": median(values),
                "contamination_min": min(values),
                "contamination_max": max(values),
                "b_test_accuracy_mean": accuracy_mean,
                "b_test_accuracy_std": accuracy_std,
                "b_test_accuracy_median": median(accuracies[key]),
                "seeds": len(values),
            }
        )

    stream_windows: Dict[Tuple[str, int], dict] = {}
    for (shift_type, seed), by_step in sorted(accuracy_by_stream.items()):
        post_steps = sorted(step for step in by_step if step > 0)
        if not post_steps:
            raise ValueError(f"no post-shift measurements for {shift_type}, seed={seed}")
        final_step = post_steps[-1]
        final_accuracy = by_step[final_step]
        if shift_type == "gradual":
            transient_end = (
                final_step if post_shift_steps is None else post_shift_steps
            )
            if transient_end not in by_step:
                raise ValueError(
                    "gradual shift is missing its configured final post-shift "
                    f"measurement at step {transient_end}"
                )
        else:
            threshold = 0.9 * final_accuracy
            transient_end = next(
                step for step in post_steps if by_step[step] >= threshold
            )
        stream_windows[(shift_type, seed)] = {
            "final_step": final_step,
            "final_accuracy": final_accuracy,
            "transient_end": transient_end,
        }

    rows_by_seed: Dict[Tuple[str, int, str, str], List[Mapping]] = defaultdict(list)
    for row in rows:
        step = int(float(row["post_shift_step"]))
        if step <= 0:
            continue
        rows_by_seed[
            (
                str(row["shift_type"]),
                int(float(row["seed"])),
                str(row["estimator"]),
                str(row["subset"]),
            )
        ].append(row)

    window_by_seed: List[dict] = []
    for key in sorted(rows_by_seed):
        shift_type, seed, estimator, subset = key
        seed_rows = sorted(
            rows_by_seed[key], key=lambda row: int(float(row["post_shift_step"]))
        )
        window = stream_windows[(shift_type, seed)]
        transient_rows = [
            row
            for row in seed_rows
            if int(float(row["post_shift_step"])) <= window["transient_end"]
        ]
        if not transient_rows:
            raise ValueError(f"empty transient window for {shift_type}, seed={seed}")
        window_by_seed.append(
            {
                "shift_type": shift_type,
                "seed": seed,
                "estimator": estimator,
                "subset": subset,
                "transient_end": window["transient_end"],
                "final_step": window["final_step"],
                "final_accuracy": window["final_accuracy"],
                "transient_peak": max(
                    float(row["contamination"]) for row in transient_rows
                ),
                "steady_state": float(seed_rows[-1]["contamination"]),
            }
        )

    window_groups: Dict[Tuple[str, str, str], List[Mapping]] = defaultdict(list)
    for row in window_by_seed:
        window_groups[
            (str(row["shift_type"]), str(row["estimator"]), str(row["subset"]))
        ].append(row)

    window_summary: List[dict] = []
    for key in sorted(window_groups):
        shift_type, estimator, subset = key
        seed_rows = window_groups[key]
        transient_peaks = [float(row["transient_peak"]) for row in seed_rows]
        steady_states = [float(row["steady_state"]) for row in seed_rows]
        transient_ends = [float(row["transient_end"]) for row in seed_rows]
        final_accuracies = [float(row["final_accuracy"]) for row in seed_rows]
        window_summary.append(
            {
                "shift_type": shift_type,
                "estimator": estimator,
                "subset": subset,
                "transient_peak_median": median(transient_peaks),
                "steady_state_median": median(steady_states),
                "transient_end_median": median(transient_ends),
                "final_accuracy_median": median(final_accuracies),
                "seeds": len(seed_rows),
            }
        )
    return curve_summary, window_by_seed, window_summary


def normalized_burn_in_rows(rows: Sequence[Mapping]) -> List[dict]:
    """Normalize loss descent and importance growth independently per seed."""
    by_seed: Dict[int, List[Mapping]] = defaultdict(list)
    for row in rows:
        by_seed[int(float(row["seed"]))].append(row)

    normalized: List[dict] = []
    for seed, seed_rows in by_seed.items():
        seed_rows.sort(key=lambda row: int(float(row["step"])))
        tail_count = max(3, math.ceil(0.1 * len(seed_rows)))
        initial_loss = float(seed_rows[0]["batch_loss"])
        plateau_loss = mean(float(row["batch_loss"]) for row in seed_rows[-tail_count:])
        loss_drop = max(initial_loss - plateau_loss, 1e-12)
        centered_final = max(float(seed_rows[-1]["centered_median"]), 1e-30)
        raw_final = max(float(seed_rows[-1]["raw_median"]), 1e-30)
        for row in seed_rows:
            normalized.append(
                {
                    "seed": seed,
                    "step": int(float(row["step"])),
                    "batch_loss": float(row["batch_loss"]),
                    "loss_descent_progress": float(
                        np.clip(
                            (initial_loss - float(row["batch_loss"])) / loss_drop,
                            0.0,
                            1.0,
                        )
                    ),
                    "centered_progress": float(row["centered_median"])
                    / centered_final,
                    "raw_progress": float(row["raw_median"]) / raw_final,
                }
            )
    return normalized


def summarize_burn_in(normalized: Sequence[Mapping]) -> List[dict]:
    """Aggregate normalized burn-in curves over seeds."""
    grouped: Dict[int, List[Mapping]] = defaultdict(list)
    for row in normalized:
        grouped[int(float(row["step"]))].append(row)
    summary = []
    for step in sorted(grouped):
        step_rows = grouped[step]
        output = {"step": step, "seeds": len(step_rows)}
        for field in (
            "batch_loss",
            "loss_descent_progress",
            "centered_progress",
            "raw_progress",
        ):
            values = [float(row[field]) for row in step_rows]
            output[f"{field}_mean"], output[f"{field}_std"] = mean_std(values)
        summary.append(output)
    return summary


def threshold_step(rows: Sequence[Mapping], field: str, threshold: float) -> Optional[int]:
    """Return the first logged step whose field reaches a threshold."""
    for row in sorted(rows, key=lambda item: int(float(item["step"]))):
        if float(row[field]) >= threshold:
            return int(float(row["step"]))
    return None


def assess_burn_in(normalized: Sequence[Mapping]) -> dict:
    """Quantify whether centered importance visibly follows fast loss descent."""
    by_seed: Dict[int, List[Mapping]] = defaultdict(list)
    for row in normalized:
        by_seed[int(float(row["seed"]))].append(row)
    diagnostics = []
    for seed, seed_rows in sorted(by_seed.items()):
        loss90 = threshold_step(seed_rows, "loss_descent_progress", 0.90)
        centered10 = threshold_step(seed_rows, "centered_progress", 0.10)
        centered50 = threshold_step(seed_rows, "centered_progress", 0.50)
        diagnostics.append(
            {
                "seed": seed,
                "loss_90pct_step": loss90,
                "centered_10pct_step": centered10,
                "centered_50pct_step": centered50,
                "lag_10pct_vs_loss90": (
                    None
                    if loss90 is None or centered10 is None
                    else centered10 - loss90
                ),
            }
        )
    valid_lags = [
        row["lag_10pct_vs_loss90"]
        for row in diagnostics
        if row["lag_10pct_vs_loss90"] is not None
    ]
    median_lag = median(valid_lags) if valid_lags else None
    return {
        "available": bool(valid_lags),
        "median_lag_steps": median_lag,
        "visibly_lags": median_lag is not None and median_lag > 0,
        "by_seed": diagnostics,
    }


def summarize_functional(rows: Sequence[Mapping]) -> List[dict]:
    """Build the B learning-speed table, counting stalls at the run horizon."""
    grouped: Dict[str, List[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    summary = []
    for arm in ("adamw", "centered_protected", "raw_protected"):
        arm_rows = grouped.get(arm, [])
        if not arm_rows:
            continue
        reached_rows = [
            row for row in arm_rows if str(row["reached_target"]).lower() == "true"
        ]
        capped = [float(row["capped_steps_to_target"]) for row in arm_rows]
        capped_mean, capped_std = mean_std(capped)
        reached_steps = [float(row["steps_to_target"]) for row in reached_rows]
        reached_mean, reached_std = (
            mean_std(reached_steps) if reached_steps else (math.nan, math.nan)
        )
        a_accuracies = [float(row["task_a_accuracy_at_shift"]) for row in arm_rows]
        a_after_accuracies = [
            float(row["task_a_accuracy_after_b"]) for row in arm_rows
        ]
        b_accuracies = [float(row["final_b_accuracy"]) for row in arm_rows]
        bind_rates = [
            float(row["early_bind_rate"])
            for row in arm_rows
            if str(row["early_bind_rate"]).strip() != ""
        ]
        a_mean, a_std = mean_std(a_accuracies)
        a_after_mean, a_after_std = mean_std(a_after_accuracies)
        b_mean, b_std = mean_std(b_accuracies)
        bind_mean, bind_std = (
            mean_std(bind_rates) if bind_rates else (math.nan, math.nan)
        )
        summary.append(
            {
                "arm": arm,
                "seeds": len(arm_rows),
                "target_accuracy": float(arm_rows[0]["target_accuracy"]),
                "seeds_reached": len(reached_rows),
                "reach_rate": len(reached_rows) / len(arm_rows),
                "steps_to_target_mean_reached": reached_mean,
                "steps_to_target_std_reached": reached_std,
                "capped_steps_mean": capped_mean,
                "capped_steps_std": capped_std,
                "task_a_accuracy_at_shift_mean": a_mean,
                "task_a_accuracy_at_shift_std": a_std,
                "task_a_accuracy_after_b_mean": a_after_mean,
                "task_a_accuracy_after_b_std": a_after_std,
                "final_b_accuracy_mean": b_mean,
                "final_b_accuracy_std": b_std,
                "early_bind_rate_mean": bind_mean,
                "early_bind_rate_std": bind_std,
            }
        )
    by_arm = {row["arm"]: row for row in summary}
    reference = by_arm.get("adamw")
    if reference and float(reference["capped_steps_mean"]) > 0:
        for row in summary:
            row["slowdown_vs_adamw"] = (
                float(row["capped_steps_mean"])
                / float(reference["capped_steps_mean"])
            )
    else:
        for row in summary:
            row["slowdown_vs_adamw"] = math.nan
    return summary


def plot_contamination(
    summary: Sequence[Mapping],
    window_summary: Sequence[Mapping],
    output_dir: Path,
) -> None:
    """Save one paired contamination curve per measured shift type."""
    import matplotlib.pyplot as plt

    colors = {"centered": "#0072B2", "raw": "#D55E00"}
    linestyles = {"all": "-", "top5_new_gradient": "--"}
    for shift_type in SHIFT_TYPES:
        shift_rows = [row for row in summary if row["shift_type"] == shift_type]
        if not shift_rows:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 5.0))
        for estimator in ESTIMATORS:
            for subset in SUBSETS:
                curve = sorted(
                    (
                        row
                        for row in shift_rows
                        if row["estimator"] == estimator and row["subset"] == subset
                    ),
                    key=lambda row: int(row["post_shift_step"]),
                )
                x = np.asarray([float(row["post_shift_step"]) for row in curve])
                y = np.asarray([float(row["contamination_mean"]) for row in curve])
                error = np.asarray([float(row["contamination_std"]) for row in curve])
                label = f"{estimator}, {'all weights' if subset == 'all' else 'top 5% |g_B|'}"
                axis.plot(
                    x,
                    y,
                    color=colors[estimator],
                    linestyle=linestyles[subset],
                    linewidth=2.0,
                    label=label,
                )
                axis.fill_between(
                    x,
                    np.maximum(y - error, 1e-3),
                    y + error,
                    color=colors[estimator],
                    alpha=0.10,
                )
        axis.axhline(2.0, color="#0072B2", linewidth=0.9, alpha=0.65)
        axis.axhline(10.0, color="#D55E00", linewidth=0.9, alpha=0.65)
        transient_ends = [
            float(row["transient_end_median"])
            for row in window_summary
            if row["shift_type"] == shift_type
        ]
        if transient_ends:
            axis.axvline(
                transient_ends[0],
                color="black",
                linewidth=1.1,
                linestyle=":",
                label="median transient end",
            )
        axis.set_yscale("log")
        axis.set_xlabel("Steps after drift onset")
        axis.set_ylabel("Median per-weight contamination C(t)")
        axis.set_title(f"E3 contamination — {shift_type} shift")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"contamination_{shift_type}.png", dpi=180)
        plt.close(figure)


def plot_burn_in(summary: Sequence[Mapping], output_dir: Path) -> None:
    """Plot fresh-initialization loss against normalized importance growth."""
    if not summary:
        return
    import matplotlib.pyplot as plt

    x = np.asarray([float(row["step"]) for row in summary])
    loss = np.asarray([float(row["batch_loss_mean"]) for row in summary])
    loss_error = np.asarray([float(row["batch_loss_std"]) for row in summary])
    centered = np.asarray([float(row["centered_progress_mean"]) for row in summary])
    centered_error = np.asarray(
        [float(row["centered_progress_std"]) for row in summary]
    )
    raw = np.asarray([float(row["raw_progress_mean"]) for row in summary])

    figure, loss_axis = plt.subplots(figsize=(7.7, 5.0))
    imp_axis = loss_axis.twinx()
    loss_axis.plot(x, loss, color="black", linewidth=1.8, label="minibatch loss")
    loss_axis.fill_between(
        x, loss - loss_error, loss + loss_error, color="black", alpha=0.10
    )
    imp_axis.plot(
        x,
        centered,
        color="#0072B2",
        linewidth=2.2,
        label="centered median imp / final",
    )
    imp_axis.fill_between(
        x,
        np.maximum(centered - centered_error, 0.0),
        centered + centered_error,
        color="#0072B2",
        alpha=0.12,
    )
    imp_axis.plot(
        x,
        raw,
        color="#D55E00",
        linewidth=1.5,
        linestyle="--",
        label="raw median imp / final",
    )
    loss_axis.set_xlabel("Optimizer step from fresh initialization")
    loss_axis.set_ylabel("MNIST minibatch loss")
    imp_axis.set_ylabel("Importance growth (fraction of final median)")
    loss_axis.set_title("E3 automatic burn-in alignment")
    lines = loss_axis.get_lines() + imp_axis.get_lines()
    loss_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)
    loss_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "burn_in_alignment.png", dpi=180)
    plt.close(figure)


def contamination_window_value(
    summary: Sequence[Mapping],
    shift: str,
    estimator: str,
    subset: str,
    field: str,
) -> Optional[float]:
    """Look up one aggregated transient/steady contamination statistic."""
    values = [
        float(row[field])
        for row in summary
        if row["shift_type"] == shift
        and row["estimator"] == estimator
        and row["subset"] == subset
    ]
    if len(values) > 1:
        raise ValueError(
            f"duplicate contamination window rows for {shift}/{estimator}/{subset}"
        )
    return values[0] if values else None


def top5_transient_separation(window_by_seed: Sequence[Mapping]) -> Optional[float]:
    """Median paired raw/centered transient-peak ratio on hard-shift top weights."""
    paired: Dict[int, Dict[str, float]] = defaultdict(dict)
    for row in window_by_seed:
        if (
            row["shift_type"] == "coherent_hard"
            and row["subset"] == "top5_new_gradient"
        ):
            paired[int(float(row["seed"]))][str(row["estimator"])] = float(
                row["transient_peak"]
            )
    ratios = []
    for values in paired.values():
        if "raw" not in values or "centered" not in values:
            continue
        ratios.append(
            math.inf
            if values["centered"] <= 0.0
            else values["raw"] / values["centered"]
        )
    return median(ratios) if ratios else None


def assess_experiment(
    window_summary: Sequence[Mapping],
    window_by_seed: Sequence[Mapping],
    burn_assessment: Mapping,
    functional_summary: Sequence[Mapping],
) -> dict:
    """Apply the hard-shift three-outcome preregistered decision rule."""
    raw_all = contamination_window_value(
        window_summary,
        "coherent_hard",
        "raw",
        "all",
        "transient_peak_median",
    )
    centered_all = contamination_window_value(
        window_summary,
        "coherent_hard",
        "centered",
        "all",
        "transient_peak_median",
    )
    separation_top5 = top5_transient_separation(window_by_seed)

    functional_by_arm = {row["arm"]: row for row in functional_summary}
    centered_functional = functional_by_arm.get("centered_protected")
    raw_functional = functional_by_arm.get("raw_protected")
    reference = functional_by_arm.get("adamw")
    functional_available = all(
        row is not None for row in (centered_functional, raw_functional, reference)
    )
    functional_pass = False
    raw_dramatically_slower = False
    binds_engaged = False
    if functional_available:
        centered_functional = dict(centered_functional)  # type: ignore[arg-type]
        raw_functional = dict(raw_functional)  # type: ignore[arg-type]
        functional_pass = (
            float(centered_functional["reach_rate"]) == 1.0
            and float(centered_functional["slowdown_vs_adamw"]) <= 1.20
        )
        raw_dramatically_slower = (
            float(raw_functional["slowdown_vs_adamw"]) >= 2.0
            or float(raw_functional["reach_rate"]) <= 0.5
        )
        functional_pass = functional_pass and raw_dramatically_slower
        binds_engaged = (
            float(centered_functional["early_bind_rate_mean"]) > 0.0
            and float(raw_functional["early_bind_rate_mean"]) > 0.0
        )

    burn_available = bool(burn_assessment.get("available", False))
    burn_pass = bool(burn_assessment.get("visibly_lags", False))
    if any(value is None for value in (raw_all, centered_all, separation_top5)):
        raise ValueError(
            "the preregistered verdict requires complete coherent_hard "
            "contamination measurements"
        )
    flood_present = raw_all > 10.0
    centered_suppressed = centered_all < 2.0
    top5_separated = separation_top5 >= 3.0
    if not flood_present:
        verdict = "FLOODING-ABSENT (valid negative — remove centering)"
    elif (
        centered_suppressed
        and top5_separated
        and functional_pass
        and binds_engaged
        and burn_pass
    ):
        verdict = "PASS"
    else:
        verdict = "HARD FAIL"
    return {
        "verdict": verdict,
        "preregistered_complete": True,
        "raw_all_transient_peak": raw_all,
        "centered_all_transient_peak": centered_all,
        "top5_transient_separation": separation_top5,
        "flood_present": flood_present,
        "centered_suppressed": centered_suppressed,
        "top5_separated": top5_separated,
        "functional_available": functional_available,
        "functional_pass": functional_pass,
        "raw_dramatically_slower_or_stalled": raw_dramatically_slower,
        "binds_engaged": binds_engaged,
        "burn_in_available": burn_available,
        "burn_in_pass": burn_pass,
    }


def format_optional(value: Optional[float]) -> str:
    """Format an optional scalar for the Markdown report."""
    return "not measured" if value is None else f"{value:.3g}×"


def build_report(
    assessment: Mapping,
    window_summary: Sequence[Mapping],
    burn_assessment: Mapping,
    functional_summary: Sequence[Mapping],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> str:
    """Create the accuracy-windowed, three-outcome preregistered E3 report."""
    lines = [
        "# E3 report — centered importance across shifts",
        "",
        f"**Verdict: {assessment['verdict']}.**",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        f"Measurement retention: `{args.measurement_retention}`",
        f"Functional retention: `{args.functional_retention}`",
        "",
    ]
    if args.measurement_retention != "leak":
        lines.extend(
            [
                (
                    "**WARNING: contamination peaks are steady-state-conflated "
                    "under ratchet/release; verdict invalid.**"
                ),
                "",
            ]
        )
    lines.extend(
        [
        (
            "C(t) is the median of the per-weight ratio to the importance just "
            "before drift onset. Matrix weights only are included. The top-5% "
            "subset is fixed from |g| on the first new-task minibatch. Exact "
            "zero-to-zero ratios are one; newly nonzero weights use the "
            f"documented relative denominator floor ({args.ratio_floor:g})."
        ),
        "",
        (
            "A transient ends at the first logged step reaching 90% of that "
            "seed's final Task-B accuracy. Gradual shift uses the full measured "
            "window. Peaks are computed within seed before taking the median."
        ),
        "",
        "## Transient and steady-state contamination",
        "",
        "| Shift | Median transient end | Estimator | Weights | Transient peak | Steady state |",
        "|---|---:|---|---|---:|---:|",
        ]
    )
    summary_lookup = {
        (row["shift_type"], row["estimator"], row["subset"]): row
        for row in window_summary
    }
    for shift in SHIFT_TYPES:
        for estimator in ESTIMATORS:
            for subset in SUBSETS:
                row = summary_lookup.get((shift, estimator, subset))
                if row is None:
                    transient_end = "—"
                    transient_peak = "not measured"
                    steady_state = "not measured"
                else:
                    transient_end = f"{float(row['transient_end_median']):g}"
                    transient_peak = format_optional(
                        float(row["transient_peak_median"])
                    )
                    steady_state = format_optional(
                        float(row["steady_state_median"])
                    )
                subset_label = (
                    "all" if subset == "all" else "top 5% new-task |g|"
                )
                lines.append(
                    f"| {shift} | {transient_end} | {estimator} | "
                    f"{subset_label} | {transient_peak} | {steady_state} |"
                )

    raw_all = assessment.get("raw_all_transient_peak")
    centered_all = assessment.get("centered_all_transient_peak")
    separation = assessment.get("top5_transient_separation")
    raw_control_label = "PASS" if assessment.get("flood_present") else "CONTROL FAILED"
    centered_label = "PASS" if assessment.get("centered_suppressed") else "NOT PASSED"
    separation_label = "PASS" if assessment.get("top5_separated") else "NOT PASSED"
    lines.extend(
        [
            "",
            (
                "Validity control: raw all-weights transient peak > 10× "
                f"(measured: {format_optional(raw_all)}): **{raw_control_label}**."
            ),
            (
                "Treatment: centered all-weights transient peak < 2× "
                f"(measured: {format_optional(centered_all)}): **{centered_label}**."
            ),
            (
                "Top-5% separation ≥ 3 "
                f"(measured: {format_optional(separation)}): **{separation_label}**."
            ),
            "",
        ]
    )
    if not assessment.get("flood_present"):
        lines.extend(
            [
                (
                    "Because the validity control failed, the centered and "
                    "functional results are descriptive only and carry no "
                    "verdict weight."
                ),
                "",
            ]
        )
    incoherent_peak = contamination_window_value(
        window_summary,
        "incoherent",
        "centered",
        "all",
        "transient_peak_median",
    )
    if incoherent_peak is not None:
        if 3.0 <= incoherent_peak <= 10.0:
            leakage = "the expected 3–10× partial-fail band"
        elif incoherent_peak < 3.0:
            leakage = "below the expected leakage band"
        else:
            leakage = "above the expected leakage band"
        lines.extend(
            [
                (
                    f"Incoherent centered leakage peaks at {incoherent_peak:.3g}×, "
                    f"{leakage}; use this measured value when choosing T_s."
                ),
                "",
            ]
        )

    lines.extend(["## Automatic burn-in", ""])
    if burn_assessment.get("available"):
        lag = burn_assessment.get("median_lag_steps")
        lines.append(
            "Centered importance reaches 10% of its final median "
            f"{lag:+g} logged steps after the loss completes 90% of its descent "
            f"(median across seeds): **{'PASS' if assessment['burn_in_pass'] else 'NOT PASSED'}**."
        )
    else:
        lines.append("Burn-in was not measured in this run.")
    lines.extend(["", "![Burn-in alignment](burn_in_alignment.png)", ""])

    lines.extend(
        [
            "## Functional hard-coherent-shift test",
            "",
            (
                f"Target Task-B accuracy: {args.target_accuracy:.1%}. Stalled runs "
                "are conservatively counted at the configured B-step horizon in "
                "the capped mean."
            ),
            "",
        ]
    )
    if functional_summary:
        lines.extend(
            [
                "| Arm | Reached | Mean steps (reached) | Capped mean | Slowdown | Early bind rate | A at shift | A after B | Final B |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in functional_summary:
            reached_mean = float(row["steps_to_target_mean_reached"])
            reached_text = "—" if math.isnan(reached_mean) else f"{reached_mean:.1f}"
            bind_rate = float(row["early_bind_rate_mean"])
            bind_text = "—" if math.isnan(bind_rate) else f"{bind_rate:.3%}"
            lines.append(
                f"| {row['arm']} | {row['seeds_reached']}/{row['seeds']} | "
                f"{reached_text} | {float(row['capped_steps_mean']):.1f} | "
                f"{float(row['slowdown_vs_adamw']):.2f}× | "
                f"{bind_text} | "
                f"{float(row['task_a_accuracy_at_shift_mean']):.3f} | "
                f"{float(row['task_a_accuracy_after_b_mean']):.3f} | "
                f"{float(row['final_b_accuracy_mean']):.3f} |"
            )
        functional_label = (
            "DESCRIPTIVE ONLY — FLOODING CONTROL FAILED"
            if not assessment.get("flood_present")
            else ("PASS" if assessment.get("functional_pass") else "NOT PASSED")
        )
        lines.extend(
            [
                "",
                (
                    "Functional criterion (centered ≤1.20× AdamW and raw ≥2× "
                    "or stalls in at least half the seeds): "
                    f"**{functional_label}**."
                ),
                "",
            ]
        )
    else:
        lines.extend(["Functional test was skipped.", ""])
    if assessment.get("flood_present") and not assessment.get("binds_engaged"):
        lines.extend(
            [
                "**protection never engaged — functional comparison invalid**",
                "",
            ]
        )

    lines.extend(["## Preregistered conclusion", ""])
    if assessment["verdict"] == "FLOODING-ABSENT (valid negative — remove centering)":
        lines.extend(
            [
                (
                    "Raw importance did not flood (transient peak ≤ 10×) under "
                    "a hard sustained shift. Pre-registered conclusion: transient "
                    "flooding is not a failure mode at these timescales; remove "
                    "the centering component (optimizer v2.3). Do not build the "
                    "settledness gate."
                ),
                "",
            ]
        )
    elif assessment["verdict"] == "PASS":
        lines.extend(
            ["Centering suppressed a demonstrated flood; component retained.", ""]
        )
    elif assessment["verdict"] == "HARD FAIL":
        lines.extend(
            [
                (
                    "Centered importance flooded on the coherent shift. The next "
                    "candidate is a settledness gate on accumulation. Before it is "
                    "adopted, re-check the deadlock rule: the gate may read only "
                    "pre-clip signals."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The hard sustained control was not measured; no verdict is valid.",
                "",
            ]
        )
    lines.extend(
        [
            "## Figures",
            "",
            "![Coherent contamination](contamination_coherent.png)",
            "",
            "![Incoherent contamination](contamination_incoherent.png)",
            "",
            "![Gradual contamination](contamination_gradual.png)",
            "",
            "![Hard coherent contamination](contamination_coherent_hard.png)",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse and validate the E3 command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--shift-types", nargs="+", choices=SHIFT_TYPES, default=SHIFT_TYPES
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e3"))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-functional", action="store_true")
    parser.add_argument("--skip-burn-in", action="store_true")
    parser.add_argument("--input-permutation-seed", type=int, default=2026)
    parser.add_argument("--label-permutation-seed", type=int, default=2027)
    parser.add_argument("--pre-shift-steps", type=int, default=5_000)
    parser.add_argument("--post-shift-steps", type=int, default=4_000)
    parser.add_argument("--functional-max-b-steps", type=int, default=5_000)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=1_000)
    parser.add_argument("--contamination-log-every", type=int, default=10)
    parser.add_argument("--burn-in-log-every", type=int, default=10)
    parser.add_argument("--functional-eval-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--T-s", dest="T_s", type=float, default=T_S)
    parser.add_argument("--T-a", dest="T_a", type=float, default=T_A)
    parser.add_argument("--T-m", dest="T_m", type=float, default=T_M)
    parser.add_argument(
        "--measurement-retention",
        choices=("leak", "ratchet", "release"),
        default="leak",
    )
    parser.add_argument(
        "--functional-retention",
        choices=("leak", "ratchet", "release"),
        default="release",
    )
    parser.add_argument(
        "--ratio-floor",
        type=float,
        default=1e-12,
        help="Fraction of the positive baseline median used for zero denominators.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    positive = {
        "pre-shift-steps": args.pre_shift_steps,
        "post-shift-steps": args.post_shift_steps,
        "functional-max-b-steps": args.functional_max_b_steps,
        "train-batch-size": args.train_batch_size,
        "evaluation-batch-size": args.evaluation_batch_size,
        "contamination-log-every": args.contamination_log_every,
        "burn-in-log-every": args.burn_in_log_every,
        "functional-eval-every": args.functional_eval_every,
        "lr": args.lr,
        "adam-eps": args.adam_eps,
        "k": args.k,
        "T-s": args.T_s,
        "T-a": args.T_a,
        "T-m": args.T_m,
        "ratio-floor": args.ratio_floor,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"these arguments must be positive: {', '.join(invalid)}")
    if not args.seeds:
        parser.error("at least one seed is required")
    if not 0.0 < args.target_accuracy <= 1.0:
        parser.error("--target-accuracy must lie in (0, 1]")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be nonnegative")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        parser.error("--beta1 and --beta2 must lie in [0, 1)")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def main() -> None:
    """Run E3, summarize all paired measurements, and write the report."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    contamination_path = args.output_dir / "contamination_by_seed.csv"
    burn_path = args.output_dir / "burn_in_by_seed.csv"
    functional_path = args.output_dir / "learning_speed_by_seed.csv"
    functional_trajectory_path = args.output_dir / "functional_trajectories.csv"

    input_permutation = fixed_permutation(28 * 28, args.input_permutation_seed)
    label_permutation = fixed_permutation(47, args.label_permutation_seed)
    if args.summarize_only:
        contamination_rows = read_csv(contamination_path)
        burn_rows = read_csv(burn_path)
        functional_rows = read_csv(functional_path)
        functional_trajectories = read_csv(functional_trajectory_path)
        print("Loaded existing E3 CSVs; training was skipped.")
    else:
        needs_mnist = any(
            shift in args.shift_types for shift in ("coherent", "gradual")
        )
        needs_emnist = any(
            shift in args.shift_types for shift in ("incoherent", "coherent_hard")
        ) or not args.skip_functional
        mnist_train: Optional[Dataset] = None
        mnist_test: Optional[Dataset] = None
        emnist_train: Optional[Dataset] = None
        emnist_test: Optional[Dataset] = None
        if needs_mnist:
            mnist_train, mnist_test = load_mnist(args.data_dir)
        if needs_emnist:
            emnist_train, emnist_test = load_emnist(args.data_dir)

        torch.save(input_permutation, args.output_dir / "input_permutation.pt")
        torch.save(label_permutation, args.output_dir / "label_permutation.pt")
        contamination_rows: List[dict] = []
        burn_rows: List[dict] = []
        print(
            f"Running E3 passive streams on {device}; seeds={tuple(args.seeds)}; "
            f"shifts={tuple(args.shift_types)}"
        )
        for shift_type in args.shift_types:
            task_a, task_b, task_b_test, output_units = stream_datasets(
                shift_type,
                mnist_train,
                mnist_test,
                emnist_train,
                emnist_test,
                input_permutation,
                label_permutation,
            )
            for seed in args.seeds:
                print(f"  passive {shift_type}: seed={seed}")
                run_contamination, run_burn = run_passive_stream(
                    shift_type,
                    task_a,
                    task_b,
                    task_b_test,
                    output_units,
                    seed,
                    args,
                    device,
                    collect_burn_in=(
                        shift_type == "coherent" and not args.skip_burn_in
                    ),
                )
                contamination_rows.extend(run_contamination)
                burn_rows.extend(run_burn)
                write_csv(contamination_path, contamination_rows)
                write_csv(burn_path, burn_rows)

        functional_rows: List[dict] = []
        functional_trajectories: List[dict] = []
        if not args.skip_functional:
            if emnist_train is None or emnist_test is None:
                raise RuntimeError("functional comparison requires EMNIST")
            functional_a_train = ClassSubset(emnist_train, range(0, 10))
            functional_a_test = ClassSubset(emnist_test, range(0, 10))
            functional_b_train = ClassSubset(emnist_train, range(10, 47))
            functional_b_test = ClassSubset(emnist_test, range(10, 47))
            functional_output_units = 47
            print("Running active hard-coherent-shift functional comparison")
            for seed in args.seeds:
                seed_everything(seed)
                initial_model = MLP300(
                    input_units=28 * 28, output_units=functional_output_units
                )
                initial_state = {
                    name: value.detach().clone()
                    for name, value in initial_model.state_dict().items()
                }
                for arm in ("adamw", "centered_protected", "raw_protected"):
                    print(f"  functional {arm}: seed={seed}")
                    result, trajectory = train_functional_arm(
                        arm,
                        initial_state,
                        functional_output_units,
                        functional_a_train,
                        functional_a_test,
                        functional_b_train,
                        functional_b_test,
                        seed,
                        args,
                        device,
                    )
                    functional_rows.append(result)
                    functional_trajectories.extend(trajectory)
                    write_csv(functional_path, functional_rows)
                    write_csv(
                        functional_trajectory_path, functional_trajectories
                    )

    (
        contamination_summary,
        contamination_windows_by_seed,
        contamination_windows_summary,
    ) = summarize_contamination(contamination_rows, args.post_shift_steps)
    normalized_burn = normalized_burn_in_rows(burn_rows)
    burn_summary = summarize_burn_in(normalized_burn)
    burn_assessment = assess_burn_in(normalized_burn)
    functional_summary = summarize_functional(functional_rows)
    assessment = assess_experiment(
        contamination_windows_summary,
        contamination_windows_by_seed,
        burn_assessment,
        functional_summary,
    )

    write_csv(args.output_dir / "contamination_summary.csv", contamination_summary)
    write_csv(
        args.output_dir / "contamination_windows_by_seed.csv",
        contamination_windows_by_seed,
    )
    write_csv(
        args.output_dir / "contamination_windows_summary.csv",
        contamination_windows_summary,
    )
    write_csv(args.output_dir / "burn_in_normalized_by_seed.csv", normalized_burn)
    write_csv(args.output_dir / "burn_in_summary.csv", burn_summary)
    write_csv(args.output_dir / "learning_speed_table.csv", functional_summary)
    plot_contamination(
        contamination_summary, contamination_windows_summary, args.output_dir
    )
    plot_burn_in(burn_summary, args.output_dir)

    observed_seeds = sorted(
        {int(float(row["seed"])) for row in contamination_rows}
        | {int(float(row["seed"])) for row in functional_rows}
    )
    report = build_report(
        assessment,
        contamination_windows_summary,
        burn_assessment,
        functional_summary,
        observed_seeds,
        args,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    metadata = {
        "seeds": observed_seeds,
        "device": str(device),
        "input_permutation": input_permutation.tolist(),
        "label_permutation": label_permutation.tolist(),
        "assessment": assessment,
        "burn_in_assessment": burn_assessment,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\n" + report)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
