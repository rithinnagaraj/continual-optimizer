"""E1: test whether CAP-Adam importance predicts forgetting damage.

The default command runs five seeds on Fashion-MNIST and MNIST, writes the
six saved signals and trained model for every run, performs the individual
and group probes, and creates a short pass/fail report.

Example:
    python patched_e1.py --data-dir data --output-dir results/e1

For a quick smoke test, reduce ``--epochs``, ``--probe-weights``,
``--held-out``, and ``--group-trials``.  Such a run is useful for debugging,
but is not a valid E1 result.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from model import MLP300
from cap_adam_experimental import RawImportanceTracker, CAPAdam


SEEDS = (42, 123, 2026, 451, 789)
K_VALUES = (0.5, 1.0, 2.0, 4.0)
SIGNAL_NAMES = (
    "imp",
    "raw_g2",
    "batch_fisher",
    "fisher",
    "weight_magnitude",
    "si",
)
GROUP_FRACTIONS = (0.01, 0.05, 0.20)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
T_S = 2_000.0
T_A = 2_000.0
T_M = 10_000.0
C_REFRESH_STEPS = 100


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(name: str, root: Path) -> Tuple[Dataset, Dataset]:
    """Download and return the requested train/test dataset pair."""
    if name == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
    elif name == "mnist":
        dataset_class = datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = dataset_class(root=root, train=True, download=True, transform=transform)
    test_set = dataset_class(root=root, train=False, download=True, transform=transform)
    return train_set, test_set


def make_parameter_groups(model: nn.Module) -> List[dict]:
    """Use one group per tensor so each layer can have its own decay blend c."""
    groups = []
    for name, parameter in model.named_parameters():
        is_weight = parameter.ndim >= 2
        groups.append(
            {
                "params": [parameter],
                "param_name": name,
                "protect": is_weight,
                "weight_decay": WEIGHT_DECAY if is_weight else 0.0,
            }
        )
    return groups


def refresh_layer_c(optimizer: CAPAdam) -> None:
    """Set each parameter group's c to the median of its current importance."""
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        state = optimizer.state.get(parameter)
        if state and "imp" in state:
            # c must remain positive while the initial importance is all zero.
            group["c"] = max(float(state["imp"].median().item()), 1e-12)


def train_model(
    train_set: Dataset,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[MLP300, CAPAdam, RawImportanceTracker, Dict[str, Tensor], List[float]]:
    """Train MLP-300 passively and collect centered, raw, and SI statistics."""
    seed_everything(seed)
    model = MLP300(input_units=28 * 28, output_units=10).to(device)
    optimizer = CAPAdam(
        make_parameter_groups(model),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        T_s=T_S,
        T_a=T_A,
        T_m=T_M,
        k=2.0,
        passive=True,
        retention="ratchet",
        importance_evidence="centered",
    )
    raw_tracker = RawImportanceTracker(optimizer)

    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    si_path = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    losses: List[float] = []
    global_step = 0
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        example_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()

            before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            optimizer.step()
            raw_tracker.update()

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    # Synaptic Intelligence path contribution: -g * delta_theta.
                    si_path[name].add_(
                        -parameter.grad * (parameter.detach() - before[name])
                    )

            global_step += 1
            if global_step % C_REFRESH_STEPS == 0:
                refresh_layer_c(optimizer)

            loss_sum += float(loss.item()) * targets.numel()
            example_count += targets.numel()

        epoch_loss = loss_sum / example_count
        losses.append(epoch_loss)
        print(f"  seed={seed} epoch={epoch + 1:02d}/{epochs} loss={epoch_loss:.5f}")

    si = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            displacement_sq = (parameter - initial[name]).square()
            si[name] = si_path[name].clamp_min(0.0) / (displacement_sq + 1e-3)
    return model, optimizer, raw_tracker, si, losses


def compute_diagonal_fisher(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Tensor]:
    """Compute the classical diagonal Fisher from per-sample gradients."""
    model.eval()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    fisher = {
        name: torch.zeros_like(parameter) for name, parameter in parameters.items()
    }

    def sample_loss(
        current_parameters: Mapping[str, Tensor],
        current_buffers: Mapping[str, Tensor],
        sample: Tensor,
        target: Tensor,
    ) -> Tensor:
        logits = functional_call(
            model,
            (current_parameters, current_buffers),
            (sample.unsqueeze(0),),
        )
        return F.cross_entropy(logits, target.unsqueeze(0))

    sample_gradient = grad(sample_loss)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        gradients = vmap(
            sample_gradient,
            in_dims=(None, None, 0, 0),
            randomness="different",
        )(parameters, buffers, inputs, targets)
        with torch.no_grad():
            for name, values in gradients.items():
                fisher[name].add_(values.square().sum(dim=0))
        seen += targets.numel()
        del gradients

    for value in fisher.values():
        value.div_(seen)
    return fisher


def compute_batch_fisher(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Tensor]:
    """Average squared batch-mean gradients on the frozen model."""
    model.eval()
    named_parameters = list(model.named_parameters())
    parameters = tuple(parameter for _, parameter in named_parameters)
    batch_fisher = {
        name: torch.zeros_like(parameter) for name, parameter in named_parameters
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = F.cross_entropy(model(inputs), targets)
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for (name, _), gradient in zip(named_parameters, gradients):
                batch_fisher[name].add_(gradient.square())
        batches += 1
        del gradients

    for value in batch_fisher.values():
        value.div_(batches)
    return batch_fisher


def collect_held_out(
    dataset: Dataset,
    count: int,
    seed: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Materialize a deterministic held-out subset for repeated probes."""
    count = min(count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=count)
    inputs, targets = next(iter(loader))
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def test_loss(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
) -> Tensor:
    """Return one cross-entropy loss per fixed held-out example on CPU."""
    model.eval()
    losses = []
    for start in range(0, targets.numel(), batch_size):
        end = start + batch_size
        losses.append(
            F.cross_entropy(
                model(inputs[start:end]), targets[start:end], reduction="none"
            ).cpu()
        )
    return torch.cat(losses)


def weight_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Return trainable matrix-valued weights, excluding biases."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.ndim >= 2
    ]


def flatten_signals(
    model: nn.Module,
    optimizer: CAPAdam,
    raw_tracker: RawImportanceTracker,
    batch_fisher: Mapping[str, Tensor],
    fisher: Mapping[str, Tensor],
    si: Mapping[str, Tensor],
) -> Tuple[List[Tuple[str, nn.Parameter]], Dict[str, Tensor]]:
    """Flatten all six signals in a shared, weight-only parameter order."""
    weights = weight_parameters(model)
    signals: Dict[str, List[Tensor]] = {name: [] for name in SIGNAL_NAMES}
    for name, parameter in weights:
        signals["imp"].append(optimizer.state[parameter]["imp"].detach().flatten())
        signals["raw_g2"].append(raw_tracker.state[parameter].detach().flatten())
        signals["batch_fisher"].append(batch_fisher[name].detach().flatten())
        signals["fisher"].append(fisher[name].detach().flatten())
        signals["weight_magnitude"].append(parameter.detach().abs().flatten())
        signals["si"].append(si[name].detach().flatten())
    return weights, {name: torch.cat(values).cpu() for name, values in signals.items()}


def parameter_offsets(weights: Sequence[Tuple[str, nn.Parameter]]) -> List[int]:
    """Return cumulative flattened offsets, including the final endpoint."""
    offsets = [0]
    for _, parameter in weights:
        offsets.append(offsets[-1] + parameter.numel())
    return offsets


def stratified_indices(
    importance: Tensor,
    sample_count: int,
    seed: int,
    bins: int = 20,
) -> np.ndarray:
    """Sample evenly across rank bins of importance, not its near-zero bulk."""
    values = importance.numpy()
    sample_count = min(sample_count, values.size)
    ordered = np.argsort(values, kind="stable")
    rank_bins = np.array_split(ordered, min(bins, sample_count))
    rng = np.random.default_rng(seed)
    allocation = [sample_count // len(rank_bins)] * len(rank_bins)
    for index in range(sample_count % len(rank_bins)):
        allocation[index] += 1
    selected = [
        rng.choice(rank_bin, size=min(size, len(rank_bin)), replace=False)
        for rank_bin, size in zip(rank_bins, allocation)
    ]
    result = np.concatenate(selected)
    if result.size < sample_count:
        remaining = np.setdiff1d(ordered, result, assume_unique=False)
        result = np.concatenate(
            [result, rng.choice(remaining, sample_count - result.size, replace=False)]
        )
    rng.shuffle(result)
    return result


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact ties handled correctly."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: np.ndarray, damage: np.ndarray) -> float:
    """Compute Spearman rho without requiring SciPy."""
    value_ranks = average_tie_ranks(values)
    damage_ranks = average_tie_ranks(damage)
    if value_ranks.std() == 0.0 or damage_ranks.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(value_ranks, damage_ranks)[0, 1])


def measurement_ceiling(half_agreement: float) -> float:
    """Convert split-half agreement to full-measurement reliability."""
    if not math.isfinite(half_agreement) or half_agreement <= -1.0:
        return float("nan")
    reliability = 2.0 * half_agreement / (1.0 + half_agreement)
    return math.sqrt(reliability) if reliability >= 0.0 else float("nan")


def individual_damage_probe(
    model: nn.Module,
    weights: Sequence[Tuple[str, nn.Parameter]],
    signals: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    sample_count: int,
    seed: int,
    evaluation_batch_size: int,
) -> Tuple[List[dict], List[dict]]:
    """Perturb stratified individual weights and correlate signals with damage."""
    selected = stratified_indices(signals["imp"], sample_count, seed)
    offsets = parameter_offsets(weights)
    layer_delta = [
        2.0 * float(parameter.detach().std(unbiased=False).item())
        for _, parameter in weights
    ]
    baseline = test_loss(model, inputs, targets, evaluation_batch_size)
    baseline_mean = float(baseline.mean().item())
    half = baseline.numel() // 2
    symmetric_damage = np.empty(selected.size, dtype=np.float64)
    first_half_damage = np.empty(selected.size, dtype=np.float64)
    second_half_damage = np.empty(selected.size, dtype=np.float64)
    probe_rows = []

    print(
        f"  individual probe: {selected.size} weights, "
        f"baseline loss={baseline_mean:.6f}"
    )
    for probe_index, global_index in enumerate(selected):
        layer_index = bisect.bisect_right(offsets, int(global_index)) - 1
        local_index = int(global_index) - offsets[layer_index]
        parameter = weights[layer_index][1]
        flat_parameter = parameter.view(-1)
        original = flat_parameter[local_index].detach().clone()
        delta = layer_delta[layer_index]

        with torch.no_grad():
            flat_parameter[local_index] = original + delta
        plus_loss = test_loss(model, inputs, targets, evaluation_batch_size)
        with torch.no_grad():
            flat_parameter[local_index] = original - delta
        minus_loss = test_loss(model, inputs, targets, evaluation_batch_size)
        with torch.no_grad():
            flat_parameter[local_index] = original

        # Averaging +/- cancels the first-order directional term and measures
        # the local loss sensitivity that importance is intended to predict.
        plus_damage = plus_loss - baseline
        minus_damage = minus_loss - baseline
        per_image_damage = 0.5 * (plus_damage + minus_damage)
        symmetric_damage[probe_index] = float(per_image_damage.mean().item())
        first_half_damage[probe_index] = float(
            per_image_damage[:half].mean().item()
        )
        second_half_damage[probe_index] = float(
            per_image_damage[half:].mean().item()
        )
        probe_row = {
            "global_index": int(global_index),
            "parameter": weights[layer_index][0],
            "local_index": local_index,
            "original_weight": float(original.item()),
            "delta": delta,
            "plus_damage": float(plus_damage.mean().item()),
            "minus_damage": float(minus_damage.mean().item()),
            "symmetric_damage": symmetric_damage[probe_index],
            "first_half_damage": first_half_damage[probe_index],
            "second_half_damage": second_half_damage[probe_index],
        }
        for signal_name in SIGNAL_NAMES:
            probe_row[signal_name] = float(
                signals[signal_name][int(global_index)].item()
            )
        probe_rows.append(probe_row)
        if (probe_index + 1) % 100 == 0 or probe_index + 1 == selected.size:
            print(f"    completed {probe_index + 1}/{selected.size}", end="\r")
    print()

    half_agreement = spearman(first_half_damage, second_half_damage)
    ceiling = measurement_ceiling(half_agreement)
    rows = []
    for signal_name in SIGNAL_NAMES:
        sampled_signal = signals[signal_name].numpy()[selected]
        rows.append(
            {
                "signal": signal_name,
                "rho_delta": spearman(sampled_signal, symmetric_damage),
                "rho_half_1": spearman(sampled_signal, first_half_damage),
                "rho_half_2": spearman(sampled_signal, second_half_damage),
                "half_agreement": half_agreement,
                "measurement_ceiling": ceiling,
                "baseline_loss": baseline_mean,
            }
        )
    return rows, probe_rows


@torch.no_grad()
def apply_flat_delta(
    weights: Sequence[Tuple[str, nn.Parameter]],
    offsets: Sequence[int],
    delta: Tensor,
    sign: float = 1.0,
) -> None:
    """Apply a flattened perturbation vector."""
    for layer_index, (_, parameter) in enumerate(weights):
        layer_delta = delta[offsets[layer_index] : offsets[layer_index + 1]]
        parameter.add_(layer_delta.view_as(parameter), alpha=sign)


def group_trial_damage(
    model: nn.Module,
    weights: Sequence[Tuple[str, nn.Parameter]],
    offsets: Sequence[int],
    indices: Tensor,
    noise_scale: Tensor,
    inputs: Tensor,
    targets: Tensor,
    baseline: Tensor,
    evaluation_batch_size: int,
    generator: torch.Generator,
) -> float:
    """Return symmetric +/- noise damage for one selected group."""
    device = weights[0][1].device
    delta = torch.zeros(offsets[-1], device=device)
    device_indices = indices.to(device)
    noise = torch.randn(
        indices.numel(), device=device, generator=generator, dtype=delta.dtype
    )
    delta[device_indices] = noise * noise_scale[device_indices]
    original = [parameter.detach().clone() for _, parameter in weights]
    apply_flat_delta(weights, offsets, delta)
    try:
        plus_loss = test_loss(model, inputs, targets, evaluation_batch_size)
        with torch.no_grad():
            for (_, parameter), saved in zip(weights, original):
                parameter.copy_(saved)
        apply_flat_delta(weights, offsets, delta, sign=-1.0)
        minus_loss = test_loss(model, inputs, targets, evaluation_batch_size)
    finally:
        with torch.no_grad():
            for (_, parameter), saved in zip(weights, original):
                parameter.copy_(saved)
    return float((0.5 * (plus_loss + minus_loss) - baseline).mean().item())


def layer_matched_random_indices(
    top_indices: Tensor,
    offsets: Sequence[int],
    generator: torch.Generator,
) -> Tensor:
    """Draw the same number of random weights per layer as a top group."""
    parts = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        layer_count = int(((top_indices >= start) & (top_indices < end)).sum().item())
        if layer_count:
            parts.append(
                torch.randperm(end - start, generator=generator)[:layer_count] + start
            )
    return torch.cat(parts)


def group_damage_probe(
    model: nn.Module,
    weights: Sequence[Tuple[str, nn.Parameter]],
    signals: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    seed: int,
    trials: int,
    noise_fraction: float,
    evaluation_batch_size: int,
) -> List[dict]:
    """Compare top-signal groups with layer-matched random groups."""
    offsets = parameter_offsets(weights)
    device = weights[0][1].device
    scale_parts = []
    for _, parameter in weights:
        layer_std = parameter.detach().std(unbiased=False).clamp_min(1e-12)
        scale_parts.append(
            torch.full(
                (parameter.numel(),),
                noise_fraction * float(layer_std.item()),
                device=device,
            )
        )
    noise_scale = torch.cat(scale_parts)
    baseline = test_loss(model, inputs, targets, evaluation_batch_size)
    cpu_generator = torch.Generator().manual_seed(seed + 20_000)
    device_generator = torch.Generator(device=device).manual_seed(seed + 30_000)
    total_weights = offsets[-1]
    rows = []

    print(f"  group probe: {trials} noise trials per group")
    for fraction in GROUP_FRACTIONS:
        group_size = max(1, round(total_weights * fraction))
        for signal_name in SIGNAL_NAMES:
            top_indices = torch.topk(
                signals[signal_name], group_size, largest=True, sorted=False
            ).indices
            random_damage = []
            for _ in range(trials):
                random_indices = layer_matched_random_indices(
                    top_indices, offsets, cpu_generator
                )
                random_damage.append(
                    group_trial_damage(
                        model,
                        weights,
                        offsets,
                        random_indices,
                        noise_scale,
                        inputs,
                        targets,
                        baseline,
                        evaluation_batch_size,
                        device_generator,
                    )
                )
            top_damage = [
                group_trial_damage(
                    model,
                    weights,
                    offsets,
                    top_indices,
                    noise_scale,
                    inputs,
                    targets,
                    baseline,
                    evaluation_batch_size,
                    device_generator,
                )
                for _ in range(trials)
            ]
            random_mean = mean(random_damage)
            random_std = stdev(random_damage) if trials > 1 else 0.0
            top_mean = mean(top_damage)
            top_std = stdev(top_damage) if trials > 1 else 0.0
            percentile = 100.0 * sum(
                random_value <= top_mean for random_value in random_damage
            ) / trials
            rows.append(
                {
                    "signal": signal_name,
                    "fraction": fraction,
                    "group_size": group_size,
                    "damage_mean": top_mean,
                    "damage_std": top_std,
                    "random_mean": random_mean,
                    "random_std": random_std,
                    "percentile": percentile,
                    "above_95": percentile >= 95.0,
                }
            )
    return rows


def save_run_artifact(
    path: Path,
    model: nn.Module,
    signals: Mapping[str, Tensor],
    weights: Sequence[Tuple[str, nn.Parameter]],
    train_losses: Sequence[float],
    dataset_name: str,
    seed: int,
) -> None:
    """Save one trained network and the six aligned per-weight signals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": dataset_name,
            "seed": seed,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "weight_names": [name for name, _ in weights],
            "weight_shapes": [tuple(parameter.shape) for _, parameter in weights],
            "signals": dict(signals),
            "train_losses": list(train_losses),
            "constants": {
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "T_s": T_S,
                "T_a": T_A,
                "T_m": T_M,
                "k_values": K_VALUES,
                "c_refresh_steps": C_REFRESH_STEPS,
            },
        },
        path,
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write dictionaries as a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[dict]:
    """Read result dictionaries from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    """Summarize finite values without crashing on an undefined correlation."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan"), 0
    deviation = stdev(finite) if len(finite) > 1 else 0.0
    return mean(finite), deviation, len(finite)


def summarize_correlations(rows: Sequence[dict]) -> List[dict]:
    """Aggregate per-seed correlations and measurement ceilings."""
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["probe_split"], row["signal"])].append(row)

    summary = []
    dataset_splits = sorted({(row["dataset"], row["probe_split"]) for row in rows})
    for dataset_name, probe_split in dataset_splits:
        for signal_name in SIGNAL_NAMES:
            signal_rows = grouped[(dataset_name, probe_split, signal_name)]
            delta = [float(row["rho_delta"]) for row in signal_rows]
            ceilings = [float(row["measurement_ceiling"]) for row in signal_rows]
            half_agreements = [float(row["half_agreement"]) for row in signal_rows]
            delta_mean, delta_std, delta_valid = finite_mean_std(delta)
            ceiling_mean, ceiling_std, ceiling_valid = finite_mean_std(ceilings)
            half_mean, half_std, half_valid = finite_mean_std(half_agreements)
            summary.append(
                {
                    "dataset": dataset_name,
                    "probe_split": probe_split,
                    "signal": signal_name,
                    "rho_delta_mean": delta_mean,
                    "rho_delta_std": delta_std,
                    "rho_valid_seeds": delta_valid,
                    "half_agreement_mean": half_mean,
                    "half_agreement_std": half_std,
                    "half_agreement_valid_seeds": half_valid,
                    "measurement_ceiling_mean": ceiling_mean,
                    "measurement_ceiling_std": ceiling_std,
                    "ceiling_valid_seeds": ceiling_valid,
                    "seeds": len(signal_rows),
                }
            )
    return summary


def summarize_groups(rows: Sequence[dict]) -> List[dict]:
    """Aggregate group damage and empirical percentiles over seeds."""
    grouped: Dict[Tuple[str, str, float], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["signal"], float(row["fraction"]))].append(row)

    summary = []
    for key in sorted(grouped):
        dataset_name, signal_name, fraction = key
        signal_rows = grouped[key]
        damage = [float(row["damage_mean"]) for row in signal_rows]
        random_damage = [float(row["random_mean"]) for row in signal_rows]
        percentiles = [float(row["percentile"]) for row in signal_rows]
        summary.append(
            {
                "dataset": dataset_name,
                "signal": signal_name,
                "fraction": fraction,
                "damage_mean": mean(damage),
                "damage_std_across_seeds": stdev(damage) if len(damage) > 1 else 0.0,
                "random_mean": mean(random_damage),
                "random_std_across_seeds": (
                    stdev(random_damage) if len(random_damage) > 1 else 0.0
                ),
                "percentile_mean": mean(percentiles),
                "percentile_std": (
                    stdev(percentiles) if len(percentiles) > 1 else 0.0
                ),
                "seeds_above_95": sum(value >= 95.0 for value in percentiles),
                "seeds": len(signal_rows),
            }
        )
    return summary


def plot_group_curves(rows: Sequence[dict], output_dir: Path) -> None:
    """Save one mean group-damage curve plot per dataset."""
    import matplotlib.pyplot as plt

    for dataset_name in sorted({row["dataset"] for row in rows}):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        for signal_name in SIGNAL_NAMES:
            signal_rows = sorted(
                (row for row in dataset_rows if row["signal"] == signal_name),
                key=lambda row: float(row["fraction"]),
            )
            x = [100.0 * float(row["fraction"]) for row in signal_rows]
            y = [float(row["damage_mean"]) for row in signal_rows]
            error = [float(row["damage_std_across_seeds"]) for row in signal_rows]
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=signal_name)

        random_rows = sorted(
            (
                row
                for row in dataset_rows
                if row["signal"] == SIGNAL_NAMES[0]
            ),
            key=lambda row: float(row["fraction"]),
        )
        axis.errorbar(
            [100.0 * float(row["fraction"]) for row in random_rows],
            [float(row["random_mean"]) for row in random_rows],
            yerr=[float(row["random_std_across_seeds"]) for row in random_rows],
            color="black",
            linestyle="--",
            marker="x",
            capsize=3,
            label="layer-matched random (imp)",
        )
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xlabel("Noised weights (%)")
        axis.set_ylabel("Increase in held-out loss")
        axis.set_title(f"E1 group damage — {dataset_name}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"group_damage_{dataset_name}.png", dpi=180)
        plt.close(figure)


def format_mean_std(value: float, deviation: float) -> str:
    """Format a mean and sample standard deviation compactly."""
    return f"{value:.3f} ± {deviation:.3f}"


def build_report(
    correlation_summary: Sequence[dict],
    group_summary: Sequence[dict],
    correlation_rows: Sequence[dict],
    group_rows: Sequence[dict],
    sensor_rows: Sequence[dict],
    seeds: Sequence[int],
) -> str:
    """Create the criteria-v2 tables and apply paired per-seed grading."""
    lines = [
        "# E1 report — Criteria version 2 (amended after the reanalysis)",
        "",
        (
            "The absolute correlation bars moved because split-half reliability "
            "showed that damage measurements have a dataset-dependent ceiling; "
            "the unstable damage ratio moved to a layer-matched percentile; and "
            "imp-versus-Fisher grading is now paired within seed. See "
            "[reanalysis_report.md](reanalysis_report.md)."
        ),
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        (
            "The individual score is symmetric ±δ damage over all probe images. "
            "Each score is displayed beside its split-half measurement ceiling."
        ),
        "",
    ]

    datasets_present = sorted({row["dataset"] for row in correlation_summary})
    for dataset_name in datasets_present:
        lines.extend([f"## {dataset_name}", ""])
        splits = sorted(
            {
                row["probe_split"]
                for row in correlation_summary
                if row["dataset"] == dataset_name
            },
            key=lambda split: split != "test",
        )
        for probe_split in splits:
            rows = [
                row
                for row in correlation_summary
                if row["dataset"] == dataset_name
                and row["probe_split"] == probe_split
            ]
            by_signal = {row["signal"]: row for row in rows}
            lines.extend(
                [
                    f"### {probe_split}-image damage",
                    "",
                    "| Signal | Spearman ρ, ±δ damage | Measurement ceiling |",
                    "|---|---:|---:|",
                ]
            )
            for signal_name in SIGNAL_NAMES:
                row = by_signal[signal_name]
                rho_text = format_mean_std(
                    row["rho_delta_mean"], row["rho_delta_std"]
                )
                ceiling_text = format_mean_std(
                    row["measurement_ceiling_mean"],
                    row["measurement_ceiling_std"],
                )
                lines.append(
                    f"| {signal_name} | {rho_text} | {ceiling_text} |"
                )
            lines.append("")

        test_rows = [
            row
            for row in correlation_rows
            if row["dataset"] == dataset_name and row["probe_split"] == "test"
        ]
        by_seed: Dict[int, Dict[str, float]] = defaultdict(dict)
        for row in test_rows:
            by_seed[int(row["seed"])][row["signal"]] = float(row["rho_delta"])

        required_signals = ("imp", "fisher", "weight_magnitude")
        paired = [
            values
            for values in by_seed.values()
            if all(
                signal in values and math.isfinite(values[signal])
                for signal in required_signals
            )
        ]
        fisher_imp_gaps = [values["fisher"] - values["imp"] for values in paired]
        fisher_ratios = [
            values["imp"] / values["fisher"]
            for values in paired
            if values["fisher"] > 0.0
        ]
        magnitude_advantages = [
            values["imp"] - values["weight_magnitude"] for values in paired
        ]
        relative_score, relative_std, _ = finite_mean_std(fisher_ratios)
        gap_mean, gap_std, _ = finite_mean_std(fisher_imp_gaps)
        magnitude_advantage, magnitude_advantage_std, _ = finite_mean_std(
            magnitude_advantages
        )

        imp_group_runs = [
            row
            for row in group_rows
            if row["dataset"] == dataset_name and row["signal"] == "imp"
        ]
        group_by_seed: Dict[int, Dict[float, float]] = defaultdict(dict)
        for row in imp_group_runs:
            group_by_seed[int(row["seed"])][float(row["fraction"])] = float(
                row["percentile"]
            )
        seed_count = len(group_by_seed)
        required_successes = math.ceil(0.8 * seed_count)
        success_counts = {
            fraction: sum(
                values.get(fraction, -math.inf) >= 95.0
                for values in group_by_seed.values()
            )
            for fraction in (0.05, 0.20)
        }
        group_passes = seed_count > 0 and all(
            count >= required_successes for count in success_counts.values()
        )
        both_budgets_fail = sum(
            values.get(0.05, -math.inf) < 95.0
            and values.get(0.20, -math.inf) < 95.0
            for values in group_by_seed.values()
        )
        group_hard_fails = seed_count > 0 and both_budgets_fail > seed_count / 2

        hard_fail = (
            math.isfinite(relative_score) and relative_score < 0.6
        ) or group_hard_fails
        passes = (
            math.isfinite(relative_score)
            and relative_score >= 0.8
            and group_passes
            and magnitude_advantage > 0.0
        )
        if hard_fail:
            verdict = "HARD FAIL"
            next_step = (
                "Time-box the fix loop to one week: try a longer T_s, per-layer "
                "normalization, then per-neuron importance aggregation. If the "
                "failure remains, test the relative-usage fallback before Phase 1."
            )
        elif passes:
            verdict = "PASS"
            next_step = "The importance signal clears E1 and can proceed downstream."
        else:
            verdict = "INCONCLUSIVE"
            next_step = (
                "The result lies between the preregistered pass and fail bars; "
                "inspect seed variance and group curves before proceeding."
            )

        group_parts = []
        for row in sorted(
            (
                row
                for row in group_summary
                if row["dataset"] == dataset_name and row["signal"] == "imp"
            ),
            key=lambda item: item["fraction"],
        ):
            group_parts.append(
                f"top-{100 * float(row['fraction']):g}%: "
                f"percentile {row['percentile_mean']:.1f} ± "
                f"{row['percentile_std']:.1f}, "
                f"{row['seeds_above_95']}/{row['seeds']} seeds above 95th"
            )
        lines.extend(
            [
                "",
                f"**Verdict: {verdict}.**",
                "",
                (
                    f"Paired imp/Fisher score ratio={relative_score:.3f} ± "
                    f"{relative_std:.3f}; paired "
                    f"Fisher-minus-imp gap={gap_mean:.3f} ± {gap_std:.3f}; "
                    f"paired imp-minus-|θ| advantage={magnitude_advantage:.3f} "
                    f"± {magnitude_advantage_std:.3f}."
                ),
                "imp group test: " + "; ".join(group_parts) + ".",
                next_step,
                "",
                "### Direct sensor agreement by seed",
                "",
            ]
        )
        for row in sorted(
            (row for row in sensor_rows if row["dataset"] == dataset_name),
            key=lambda item: int(item["seed"]),
        ):
            lines.append(
                f"- seed {row['seed']}: imp↔Fisher ρ="
                f"{float(row['imp_fisher_rho']):.3f}; imp↔batch-Fisher ρ="
                f"{float(row['imp_batch_fisher_rho']):.3f}; "
                f"batch-Fisher↔Fisher ρ="
                f"{float(row['batch_fisher_fisher_rho']):.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Criteria version 2",
            "",
            (
                "PASS requires all three: imp scores at least 0.8× Fisher on the "
                "paired probes; top-5% and top-20% imp groups exceed the 95th "
                "random percentile in at least 4/5 seeds; and imp beats |θ|."
            ),
            (
                "HARD FAIL means imp scores below 0.6× Fisher, or both group "
                "budgets fail in most seeds. Everything between is INCONCLUSIVE."
            ),
            "",
            (
                "k was not redundantly swept during E1 training: passive=True makes "
                "the parameter path and both tracked statistics exactly independent "
                f"of k. The requested values {K_VALUES} are saved in every artifact."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_one(
    dataset_name: str,
    seed: int,
    train_set: Dataset,
    test_set: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[dict], List[dict], dict]:
    """Train and execute all E1 probes for one dataset/seed pair."""
    print(f"\n[{dataset_name}] seed={seed}")
    model, optimizer, raw_tracker, si, train_losses = train_model(
        train_set,
        seed,
        device,
        args.epochs,
        args.train_batch_size,
        args.num_workers,
    )
    print("  computing batch-Fisher with training-size batches")
    batch_fisher = compute_batch_fisher(
        model,
        train_set,
        device,
        args.train_batch_size,
        args.num_workers,
    )
    print("  computing full-data per-sample diagonal Fisher")
    fisher = compute_diagonal_fisher(
        model,
        train_set,
        device,
        args.fisher_batch_size,
        args.num_workers,
    )
    weights, signals = flatten_signals(
        model, optimizer, raw_tracker, batch_fisher, fisher, si
    )
    sensor_row = {
        "dataset": dataset_name,
        "seed": seed,
        "imp_fisher_rho": spearman(
            signals["imp"].numpy(), signals["fisher"].numpy()
        ),
        "imp_batch_fisher_rho": spearman(
            signals["imp"].numpy(), signals["batch_fisher"].numpy()
        ),
        "batch_fisher_fisher_rho": spearman(
            signals["batch_fisher"].numpy(), signals["fisher"].numpy()
        ),
    }
    held_inputs, held_targets = collect_held_out(
        test_set, args.held_out, seed + 10_000, device
    )

    artifact_path = args.output_dir / "artifacts" / f"{dataset_name}_seed{seed}.pt"
    save_run_artifact(
        artifact_path,
        model,
        signals,
        weights,
        train_losses,
        dataset_name,
        seed,
    )

    correlation_rows, probe_rows = individual_damage_probe(
        model,
        weights,
        signals,
        held_inputs,
        held_targets,
        args.probe_weights,
        seed,
        args.evaluation_batch_size,
    )
    for row in correlation_rows:
        row.update(
            {
                "dataset": dataset_name,
                "seed": seed,
                "probe_split": "test",
                "probe_weights": len(probe_rows),
            }
        )
    write_csv(
        args.output_dir
        / "individual_damage"
        / f"{dataset_name}_seed{seed}_test.csv",
        probe_rows,
    )

    if args.probe_train:
        train_inputs, train_targets = collect_held_out(
            train_set, args.held_out, seed + 40_000, device
        )
        train_correlations, train_probe_rows = individual_damage_probe(
            model,
            weights,
            signals,
            train_inputs,
            train_targets,
            args.probe_weights,
            seed,
            args.evaluation_batch_size,
        )
        for row in train_correlations:
            row.update(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "probe_split": "train",
                    "probe_weights": len(train_probe_rows),
                }
            )
        correlation_rows.extend(train_correlations)
        write_csv(
            args.output_dir
            / "individual_damage"
            / f"{dataset_name}_seed{seed}_train.csv",
            train_probe_rows,
        )

    group_rows = group_damage_probe(
        model,
        weights,
        signals,
        held_inputs,
        held_targets,
        seed,
        args.group_trials,
        args.group_noise_fraction,
        args.evaluation_batch_size,
    )
    for row in group_rows:
        row.update({"dataset": dataset_name, "seed": seed})
    return correlation_rows, group_rows, sensor_row


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("fashion_mnist", "mnist"),
        default=("fashion_mnist", "mnist"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--fisher-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=1_000)
    parser.add_argument("--held-out", type=int, default=10_000)
    parser.add_argument("--probe-weights", type=int, default=2_000)
    parser.add_argument("--group-trials", type=int, default=50)
    parser.add_argument(
        "--group-noise-fraction",
        type=float,
        default=0.1,
        help="Gaussian noise std as a fraction of each layer's weight std.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--probe-train",
        action="store_true",
        help="Also run individual damage probes on a training-image slice.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries and the report from existing per-seed CSVs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.probe_weights <= 0 or args.held_out <= 0:
        parser.error("epochs, probe-weights, and held-out must be positive")
    if args.group_trials <= 0 or args.group_noise_fraction <= 0.0:
        parser.error("group-trials and group-noise-fraction must be positive")
    return args


def main() -> None:
    """Run E1 and write raw results, summaries, curves, and the report."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Running E1 on {device} with seeds {tuple(args.seeds)}")

    if args.summarize_only:
        correlation_rows = read_csv(args.output_dir / "correlations_by_seed.csv")
        group_rows = read_csv(args.output_dir / "group_damage_by_seed.csv")
        sensor_rows = read_csv(args.output_dir / "sensor_agreement_by_seed.csv")
        print("Loaded existing per-seed results; training and probes were skipped.")
    else:
        correlation_rows = []
        group_rows = []
        sensor_rows = []
        for dataset_name in args.datasets:
            train_set, test_set = load_dataset(dataset_name, args.data_dir)
            for seed in args.seeds:
                run_correlations, run_groups, sensor_row = run_one(
                    dataset_name,
                    seed,
                    train_set,
                    test_set,
                    args,
                    device,
                )
                correlation_rows.extend(run_correlations)
                group_rows.extend(run_groups)
                sensor_rows.append(sensor_row)
                write_csv(
                    args.output_dir / "correlations_by_seed.csv", correlation_rows
                )
                write_csv(
                    args.output_dir / "group_damage_by_seed.csv", group_rows
                )
                write_csv(
                    args.output_dir / "sensor_agreement_by_seed.csv", sensor_rows
                )

    correlation_summary = summarize_correlations(correlation_rows)
    group_summary = summarize_groups(group_rows)
    write_csv(args.output_dir / "correlation_summary.csv", correlation_summary)
    write_csv(args.output_dir / "group_damage_summary.csv", group_summary)
    plot_group_curves(group_summary, args.output_dir)

    report = build_report(
        correlation_summary,
        group_summary,
        correlation_rows,
        group_rows,
        sensor_rows,
        args.seeds,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    metadata = {
        "seeds": list(args.seeds),
        "datasets": list(args.datasets),
        "device": str(device),
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
