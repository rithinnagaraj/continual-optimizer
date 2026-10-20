"""E2: compare box and spring protection at identical importance.

This file implements Sections 1 and 2 of
``e2-e4-e5-e6-implementation-spec.md``.  For every seed it trains one
passive CAPAdam checkpoint for exactly 5,000 Task-A steps.  Every Task-B
arm and k value restores that full model/optimizer checkpoint, freezes the
same anchor, importance, and globally calibrated c, then runs exactly 5,000
steps with CAPAdam's box or proximal normalized spring protection.

Default execution is the preregistered experiment: five paired seeds, separate nine-value
k grids, and 90 Task-B runs.  ``--resume`` safely skips completed
Task-B runs; ``--reuse-checkpoints`` skips completed Task-A runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset
from paper_run_integrity import atomic_configuration, lock_e2_configuration

from experiment_e3 import (
    PermutedDataset,
    evaluate,
    fixed_permutation,
    load_mnist,
    make_loader,
    repeat_batches,
    cap_parameter_groups,
    seed_everything,
)
from model import MLP300
from cap_adam_experimental import CAPAdam


SEEDS = (42, 123, 2026, 451, 789)
ARMS = ("box", "spring")
BOX_K_VALUES = (1e-6, 3e-6, 7e-6, 1.5e-5, 3e-5, 7e-5, 1e-4, 2e-4, 2.0)
# CAPAdam's spring force scales as 1/k^2, so this broader grid spans strong
# mean reversion through an effectively weak spring over the 5,000-step phase.
SPRING_K_VALUES = (7e-4, 1e-3, 3e-3, 6e-3, 1e-2, 2e-2, 4e-2, 8e-2, 2.0)
RETENTION_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128
T_S = 2_000.0
T_A = 2_000.0
T_M = 10_000.0
DEFAULT_K = 2.0
DEFAULT_C = 1e-3

PHASE_A_STEPS = 5_000
PHASE_B_STEPS = 5_000
C_REFRESH_STEPS = 100
EVALUATE_EVERY = 10
EVALUATION_SUBSET_SIZE = 2_000
EVALUATION_BATCH_SIZE = 1_000
A_TEST_SUBSET_SEED = 777
B_TEST_SUBSET_SEED = 778
PERMUTATION_SEED = 2026
FRONTIER_EPSILON = 0.01

TRAJECTORY_FIELDS = (
    "arm",
    "k",
    "seed",
    "b_step",
    "a_test_accuracy",
    "b_test_accuracy",
    "bind_rate",
)
RESULT_FIELDS = (
    "arm",
    "k",
    "seed",
    "retention",
    "plasticity",
    "min_a",
    "steps_to_b90",
)
POINT_FIELDS = (
    "arm",
    "k",
    "seeds",
    "retention_mean",
    "retention_std",
    "plasticity_mean",
    "plasticity_std",
    "min_a_mean",
    "min_a_std",
    "steps_to_b90_mean",
    "steps_to_b90_std",
)
FRONTIER_FIELDS = (
    "retention_level",
    "box_reachable",
    "box_k",
    "box_plasticity",
    "box_steps_to_b90",
    "box_min_a",
    "spring_reachable",
    "spring_k",
    "spring_plasticity",
    "spring_steps_to_b90",
    "spring_min_a",
    "box_minus_spring",
)


def fixed_test_subset(dataset: Dataset, seed: int) -> Subset:
    """Select the fixed 2,000-example test subset required by the spec."""
    if len(dataset) < EVALUATION_SUBSET_SIZE:
        raise ValueError("test dataset is smaller than the required subset")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        :EVALUATION_SUBSET_SIZE
    ].tolist()
    return Subset(dataset, indices)


def create_cap_optimizer(model: nn.Module) -> CAPAdam:
    """Construct CAPAdam with all shared-convention defaults."""
    return CAPAdam(
        cap_parameter_groups(model, WEIGHT_DECAY),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        k=DEFAULT_K,
        T_s=T_S,
        T_a=T_A,
        T_m=T_M,
        c=DEFAULT_C,
        passive=True,
        protection="box",
        retention="ratchet",
        freeze_stats=False,
    )


def protected_importance(optimizer: CAPAdam) -> Tensor:
    """Concatenate importance values from protected parameter groups."""
    values: List[Tensor] = []
    for group in optimizer.param_groups:
        if not bool(group["protect"]):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "imp" not in state:
                raise RuntimeError("CAP importance state is not initialized")
            values.append(state["imp"].detach().reshape(-1))
    if not values:
        raise RuntimeError("optimizer contains no protected parameters")
    return torch.cat(values)


def importance_quantiles(values: Tensor) -> dict:
    """Return the mandatory median, p99, and max importance hardening."""
    flattened = values.detach().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("cannot summarize an empty importance tensor")
    return {
        "median": float(flattened.median().item()),
        "p99": float(torch.quantile(flattened.float(), 0.99).item()),
        "max": float(flattened.max().item()),
    }


def protected_importance_quantiles(optimizer: CAPAdam) -> dict:
    """Return report quantiles over every protected CAP weight."""
    return importance_quantiles(protected_importance(optimizer))


def prefix_preliminary_verdict(
    verdict: str,
    seeds: Sequence[int],
    full_seed_set: Sequence[int] = SEEDS,
) -> str:
    """Apply the mandatory prefix unless the exact full seed set ran."""
    return (
        verdict
        if len(seeds) == len(full_seed_set) and set(seeds) == set(full_seed_set)
        else f"PRELIMINARY-{verdict}"
    )


@torch.no_grad()
def global_importance_median(optimizer: CAPAdam) -> float:
    """Return the global median importance over protected weights."""
    return float(protected_importance(optimizer).median().item())


def set_all_group_c(optimizer: CAPAdam, value: float) -> None:
    """Assign one positive c to every parameter group."""
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"c must be finite and positive, got {value}")
    for group in optimizer.param_groups:
        group["c"] = value


def refresh_global_c(optimizer: CAPAdam) -> Optional[float]:
    """Apply shared-convention c calibration, skipping zero burn-in."""
    value = global_importance_median(optimizer)
    if value > 0.0:
        set_all_group_c(optimizer, value)
        return value
    return None


def checkpoint_constants(seed: int) -> dict:
    """Return the scientific settings that determine a Phase-A checkpoint."""
    return {
        "experiment": "E2",
        "format_version": 2,
        "seed": seed,
        "phase_a_steps": PHASE_A_STEPS,
        "lr": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "T_s": T_S,
        "T_a": T_A,
        "T_m": T_M,
        "passive": True,
        "retention": "ratchet",
        "c_refresh_steps": C_REFRESH_STEPS,
    }


def train_phase_a(
    train_set: Dataset,
    seed: int,
    device: torch.device,
    num_workers: int,
    checkpoint_path: Path,
) -> dict:
    """Train passive CAPAdam on identity MNIST for exactly 5,000 steps."""
    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer = create_cap_optimizer(model)
    train_loader = make_loader(
        train_set,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    batches = repeat_batches(train_loader)
    model.train()

    for step in range(1, PHASE_A_STEPS + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()

        if step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        if step % 500 == 0 or step == PHASE_A_STEPS:
            print(
                f"  A seed={seed} step={step:04d}/{PHASE_A_STEPS} "
                f"loss={float(loss.item()):.5f}"
            )

    final_c = global_importance_median(optimizer)
    if final_c <= 0.0:
        raise RuntimeError("global protected-weight importance median is zero")
    set_all_group_c(optimizer, final_c)

    checkpoint = {
        "constants": checkpoint_constants(seed),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_c": final_c,
        "imp_quantiles": protected_importance_quantiles(optimizer),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def load_checkpoint(path: Path, seed: int) -> dict:
    """Load and strictly validate an E2 Phase-A checkpoint."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    expected = checkpoint_constants(seed)
    actual = checkpoint.get("constants")
    if actual != expected:
        raise ValueError(
            f"checkpoint {path} is incompatible with the fixed E2 protocol; "
            f"expected constants {expected!r}, got {actual!r}"
        )
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint {path} is missing full state dictionaries")
    return checkpoint


def checkpoint_importance_quantiles(checkpoint: Mapping) -> dict:
    """Read or reconstruct the mandatory Phase-A importance quantiles."""
    if "imp_quantiles" in checkpoint:
        return dict(checkpoint["imp_quantiles"])
    model = MLP300(input_units=784, output_units=10)
    optimizer = create_cap_optimizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    return protected_importance_quantiles(optimizer)


def load_phase_b_branch(
    checkpoint: Mapping,
    arm: str,
    k: float,
    k_values_by_arm: Mapping[str, Sequence[float]],
    device: torch.device,
) -> tuple[MLP300, CAPAdam, float]:
    """Restore the iso-checkpoint and apply the exact E2 task switch."""
    if arm not in ARMS:
        raise ValueError(f"unknown E2 arm: {arm}")
    if k not in k_values_by_arm[arm]:
        raise ValueError(f"k={k} is not in the configured {arm} grid")

    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer = create_cap_optimizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Deep-copy optimizer tensors before loading.  Without this, a same-device
    # load can share storage with the in-memory checkpoint and the first branch
    # would silently mutate the starting state used by later branches.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))

    frozen_c = global_importance_median(optimizer)
    if frozen_c <= 0.0:
        raise RuntimeError("cannot freeze c from a zero importance median")
    for group in optimizer.param_groups:
        group["passive"] = False
        group["freeze_stats"] = True
        group["protection"] = arm
        group["spring_step"] = "proximal"
        group["retention"] = "ratchet"
        group["k"] = k
        group["c"] = frozen_c
    return model, optimizer, frozen_c


def protected_bind_rate(optimizer: CAPAdam) -> float:
    """Aggregate this step's bind diagnostics over protected groups only."""
    bind_count = sum(
        int(group["last_bind_count"])
        for group in optimizer.param_groups
        if bool(group["protect"])
    )
    bind_numel = sum(
        int(group["last_bind_numel"])
        for group in optimizer.param_groups
        if bool(group["protect"])
    )
    if bind_numel == 0:
        raise RuntimeError("protected bind diagnostics have a zero denominator")
    return bind_count / bind_numel


def run_phase_b(
    checkpoint: Mapping,
    arm: str,
    k: float,
    k_values_by_arm: Mapping[str, Sequence[float]],
    seed: int,
    task_b_train: Dataset,
    a_test_loader,
    b_test_loader,
    device: torch.device,
    num_workers: int,
) -> tuple[dict, List[dict]]:
    """Run one exact 5,000-step B branch and return metrics and trajectory."""
    seed_everything(seed)
    model, optimizer, _ = load_phase_b_branch(
        checkpoint, arm, k, k_values_by_arm, device
    )
    train_loader = make_loader(
        task_b_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    batches = repeat_batches(train_loader)
    trajectory: List[dict] = []
    model.train()

    for b_step in range(1, PHASE_B_STEPS + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            b_step % EVALUATE_EVERY == 0 or b_step == PHASE_B_STEPS
        )
        if should_evaluate:
            bind_rate = protected_bind_rate(optimizer)
            _, a_accuracy = evaluate(model, a_test_loader, device)
            _, b_accuracy = evaluate(model, b_test_loader, device)
            model.train()
            trajectory.append(
                {
                    "arm": arm,
                    "k": k,
                    "seed": seed,
                    "b_step": b_step,
                    "a_test_accuracy": a_accuracy,
                    "b_test_accuracy": b_accuracy,
                    "bind_rate": bind_rate,
                }
            )
        if b_step % 500 == 0 or b_step == PHASE_B_STEPS:
            latest = trajectory[-1]
            print(
                f"    {arm.upper():6s} k={k:g} seed={seed} "
                f"step={b_step:04d}/{PHASE_B_STEPS} "
                f"A={float(latest['a_test_accuracy']):.4f} "
                f"B={float(latest['b_test_accuracy']):.4f}"
            )

    final = trajectory[-1]
    final_b = float(final["b_test_accuracy"])
    b90 = 0.9 * final_b
    steps_to_b90 = next(
        int(row["b_step"])
        for row in trajectory
        if float(row["b_test_accuracy"]) >= b90
    )
    result = {
        "arm": arm,
        "k": k,
        "seed": seed,
        "retention": float(final["a_test_accuracy"]),
        "plasticity": final_b,
        "min_a": min(float(row["a_test_accuracy"]) for row in trajectory),
        "steps_to_b90": steps_to_b90,
    }
    return result, trajectory


def write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    """Atomically replace a CSV using an explicit protocol schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[dict]:
    """Read a required result CSV."""
    if not path.exists():
        raise FileNotFoundError(f"required E2 result does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    collected = list(values)
    if not collected:
        raise ValueError("cannot summarize an empty collection")
    deviation = stdev(collected) if len(collected) > 1 else 0.0
    return mean(collected), deviation


def validate_result_coverage(
    rows: Sequence[Mapping],
    seeds: Sequence[int],
    k_values_by_arm: Mapping[str, Sequence[float]],
) -> None:
    """Require exactly one completed result for every arm/k/seed cell."""
    expected = {
        (arm, k, seed)
        for arm in ARMS
        for k in k_values_by_arm[arm]
        for seed in seeds
    }
    observed = [
        (str(row["arm"]), float(row["k"]), int(float(row["seed"])))
        for row in rows
    ]
    duplicates = {key for key in observed if observed.count(key) > 1}
    missing = expected - set(observed)
    unexpected = set(observed) - expected
    if duplicates or missing or unexpected:
        raise ValueError(
            "E2 result coverage is invalid: "
            f"duplicates={sorted(duplicates)}, missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def summarize_points(
    rows: Sequence[Mapping],
    k_values_by_arm: Mapping[str, Sequence[float]],
) -> List[dict]:
    """Compute per-(arm,k) seed means used by the frontier rule."""
    grouped: MutableMapping[tuple[str, float], List[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), float(row["k"]))].append(row)

    summary: List[dict] = []
    for arm in ARMS:
        for k in k_values_by_arm[arm]:
            group = grouped[(arm, k)]
            retention_mean, retention_std = mean_std(
                float(row["retention"]) for row in group
            )
            plasticity_mean, plasticity_std = mean_std(
                float(row["plasticity"]) for row in group
            )
            min_a_mean, min_a_std = mean_std(float(row["min_a"]) for row in group)
            speed_mean, speed_std = mean_std(
                float(row["steps_to_b90"]) for row in group
            )
            summary.append(
                {
                    "arm": arm,
                    "k": k,
                    "seeds": len(group),
                    "retention_mean": retention_mean,
                    "retention_std": retention_std,
                    "plasticity_mean": plasticity_mean,
                    "plasticity_std": plasticity_std,
                    "min_a_mean": min_a_mean,
                    "min_a_std": min_a_std,
                    "steps_to_b90_mean": speed_mean,
                    "steps_to_b90_std": speed_std,
                }
            )
    return summary


def frontier_choice(
    points: Sequence[Mapping], arm: str, retention_level: float
) -> Optional[Mapping]:
    """Choose the best-plasticity k satisfying one retention threshold."""
    feasible = [
        point
        for point in points
        if point["arm"] == arm
        and float(point["retention_mean"]) >= retention_level
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda point: _frontier_sort_key(point),
    )


def _frontier_sort_key(point: Mapping) -> tuple[float, float, float]:
    """Prefer plasticity, then retention, then the tighter k on exact ties."""
    return (
        float(point["plasticity_mean"]),
        float(point["retention_mean"]),
        -float(point["k"]),
    )


def build_frontier_rows(points: Sequence[Mapping]) -> List[dict]:
    """Evaluate both arms at the six preregistered retention levels."""
    rows: List[dict] = []
    for retention_level in RETENTION_LEVELS:
        box = frontier_choice(points, "box", retention_level)
        spring = frontier_choice(points, "spring", retention_level)
        row = {
            "retention_level": retention_level,
            "box_reachable": box is not None,
            "box_k": "" if box is None else box["k"],
            "box_plasticity": "" if box is None else box["plasticity_mean"],
            "box_steps_to_b90": "" if box is None else box["steps_to_b90_mean"],
            "box_min_a": "" if box is None else box["min_a_mean"],
            "spring_reachable": spring is not None,
            "spring_k": "" if spring is None else spring["k"],
            "spring_plasticity": (
                "" if spring is None else spring["plasticity_mean"]
            ),
            "spring_steps_to_b90": (
                "" if spring is None else spring["steps_to_b90_mean"]
            ),
            "spring_min_a": "" if spring is None else spring["min_a_mean"],
            "box_minus_spring": (
                ""
                if box is None or spring is None
                else float(box["plasticity_mean"])
                - float(spring["plasticity_mean"])
            ),
        }
        rows.append(row)
    return rows


def classify_frontiers(frontier_rows: Sequence[Mapping]) -> dict:
    """Apply the exact epsilon-tolerant dominance classification."""
    comparable = [
        row
        for row in frontier_rows
        if bool(row["box_reachable"]) and bool(row["spring_reachable"])
    ]
    differences = [float(row["box_minus_spring"]) for row in comparable]
    if not differences:
        classification = "mixed"
        box_not_worse = False
        spring_not_worse = False
        equivalent = False
        box_strict_wins = 0
        spring_strict_wins = 0
    else:
        box_not_worse = all(value >= -FRONTIER_EPSILON for value in differences)
        spring_not_worse = all(value <= FRONTIER_EPSILON for value in differences)
        equivalent = all(abs(value) <= FRONTIER_EPSILON for value in differences)
        box_strict_wins = sum(value > FRONTIER_EPSILON for value in differences)
        spring_strict_wins = sum(value < -FRONTIER_EPSILON for value in differences)
        if box_not_worse and box_strict_wins >= 2:
            classification = "box dominates"
        elif equivalent:
            classification = "equivalent"
        elif spring_not_worse and spring_strict_wins >= 2:
            classification = "spring dominates"
        else:
            classification = "mixed"
    return {
        "classification": classification,
        "comparable_levels": len(comparable),
        "box_not_worse": box_not_worse,
        "box_strict_wins": box_strict_wins,
        "spring_not_worse": spring_not_worse,
        "spring_strict_wins": spring_strict_wins,
        "equivalent": equivalent,
        "max_absolute_difference": max(map(abs, differences), default=float("nan")),
    }


def grade_experiment(
    points: Sequence[Mapping],
    frontier_rows: Sequence[Mapping],
    k_values_by_arm: Mapping[str, Sequence[float]],
) -> dict:
    """Apply the largest-k control before assigning the treatment verdict."""
    largest = {
        str(point["arm"]): point
        for point in points
        if float(point["k"]) == max(k_values_by_arm[str(point["arm"])])
    }
    box_plasticity = float(largest["box"]["plasticity_mean"])
    spring_plasticity = float(largest["spring"]["plasticity_mean"])
    control_difference = abs(box_plasticity - spring_plasticity)
    control_passes = control_difference <= FRONTIER_EPSILON
    analysis = classify_frontiers(frontier_rows)

    if not control_passes:
        verdict = "HARNESS-INVALID"
    elif analysis["classification"] in ("box dominates", "equivalent"):
        verdict = "PASS"
    elif analysis["classification"] == "spring dominates":
        verdict = "PROTECTION-FORM-PIVOT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        **analysis,
        "control_box_k": max(k_values_by_arm["box"]),
        "control_spring_k": max(k_values_by_arm["spring"]),
        "control_box_plasticity": box_plasticity,
        "control_spring_plasticity": spring_plasticity,
        "control_absolute_difference": control_difference,
        "control_passes": control_passes,
        "verdict": verdict,
    }


def plot_frontiers(points: Sequence[Mapping], output_dir: Path) -> None:
    """Save the mean retention-plasticity sweep and seed variability."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    styles = {"box": ("tab:blue", "o"), "spring": ("tab:orange", "s")}
    for arm in ARMS:
        rows = [point for point in points if point["arm"] == arm]
        color, marker = styles[arm]
        axis.errorbar(
            [100.0 * float(row["retention_mean"]) for row in rows],
            [100.0 * float(row["plasticity_mean"]) for row in rows],
            xerr=[100.0 * float(row["retention_std"]) for row in rows],
            yerr=[100.0 * float(row["plasticity_std"]) for row in rows],
            color=color,
            marker=marker,
            linestyle="-",
            capsize=3,
            label=arm.upper(),
        )
        for row in rows:
            axis.annotate(
                f"k={float(row['k']):g}",
                (
                    100.0 * float(row["retention_mean"]),
                    100.0 * float(row["plasticity_mean"]),
                ),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
    axis.set_xlabel("Task-A final accuracy / retention (%)")
    axis.set_ylabel("Task-B final accuracy / plasticity (%)")
    axis.set_title("E2: BOX vs SPRING at identical importance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "e2_frontiers.png", dpi=200)
    plt.close(figure)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_report(
    results: Sequence[Mapping],
    points: Sequence[Mapping],
    frontier_rows: Sequence[Mapping],
    grade: Mapping,
    seeds: Sequence[int],
    k_values_by_arm: Mapping[str, Sequence[float]],
    imp_quantiles: Optional[Mapping[str, float]] = None,
) -> str:
    """Build the required settings, criteria, measurements, and verdict report."""
    arm_min_a = {
        arm: mean(float(row["min_a"]) for row in results if row["arm"] == arm)
        for arm in ARMS
    }
    complete_seed_set = tuple(seeds) == SEEDS
    report_verdict = prefix_preliminary_verdict(str(grade["verdict"]), seeds)
    quantile_line = (
        "- Protected-weight imp quantiles at the Phase-A boundary: "
        f"median={float(imp_quantiles['median']):.6g}; "
        f"p99={float(imp_quantiles['p99']):.6g}; "
        f"max={float(imp_quantiles['max']):.6g}."
        if imp_quantiles is not None
        else "- Protected-weight imp quantiles at the Phase-A boundary: unavailable."
    )
    lines = [
        "# E2 report — BOX vs SPRING at identical importance",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        f"Preregistered five-seed set complete: {complete_seed_set}",
        "",
        "## Optimizer and protocol settings",
        "",
        f"- Model: MLP300(input_units=784, output_units=10)",
        f"- Optimizer: CAPAdam; lr={LEARNING_RATE}; betas={BETAS}; "
        f"eps={ADAM_EPS}; wd={WEIGHT_DECAY}; batch={BATCH_SIZE}; "
        "amsgrad=False; maximize=False",
        f"- T_s={T_S:g}; T_a={T_A:g}; T_m={T_M:g}; "
        "decay_mode=blend; track_bind_stats=True",
        f"- Defaults: protection=box; k={DEFAULT_K:g}; c={DEFAULT_C:g} "
        "before calibration",
        f"- Phase A: passive=True; retention=ratchet; steps={PHASE_A_STEPS}",
        f"- Phase B: freeze_stats=True; steps={PHASE_B_STEPS}; "
        f"evaluation every {EVALUATE_EVERY} steps",
        f"- BOX k grid: {tuple(k_values_by_arm['box'])}",
        f"- SPRING k grid: {tuple(k_values_by_arm['spring'])}",
        "- c: global median importance over protected weights at the switch, "
        "then frozen for all groups",
        f"- Test subsets: {EVALUATION_SUBSET_SIZE} examples; "
        f"A seed={A_TEST_SUBSET_SEED}; B seed={B_TEST_SUBSET_SEED}",
        f"- Task-B pixel permutation seed: {PERMUTATION_SEED}",
        quantile_line,
        "",
        "## Criteria and measured values",
        "",
        (
            f"- Control at largest k per arm (BOX "
            f"k={float(grade['control_box_k']):g}; SPRING "
            f"k={float(grade['control_spring_k']):g}): "
            f"|BOX plasticity âˆ’ SPRING plasticity| = "
            f"{float(grade['control_absolute_difference']):.5f}; threshold ≤ "
            f"{FRONTIER_EPSILON:.2f}; passes={grade['control_passes']}."
        ),
        (
            f"- BOX-dominates criterion: BOX ≥ SPRING âˆ’ {FRONTIER_EPSILON:.2f} "
            f"at every comparable level, measured={grade['box_not_worse']}; "
            f"BOX > SPRING + {FRONTIER_EPSILON:.2f} at ≥2 levels, "
            f"measured={grade['box_strict_wins']}."
        ),
        (
            f"- Equivalent criterion: |BOX âˆ’ SPRING| ≤ "
            f"{FRONTIER_EPSILON:.2f} at every comparable level; "
            f"measured={grade['equivalent']}; maximum measured absolute "
            f"difference={float(grade['max_absolute_difference']):.5f}."
        ),
        (
            f"- SPRING-dominates criterion: SPRING ≥ BOX âˆ’ "
            f"{FRONTIER_EPSILON:.2f} at every comparable level, "
            f"measured={grade['spring_not_worse']}; SPRING > BOX + "
            f"{FRONTIER_EPSILON:.2f} at ≥2 levels, "
            f"measured={grade['spring_strict_wins']}."
        ),
        f"- Comparable retention levels measured: {grade['comparable_levels']}/6.",
        f"- Frontier classification: **{grade['classification']}**.",
        f"- Final verdict: **{report_verdict}**.",
        "",
        "## Mean sweep points",
        "",
        "| Arm | k | Retention | Plasticity | min_a | steps_to_b90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        lines.append(
            f"| {str(point['arm']).upper()} | {float(point['k']):g} | "
            f"{percent(float(point['retention_mean']))} ± "
            f"{percent(float(point['retention_std']))} | "
            f"{percent(float(point['plasticity_mean']))} ± "
            f"{percent(float(point['plasticity_std']))} | "
            f"{percent(float(point['min_a_mean']))} | "
            f"{float(point['steps_to_b90_mean']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Matched-retention frontier and secondary measurements",
            "",
            "| R | BOX k | BOX B | BOX speed | BOX min_a | SPRING k | "
            "SPRING B | SPRING speed | SPRING min_a | BOXâˆ’SPRING |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frontier_rows:
        if bool(row["box_reachable"]):
            box_values = (
                f"{float(row['box_k']):g}",
                percent(float(row["box_plasticity"])),
                f"{float(row['box_steps_to_b90']):.1f}",
                percent(float(row["box_min_a"])),
            )
        else:
            box_values = ("unreachable",) * 4
        if bool(row["spring_reachable"]):
            spring_values = (
                f"{float(row['spring_k']):g}",
                percent(float(row["spring_plasticity"])),
                f"{float(row['spring_steps_to_b90']):.1f}",
                percent(float(row["spring_min_a"])),
            )
        else:
            spring_values = ("unreachable",) * 4
        difference = (
            "n/a"
            if row["box_minus_spring"] == ""
            else percent(float(row["box_minus_spring"]))
        )
        lines.append(
            f"| {float(row['retention_level']):.2f} | "
            f"{' | '.join(box_values)} | {' | '.join(spring_values)} | "
            f"{difference} |"
        )

    lines.extend(
        [
            "",
            f"Mean min_a over every BOX run: {percent(arm_min_a['box'])}.",
            f"Mean min_a over every SPRING run: {percent(arm_min_a['spring'])}.",
            "",
            "![E2 frontiers](e2_frontiers.png)",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse operational options; scientific settings remain fixed constants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e2"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--box-k-values",
        nargs="+",
        type=float,
        default=BOX_K_VALUES,
        help="Manual BOX k grid.",
    )
    parser.add_argument(
        "--spring-k-values",
        nargs="+",
        type=float,
        default=SPRING_K_VALUES,
        help="Manual SPRING k grid.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse compatible results/e2/checkpoints/seed{seed}.pt files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checkpoints and skip completed arm/k/seed result cells.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries/report from complete existing E2 CSVs.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not args.seeds:
        parser.error("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    args.box_k_values = tuple(args.box_k_values)
    args.spring_k_values = tuple(args.spring_k_values)
    for name, values in (
        ("--box-k-values", args.box_k_values),
        ("--spring-k-values", args.spring_k_values),
    ):
        if any(value <= 0.0 or not math.isfinite(value) for value in values):
            parser.error(f"{name} values must be finite and positive")
        if len(set(values)) != len(values):
            parser.error(f"{name} values must be unique")
    return args


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_config(
    args: argparse.Namespace,
    seeds: Sequence[int],
    grade: Mapping,
    k_values_by_arm: Mapping[str, Sequence[float]],
) -> dict:
    """Return full operational and fixed scientific configuration metadata."""
    return {
        "experiment": "E2",
        "implementation_spec": "e2-e4-e5-e6-implementation-spec.md sections 1-2",
        "arguments": serializable_args(args),
        "seeds": list(seeds),
        "scientific_settings": {
            "arms": list(ARMS),
            "box_k_values": list(k_values_by_arm["box"]),
            "spring_k_values": list(k_values_by_arm["spring"]),
            "spring_step": "proximal",
            "retention_levels": list(RETENTION_LEVELS),
            "frontier_epsilon": FRONTIER_EPSILON,
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "protection_default": "box",
            "default_k": DEFAULT_K,
            "T_s": T_S,
            "T_a": T_A,
            "T_m": T_M,
            "decay_mode": "blend",
            "track_bind_stats": True,
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "c_refresh_steps": C_REFRESH_STEPS,
            "evaluate_every": EVALUATE_EVERY,
            "evaluation_subset_size": EVALUATION_SUBSET_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "a_test_subset_seed": A_TEST_SUBSET_SEED,
            "b_test_subset_seed": B_TEST_SUBSET_SEED,
            "permutation_seed": PERMUTATION_SEED,
            "source_hashes": {
                name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                for name in ("experiment_e2.py", "cap_adam_experimental.py", "experiment_e3.py", "model.py", "paper_run_integrity.py")
            },
        },
        "grade": dict(grade),
    }


def main() -> None:
    """Run the fixed E2 protocol and write raw and summarized artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    trajectory_path = args.output_dir / "e2_trajectories.csv"
    result_path = args.output_dir / "e2_results.csv"
    k_values_by_arm = {
        "box": args.box_k_values,
        "spring": args.spring_k_values,
    }
    config_path = args.output_dir / "run_config.json"
    initial_config = run_config(args, args.seeds, {}, k_values_by_arm)
    initial_config["status"] = "running"
    lock_e2_configuration(
        config_path, initial_config,
        require_existing=(args.resume or args.summarize_only)
        and (result_path.exists() or trajectory_path.exists()),
    )
    device = torch.device(args.device)
    print(f"Running fixed E2 on {device}; seeds={tuple(args.seeds)}")

    if args.summarize_only:
        results = read_csv(result_path)
        phase_a_quantile_rows = []
        for seed in args.seeds:
            checkpoint_path = checkpoint_dir / f"seed{seed}.pt"
            if checkpoint_path.exists():
                phase_a_quantile_rows.append(
                    checkpoint_importance_quantiles(
                        load_checkpoint(checkpoint_path, seed)
                    )
                )
    else:
        mnist_train, mnist_test = load_mnist(args.data_dir)
        permutation = fixed_permutation(784, PERMUTATION_SEED)
        task_b_train = PermutedDataset(mnist_train, permutation)
        a_test_set = fixed_test_subset(mnist_test, A_TEST_SUBSET_SEED)
        b_test_set = PermutedDataset(
            fixed_test_subset(mnist_test, B_TEST_SUBSET_SEED), permutation
        )
        a_test_loader = make_loader(
            a_test_set,
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )
        b_test_loader = make_loader(
            b_test_set,
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )

        if args.resume and result_path.exists() and trajectory_path.exists():
            results = read_csv(result_path)
            trajectories = read_csv(trajectory_path)
        else:
            results = []
            trajectories = []
        completed = {
            (str(row["arm"]), float(row["k"]), int(float(row["seed"])))
            for row in results
        }
        phase_a_quantile_rows = []

        for seed in args.seeds:
            checkpoint_path = checkpoint_dir / f"seed{seed}.pt"
            if (args.reuse_checkpoints or args.resume) and checkpoint_path.exists():
                print(f"\nLoading Phase-A checkpoint {checkpoint_path}")
                checkpoint = load_checkpoint(checkpoint_path, seed)
            else:
                print(f"\nTraining Phase A for seed={seed}")
                checkpoint = train_phase_a(
                    mnist_train,
                    seed,
                    device,
                    args.num_workers,
                    checkpoint_path,
                )
            phase_a_quantile_rows.append(
                checkpoint_importance_quantiles(checkpoint)
            )

            for arm in ARMS:
                for k in k_values_by_arm[arm]:
                    key = (arm, k, seed)
                    if key in completed:
                        print(f"  Skipping completed {arm} k={k:g} seed={seed}")
                        continue
                    print(f"  Starting {arm.upper()} k={k:g} seed={seed}")
                    result, trajectory = run_phase_b(
                        checkpoint,
                        arm,
                        k,
                        k_values_by_arm,
                        seed,
                        task_b_train,
                        a_test_loader,
                        b_test_loader,
                        device,
                        args.num_workers,
                    )
                    results.append(result)
                    trajectories.extend(trajectory)
                    completed.add(key)
                    write_csv(trajectory_path, trajectories, TRAJECTORY_FIELDS)
                    write_csv(result_path, results, RESULT_FIELDS)

    validate_result_coverage(results, args.seeds, k_values_by_arm)
    points = summarize_points(results, k_values_by_arm)
    frontier_rows = build_frontier_rows(points)
    grade = grade_experiment(points, frontier_rows, k_values_by_arm)
    write_csv(args.output_dir / "e2_points.csv", points, POINT_FIELDS)
    write_csv(
        args.output_dir / "e2_frontier_levels.csv",
        frontier_rows,
        FRONTIER_FIELDS,
    )
    plot_frontiers(points, args.output_dir)
    report = build_report(
        results,
        points,
        frontier_rows,
        grade,
        args.seeds,
        k_values_by_arm,
        (
            {
                key: mean(float(row[key]) for row in phase_a_quantile_rows)
                for key in ("median", "p99", "max")
            }
            if phase_a_quantile_rows
            else None
        ),
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    config = run_config(args, args.seeds, grade, k_values_by_arm)
    config["status"] = "complete"
    atomic_configuration(config_path, config)
    print("\n" + report)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
