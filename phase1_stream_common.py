"""Shared, checkpoint-cap mechanics for Phase-1 continual streams."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiment_e2 import protected_importance_quantiles
from experiment_e4a import FROZEN_RADIUS_THRESHOLD, protected_bind_counts
from cap_adam_experimental import CAPAdam, protection_radius


SCALAR_LOG_EVERY = 100
CHECKPOINT_EVERY = 50_000
STEPS_TO_THRESHOLD_WINDOW = 200
STEPS_TO_THRESHOLD = 0.85

SCALAR_FIELDS = (
    "arm",
    "seed",
    "step",
    "batch_loss",
    "batch_accuracy",
    "median_radius",
    "frozen_fraction",
    "bind_rate",
    "median_imp",
    "theta_norm",
)
TASK_FIELDS = (
    "arm",
    "seed",
    "task",
    "task_end_accuracy",
    "steps_to_085",
)
POOL_FIELDS = (
    "arm",
    "seed",
    "task",
    "pool_index",
    "accuracy",
    "is_current",
)

BLANK_CAP_STATISTICS = {
    "median_radius": "",
    "frozen_fraction": "",
    "bind_rate": "",
    "median_imp": "",
    "theta_norm": "",
}


def permute_batch(inputs: Tensor, permutation: Tensor) -> Tensor:
    batch_size = inputs.shape[0]
    return inputs.view(batch_size, 784)[:, permutation].view(
        batch_size, 1, 28, 28
    )


@torch.no_grad()
def evaluate_permutation(
    model: nn.Module,
    loader,
    permutation: Tensor,
    device: torch.device,
) -> float:
    """Evaluate one permutation and always restore training mode."""
    model.eval()
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(permute_batch(inputs, permutation))
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += targets.numel()
    if count == 0:
        raise ValueError("cannot evaluate an empty test subset")
    accuracy = correct / count
    model.train()
    return accuracy


@torch.no_grad()
def cap_stream_statistics(optimizer: CAPAdam) -> dict[str, float]:
    importance_values: list[Tensor] = []
    radius_values: list[Tensor] = []
    theta_values: list[Tensor] = []
    for group in optimizer.param_groups:
        if not bool(group["protect"]):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "imp" not in state:
                raise RuntimeError("CAP importance state is not initialized")
            importance = state["imp"].detach().reshape(-1)
            radius = protection_radius(state["imp"].detach(), group).reshape(-1)
            importance_values.append(importance)
            radius_values.append(radius)
            theta_values.append(parameter.detach().reshape(-1))
    if not importance_values:
        raise RuntimeError("optimizer contains no protected parameters")
    importance = torch.cat(importance_values)
    radius = torch.cat(radius_values)
    theta = torch.cat(theta_values)
    bind_count, bind_numel = protected_bind_counts(optimizer)
    return {
        "median_radius": float(radius.median().item()),
        "frozen_fraction": float(
            (radius < FROZEN_RADIUS_THRESHOLD).float().mean().item()
        ),
        "bind_rate": bind_count / max(bind_numel, 1),
        "median_imp": float(importance.median().item()),
        "theta_norm": float(torch.linalg.vector_norm(theta).item()),
    }


def training_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    hooks: Mapping[str, Any],
    inputs: Tensor,
    targets: Tensor,
) -> tuple[Tensor, float]:
    """Train on the mixed batch; measure accuracy on incoming current-task data."""
    mixer = hooks["batch_mixer"]
    incoming_accuracy = None
    if mixer is not None:
        with torch.no_grad():
            incoming_accuracy = float(
                (model(inputs).argmax(dim=1) == targets).float().mean().item()
            )
        inputs, targets = mixer(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    grad_hook = hooks["grad_hook"]
    if grad_hook is not None:
        grad_hook(model)
    optimizer.step()
    accuracy = incoming_accuracy if incoming_accuracy is not None else float(
        (logits.argmax(dim=1) == targets).float().mean().item()
    )
    return loss, accuracy


class StepsToThreshold:
    """Exact E4b rolling-window steps_to_085 state for one task."""

    def __init__(self) -> None:
        self.window: deque[float] = deque(maxlen=STEPS_TO_THRESHOLD_WINDOW)
        self.window_sum = 0.0
        self.reached: Optional[int] = None

    def update(self, accuracy: float, within_task_step: int) -> None:
        if len(self.window) == STEPS_TO_THRESHOLD_WINDOW:
            self.window_sum -= self.window[0]
        self.window.append(accuracy)
        self.window_sum += accuracy
        if (
            self.reached is None
            and within_task_step >= STEPS_TO_THRESHOLD_WINDOW
            and len(self.window) == STEPS_TO_THRESHOLD_WINDOW
            and self.window_sum / STEPS_TO_THRESHOLD_WINDOW
            >= STEPS_TO_THRESHOLD
        ):
            self.reached = within_task_step

    def reset(self) -> None:
        self.window.clear()
        self.window_sum = 0.0
        self.reached = None


def checkpoint_path(
    checkpoint_dir: Path, arm: str, seed: int, step: int
) -> Path:
    return checkpoint_dir / f"{arm}_seed{seed}_step{step}.pt"


def checkpoint_step_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_step", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed stream checkpoint filename: {path}") from error


def latest_checkpoint(
    checkpoint_dir: Path, arm: str, seed: int
) -> Optional[Path]:
    paths = list(checkpoint_dir.glob(f"{arm}_seed{seed}_step*.pt"))
    return max(paths, key=checkpoint_step_from_path) if paths else None


def load_torch_checkpoint(path: Path, map_location="cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    row_counts: Mapping[str, int],
    *,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "row_counts": {key: int(value) for key, value in row_counts.items()},
        "cpu_rng_state": torch.get_rng_state(),
    }
    if extra_payload:
        overlap = set(payload) & set(extra_payload)
        if overlap:
            raise ValueError(f"checkpoint extra payload replaces core keys: {overlap}")
        payload.update(extra_payload)
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    if "cpu_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def trim_rows_to_checkpoint(
    rows_by_name: Mapping[str, list[dict]], checkpoint: Mapping[str, Any]
) -> None:
    row_counts = checkpoint.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("stream checkpoint is missing row_counts")
    for name, rows in rows_by_name.items():
        count = int(row_counts[name])
        if count > len(rows):
            raise ValueError(
                f"checkpoint {name} count exceeds the current CSV rows"
            )
        del rows[count:]


def rows_for_cell(
    rows: Sequence[Mapping[str, Any]], arm: str, seed: int
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row["arm"]) == arm and int(float(row["seed"])) == seed
    ]


def remove_cell_rows(rows: list[dict], arm: str, seed: int) -> None:
    rows[:] = [
        row
        for row in rows
        if not (
            str(row["arm"]) == arm and int(float(row["seed"])) == seed
        )
    ]


def fast_forward_batches(batches, count: int) -> None:
    """Discard CPU batches without perturbing optimizer/global RNG state."""
    for skipped in range(count):
        next(batches)
        if (skipped + 1) % 100_000 == 0:
            print(f"  fast-forwarded {skipped + 1}/{count} CPU batches")


def final_quantile_record(
    arm: str,
    seed: int,
    optimizer: torch.optim.Optimizer,
    is_cap: bool,
) -> Optional[dict[str, Any]]:
    if not is_cap:
        return None
    return {
        "arm": arm,
        "seed": seed,
        **protected_importance_quantiles(optimizer),
    }


def validate_coverage(
    *,
    scalar_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    pool_rows: Optional[Sequence[Mapping[str, Any]]],
    arms: Sequence[str],
    seeds: Sequence[int],
    task_count: int,
    total_steps: int,
    pool_size: int = 0,
) -> None:
    expected_tasks = {
        (arm, int(seed), task)
        for arm in arms
        for seed in seeds
        for task in range(1, task_count + 1)
    }
    expected_scalars = {
        (arm, int(seed), step)
        for arm in arms
        for seed in seeds
        for step in range(SCALAR_LOG_EVERY, total_steps + 1, SCALAR_LOG_EVERY)
    }
    actual_tasks = {
        (str(row["arm"]), int(float(row["seed"])), int(float(row["task"])))
        for row in task_rows
    }
    actual_scalars = {
        (str(row["arm"]), int(float(row["seed"])), int(float(row["step"])))
        for row in scalar_rows
    }
    if len(task_rows) != len(expected_tasks) or actual_tasks != expected_tasks:
        raise RuntimeError(
            f"task CSV coverage is incomplete: {len(task_rows)} rows, "
            f"expected {len(expected_tasks)}"
        )
    if len(scalar_rows) != len(expected_scalars) or actual_scalars != expected_scalars:
        raise RuntimeError(
            f"scalar CSV coverage is incomplete: {len(scalar_rows)} rows, "
            f"expected {len(expected_scalars)}"
        )
    if pool_rows is not None:
        expected_pool = {
            (arm, int(seed), task, pool_index)
            for arm in arms
            for seed in seeds
            for task in range(1, task_count + 1)
            for pool_index in range(pool_size)
        }
        actual_pool = {
            (
                str(row["arm"]),
                int(float(row["seed"])),
                int(float(row["task"])),
                int(float(row["pool_index"])),
            )
            for row in pool_rows
        }
        if len(pool_rows) != len(expected_pool) or actual_pool != expected_pool:
            raise RuntimeError(
                f"pool CSV coverage is incomplete: {len(pool_rows)} rows, "
                f"expected {len(expected_pool)}"
            )


__all__ = [
    "BLANK_CAP_STATISTICS",
    "CHECKPOINT_EVERY",
    "POOL_FIELDS",
    "SCALAR_FIELDS",
    "SCALAR_LOG_EVERY",
    "STEPS_TO_THRESHOLD",
    "STEPS_TO_THRESHOLD_WINDOW",
    "StepsToThreshold",
    "TASK_FIELDS",
    "checkpoint_path",
    "evaluate_permutation",
    "fast_forward_batches",
    "final_quantile_record",
    "latest_checkpoint",
    "load_torch_checkpoint",
    "permute_batch",
    "remove_cell_rows",
    "restore_rng_state",
    "cap_stream_statistics",
    "save_checkpoint",
    "training_step",
    "trim_rows_to_checkpoint",
    "validate_coverage",
]
