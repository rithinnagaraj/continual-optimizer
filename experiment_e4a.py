"""E4a: long-silence memory test and burn-in retest.

This is the direct implementation of Section 1 of
``e4-implementation-procedure.md``.  The default invocation runs the full
five-seed, 120-cell matrix and writes the fixed E4a artifacts under
``results/e4a``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from experiment_e2 import (
    prefix_preliminary_verdict,
    protected_importance_quantiles,
    read_csv,
    write_csv,
)
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


FULL_SEEDS = (42, 123, 2026, 451, 789)
COMPUTE_FLOOR_SEEDS = (42, 123, 2026)
POLICIES = ("leak", "ratchet", "release")
CONSTRUCTIONS = ("synthetic", "realistic")
SILENCE_LENGTHS = (0, 1_000, 4_000, 20_000)

LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128
K = 3e-5
T_S = 2_000.0
T_A = 2_000.0
T_M_RELEASE = 10_000.0
PROTECTION = "box"
DECAY_MODE = "blend"
TRACK_BIND_STATS = True
C_REFRESH_STEPS = 100

PHASE_A_STEPS = 5_000
PHASE_B_STEPS = 3_000
EARLY_A_BIND_STEPS = 1_000
B_EVALUATE_EVERY = 50
EVALUATION_SUBSET_SIZE = 2_000
EVALUATION_BATCH_SIZE = 1_000
A_TEST_SUBSET_SEED = 777
B_PERMUTATION_SEED = 2026
CONFIG_ACCURACY_FLOOR = 0.95
GRADE_DELTA = 0.02
FROZEN_RADIUS_THRESHOLD = 1e-3

RESULT_FIELDS = (
    "policy",
    "construction",
    "L",
    "seed",
    "a_end",
    "a_after_silence",
    "a_final",
    "min_a_during_b",
    "early_a_bind_rate",
    "status",
)
MECHANISM_FIELDS = (
    "policy",
    "construction",
    "L",
    "seed",
    "checkpoint",
    "median_imp",
    "median_radius",
    "a_test_accuracy",
)
TRACE_FIELDS = (
    "policy",
    "construction",
    "L",
    "seed",
    "b_step",
    "a_test_accuracy",
)


class QuadrantMaskedDataset(Dataset):
    """Zero the normalized-space upper-left 14x14 quadrant of MNIST."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        mask = torch.ones((1, 28, 28), dtype=torch.float32)
        mask[:, :14, :14] = 0.0
        self.mask = mask

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        if tuple(image.shape) != (1, 28, 28):
            raise ValueError(
                "QuadrantMaskedDataset requires normalized MNIST tensors shaped "
                f"(1, 28, 28), got {tuple(image.shape)}"
            )
        return image * self.mask.to(dtype=image.dtype), int(target)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_optional_csv(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preconditions() -> None:
    """Run the mandatory Section-0 checks before constructing E4 runs."""
    parameters = inspect.signature(CAPAdam.__init__).parameters
    for required in ("decay_mode", "track_bind_stats", "freeze_stats"):
        if required not in parameters:
            raise RuntimeError(f"CAPAdam is missing required argument {required!r}")

    model = MLP300(input_units=784, output_units=10)
    optimizer = CAPAdam(
        cap_parameter_groups(model, WEIGHT_DECAY),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        k=K,
        T_s=T_S,
        T_a=T_A,
        T_m=T_M_RELEASE,
        passive=False,
        protection=PROTECTION,
        retention="release",
        decay_mode="zero",
        track_bind_stats=False,
    )
    for group in optimizer.param_groups:
        group["freeze_stats"] = False
        group["decay_mode"] = DECAY_MODE
        group["track_bind_stats"] = TRACK_BIND_STATS

    seed_everything(0)
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    check_optimizer = CAPAdam(
        [{"params": [parameter], "protect": True}],
        lr=0.0,
        weight_decay=0.0,
        T_s=10.0,
        retention="leak",
        passive=True,
    )
    for gradient in (0.5, -0.2):
        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        check_optimizer.step()
    observed = float(check_optimizer.state[parameter]["imp"].item())
    expected = 0.1 * 0.25 * 0.9 + 0.1 * 0.04
    if abs(observed - expected) >= 1e-9:
        raise RuntimeError(
            "CAPAdam failed the v2.3 raw-g^2 two-step check: "
            f"observed={observed}, expected={expected}"
        )
    for group in check_optimizer.param_groups:
        if "last_bind_count" not in group or "last_bind_numel" not in group:
            raise RuntimeError("CAPAdam bind counters are unavailable after step()")

    if not callable(protected_importance_quantiles):
        raise RuntimeError("the shared imp-quantile report hardening is unavailable")
    if not callable(prefix_preliminary_verdict):
        raise RuntimeError("the shared preliminary-verdict hardening is unavailable")


def fixed_test_subset(dataset: Dataset, seed: int) -> Subset:
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:EVALUATION_SUBSET_SIZE]
    return Subset(dataset, indices.tolist())


def create_optimizer(model: nn.Module, policy: str) -> CAPAdam:
    if policy not in POLICIES:
        raise ValueError(f"unknown E4a policy: {policy}")
    return CAPAdam(
        cap_parameter_groups(model, WEIGHT_DECAY),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        k=K,
        T_s=T_S,
        T_a=T_A,
        T_m=T_M_RELEASE,
        passive=False,
        protection=PROTECTION,
        retention=policy,
        decay_mode=DECAY_MODE,
        track_bind_stats=TRACK_BIND_STATS,
    )


def protected_importance(optimizer: CAPAdam) -> Tensor:
    values: list[Tensor] = []
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


@torch.no_grad()
def refresh_global_c(optimizer: CAPAdam) -> Optional[float]:
    value = float(protected_importance(optimizer).median().item())
    if value == 0.0:
        return None
    if value < 0.0 or not math.isfinite(value):
        raise RuntimeError(f"invalid global importance median: {value}")
    for group in optimizer.param_groups:
        group["c"] = value
    return value


def protected_bind_counts(optimizer: CAPAdam) -> tuple[int, int]:
    count = 0
    numel = 0
    for group in optimizer.param_groups:
        if bool(group["protect"]):
            count += int(group["last_bind_count"])
            numel += int(group["last_bind_numel"])
    return count, numel


def first_linear_weight(model: nn.Module) -> nn.Parameter:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if tuple(module.weight.shape) != (300, 784):
                raise RuntimeError(
                    "E4a realistic instrumentation requires first Linear weight "
                    f"shape (300, 784), got {tuple(module.weight.shape)}"
                )
            return module.weight
    raise RuntimeError("MLP300 contains no Linear layer")


def realistic_silent_mask(device: torch.device) -> Tensor:
    input_mask = torch.zeros(784, dtype=torch.bool, device=device)
    for row in range(14):
        for column in range(14):
            input_mask[28 * row + column] = True
    mask = input_mask.repeat(300)
    if int(mask.count_nonzero().item()) != 58_800:
        raise RuntimeError("realistic silent instrumentation mask is malformed")
    return mask


@torch.no_grad()
def mechanism_statistics(
    model: nn.Module, optimizer: CAPAdam, construction: str
) -> tuple[float, float]:
    importance_values: list[Tensor] = []
    radius_values: list[Tensor] = []
    target_parameter = first_linear_weight(model) if construction == "realistic" else None

    for group in optimizer.param_groups:
        if not bool(group["protect"]):
            continue
        for parameter in group["params"]:
            if target_parameter is not None and parameter is not target_parameter:
                continue
            state = optimizer.state.get(parameter)
            if not state or "imp" not in state:
                raise RuntimeError("CAP importance state is not initialized")
            importance = state["imp"].detach().reshape(-1)
            radius = (
                float(group["k"])
                / (state["imp"].detach().sqrt() + float(group["eps"]))
            ).reshape(-1)
            if target_parameter is not None:
                mask = realistic_silent_mask(parameter.device)
                importance = importance[mask]
                radius = radius[mask]
            importance_values.append(importance)
            radius_values.append(radius)

    if not importance_values:
        raise RuntimeError("E4a instrumentation set is empty")
    importance = torch.cat(importance_values)
    radius = torch.cat(radius_values)
    return float(importance.median().item()), float(radius.median().item())


def evaluate_and_resume_training(
    model: nn.Module, loader, device: torch.device
) -> float:
    _, accuracy = evaluate(model, loader, device)
    model.train()
    return accuracy


def train_batch_step(
    model: nn.Module,
    optimizer: CAPAdam,
    batch,
    device: torch.device,
    optimizer_step: int,
) -> float:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    if optimizer_step % C_REFRESH_STEPS == 0:
        refresh_global_c(optimizer)
    return float(loss.item())


def zero_gradient_step(
    model: nn.Module, optimizer: CAPAdam, optimizer_step: int
) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    if optimizer_step % C_REFRESH_STEPS == 0:
        refresh_global_c(optimizer)


def run_one(
    *,
    policy: str,
    construction: str,
    silence_length: int,
    seed: int,
    mnist_train: Dataset,
    a_test_loader,
    device: torch.device,
    num_workers: int,
) -> tuple[dict, list[dict], list[dict], dict]:
    """Execute one exact E4a cell in the prescribed phase order."""
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"unknown E4a construction: {construction}")
    if silence_length not in SILENCE_LENGTHS:
        raise ValueError(f"unknown E4a silence length: {silence_length}")

    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer = create_optimizer(model, policy)
    model.train()

    a_loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    a_batches = repeat_batches(a_loader)

    early_bind_count = 0
    early_bind_numel = 0
    optimizer_step = 0
    for a_step in range(1, PHASE_A_STEPS + 1):
        optimizer_step += 1
        train_batch_step(
            model, optimizer, next(a_batches), device, optimizer_step
        )
        if a_step <= EARLY_A_BIND_STEPS:
            bind_count, bind_numel = protected_bind_counts(optimizer)
            early_bind_count += bind_count
            early_bind_numel += bind_numel

    a_end = evaluate_and_resume_training(model, a_test_loader, device)
    end_a_quantiles = protected_importance_quantiles(optimizer)
    median_imp, median_radius = mechanism_statistics(
        model, optimizer, construction
    )
    mechanism_rows = [
        {
            "policy": policy,
            "construction": construction,
            "L": silence_length,
            "seed": seed,
            "checkpoint": "end_A",
            "median_imp": median_imp,
            "median_radius": median_radius,
            "a_test_accuracy": a_end,
        }
    ]
    early_bind_rate = early_bind_count / max(early_bind_numel, 1)

    if a_end < CONFIG_ACCURACY_FLOOR:
        return (
            {
                "policy": policy,
                "construction": construction,
                "L": silence_length,
                "seed": seed,
                "a_end": a_end,
                "a_after_silence": "",
                "a_final": "",
                "min_a_during_b": "",
                "early_a_bind_rate": early_bind_rate,
                "status": "CONFIG-SUSPECT",
            },
            mechanism_rows,
            [],
            {
                "policy": policy,
                "construction": construction,
                "L": silence_length,
                "seed": seed,
                **end_a_quantiles,
            },
        )

    a_after_silence: float | str = ""
    if silence_length > 0:
        if construction == "synthetic":
            for _ in range(silence_length):
                optimizer_step += 1
                zero_gradient_step(model, optimizer, optimizer_step)
        else:
            masked_dataset = QuadrantMaskedDataset(mnist_train)
            silence_loader = make_loader(
                masked_dataset,
                BATCH_SIZE,
                device,
                num_workers,
                shuffle=True,
                seed=seed + 10_000,
            )
            silence_batches = repeat_batches(silence_loader)
            for _ in range(silence_length):
                optimizer_step += 1
                train_batch_step(
                    model,
                    optimizer,
                    next(silence_batches),
                    device,
                    optimizer_step,
                )

        a_after_silence = evaluate_and_resume_training(
            model, a_test_loader, device
        )
        median_imp, median_radius = mechanism_statistics(
            model, optimizer, construction
        )
        mechanism_rows.append(
            {
                "policy": policy,
                "construction": construction,
                "L": silence_length,
                "seed": seed,
                "checkpoint": "end_silence",
                "median_imp": median_imp,
                "median_radius": median_radius,
                "a_test_accuracy": a_after_silence,
            }
        )

    task_b = PermutedDataset(
        mnist_train, fixed_permutation(784, B_PERMUTATION_SEED)
    )
    b_loader = make_loader(
        task_b,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed + 20_000,
    )
    b_batches = repeat_batches(b_loader)
    trace_rows: list[dict] = []
    minimum_a = math.inf
    a_final = math.nan
    for b_step in range(1, PHASE_B_STEPS + 1):
        optimizer_step += 1
        train_batch_step(
            model, optimizer, next(b_batches), device, optimizer_step
        )
        if b_step % B_EVALUATE_EVERY == 0 or b_step == PHASE_B_STEPS:
            accuracy = evaluate_and_resume_training(
                model, a_test_loader, device
            )
            minimum_a = min(minimum_a, accuracy)
            a_final = accuracy
            trace_rows.append(
                {
                    "policy": policy,
                    "construction": construction,
                    "L": silence_length,
                    "seed": seed,
                    "b_step": b_step,
                    "a_test_accuracy": accuracy,
                }
            )

    return (
        {
            "policy": policy,
            "construction": construction,
            "L": silence_length,
            "seed": seed,
            "a_end": a_end,
            "a_after_silence": a_after_silence,
            "a_final": a_final,
            "min_a_during_b": minimum_a,
            "early_a_bind_rate": early_bind_rate,
            "status": "ok",
        },
        mechanism_rows,
        trace_rows,
        {
            "policy": policy,
            "construction": construction,
            "L": silence_length,
            "seed": seed,
            **end_a_quantiles,
        },
    )


def numeric_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else math.nan


def numeric_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def grade_e4a(rows: Sequence[Mapping]) -> dict:
    """Apply the exact status filtering, criteria, and named-control order."""
    ok_rows = [row for row in rows if str(row["status"]) == "ok"]
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in ok_rows:
        grouped[
            (
                str(row["policy"]),
                str(row["construction"]),
                int(float(row["L"])),
            )
        ].append(float(row["a_final"]))
    cell_means = {key: mean(values) for key, values in grouped.items()}

    control_margins = {}
    for construction in CONSTRUCTIONS:
        low = cell_means.get(("leak", construction, 1_000), math.nan)
        high = cell_means.get(("leak", construction, 20_000), math.nan)
        control_margins[construction] = low - high
    control_ok = any(
        math.isfinite(value) and value >= GRADE_DELTA
        for value in control_margins.values()
    )

    release_drops = []
    release_bug_cells = []
    for construction in CONSTRUCTIONS:
        for low_index, low_length in enumerate(SILENCE_LENGTHS):
            for high_length in SILENCE_LENGTHS[low_index + 1 :]:
                low = cell_means.get(
                    ("release", construction, low_length), math.nan
                )
                high = cell_means.get(
                    ("release", construction, high_length), math.nan
                )
                drop = low - high
                if math.isfinite(drop):
                    release_drops.append(drop)
                    if drop > GRADE_DELTA:
                        release_bug_cells.append(
                            {
                                "construction": construction,
                                "L_lo": low_length,
                                "L_hi": high_length,
                                "drop": drop,
                            }
                        )
    release_bug = bool(release_bug_cells)

    treatment_margins = {}
    for policy in ("ratchet", "release"):
        for construction in CONSTRUCTIONS:
            low = cell_means.get((policy, construction, 1_000), math.nan)
            high = cell_means.get((policy, construction, 20_000), math.nan)
            treatment_margins[f"{policy}/{construction}"] = high - low
    treatment_ok = all(
        math.isfinite(value) and value >= -GRADE_DELTA
        for value in treatment_margins.values()
    )

    release_bind_rates = [
        float(row["early_a_bind_rate"])
        for row in ok_rows
        if str(row["policy"]) == "release"
    ]
    burn_in_means_by_policy = {
        policy: numeric_mean(
            float(row["early_a_bind_rate"])
            for row in ok_rows
            if str(row["policy"]) == policy
        )
        for policy in POLICIES
    }
    burn_in_mean = numeric_mean(release_bind_rates)
    burn_in_ok = math.isfinite(burn_in_mean) and burn_in_mean < 0.001

    if release_bug:
        verdict = "RELEASE-LEAKED-IN-SILENCE"
    elif not control_ok:
        verdict = "SILENCE-CONTROL-ABSENT"
    elif treatment_ok:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "cell_means": cell_means,
        "control_margins": control_margins,
        "control_ok": control_ok,
        "release_bug": release_bug,
        "release_bug_cells": release_bug_cells,
        "maximum_release_drop": max(release_drops, default=math.nan),
        "treatment_margins": treatment_margins,
        "treatment_ok": treatment_ok,
        "burn_in_means_by_policy": burn_in_means_by_policy,
        "burn_in_mean": burn_in_mean,
        "burn_in_ok": burn_in_ok,
        "verdict": verdict,
    }


def e4b_treatment_status(path: Path) -> tuple[str, list[int]]:
    """Return E4b's controlled treatment status for the combined verdict."""
    if not path.exists():
        return "PENDING", []
    rows = read_csv(path)
    config_path = path.parent / "run_config.json"
    if config_path.exists():
        configured_seeds = {
            int(seed)
            for seed in json.loads(
                config_path.read_text(encoding="utf-8")
            ).get("seeds", [])
        }
    else:
        configured_seeds = {int(float(row["seed"])) for row in rows}
    if not configured_seeds:
        return "PENDING", []
    required_arms = {
        "cap_leak",
        "cap_ratchet",
        "cap_release_2000",
        "cap_release_10000",
        "cap_release_50000",
    }
    task_rows = [
        row
        for row in rows
        if str(row["arm"]) in required_arms
        and 1 <= int(float(row["task"])) <= 200
    ]
    if not task_rows:
        return "PENDING", []
    completed_cells = {
        (str(row["arm"]), int(float(row["seed"])))
        for row in task_rows
        if int(float(row["task"])) == 200
    }
    expected_cells = {
        (arm, seed) for arm in required_arms for seed in configured_seeds
    }
    if not expected_cells.issubset(completed_cells):
        return "PENDING", []

    by_arm_task: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in task_rows:
        by_arm_task[
            (str(row["arm"]), int(float(row["task"])))
        ].append(float(row["task_end_accuracy"]))

    def window(arm: str, start: int, end: int) -> float:
        values = []
        for task in range(start, end + 1):
            cell_values = by_arm_task[(arm, task)]
            if len(cell_values) != len(configured_seeds):
                return math.nan
            values.append(mean(cell_values))
        return mean(values)

    ratchet_early = window("cap_ratchet", 1, 20)
    ratchet_late = window("cap_ratchet", 181, 200)
    control_ok = (
        math.isfinite(ratchet_early)
        and math.isfinite(ratchet_late)
        and ratchet_late <= ratchet_early - 0.05
    )
    if not control_ok:
        return "CONTROL-ABSENT", []

    leak_late = window("cap_leak", 181, 200)
    passing = []
    for timescale in (2_000, 10_000, 50_000):
        release_late = window(f"cap_release_{timescale}", 181, 200)
        if (
            math.isfinite(leak_late)
            and math.isfinite(release_late)
            and release_late >= leak_late - 0.02
        ):
            passing.append(timescale)
    return ("PASS" if passing else "FAIL"), passing


def combined_e4_verdict(
    grade: Mapping, stream_tasks_path: Path
) -> tuple[str, list[int]]:
    e4b_status, passing_timescales = e4b_treatment_status(stream_tasks_path)
    if e4b_status == "PENDING":
        return "PENDING", []
    if e4b_status != "PASS" or not bool(grade["treatment_ok"]):
        return "FAIL", []
    shared = [value for value in passing_timescales if value == 10_000]
    return ("PASS" if shared else "FAIL"), shared


def summarize_quantile_records(records: Sequence[Mapping]) -> dict:
    if not records:
        return {"median": math.nan, "p99": math.nan, "max": math.nan}
    return {
        key: mean(float(row[key]) for row in records)
        for key in ("median", "p99", "max")
    }


def format_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    return f"{mean(values):.4f} ± {numeric_std(values):.4f}"


def protocol_deviations(seeds: Sequence[int]) -> list[str]:
    seed_set = set(seeds)
    if seed_set in (set(FULL_SEEDS), set(COMPUTE_FLOOR_SEEDS)):
        return []
    return [
        "Non-preregistered seed subset/order was requested: "
        + ", ".join(map(str, seeds))
    ]


def build_report(
    rows: Sequence[Mapping],
    mechanism_rows: Sequence[Mapping],
    quantile_records: Sequence[Mapping],
    seeds: Sequence[int],
    runtime_seconds: float,
    stream_tasks_path: Path = Path("results/stream_a/stream_tasks.csv"),
) -> str:
    grade = grade_e4a(rows)
    report_verdict = prefix_preliminary_verdict(
        str(grade["verdict"]), seeds, FULL_SEEDS
    )
    burn_verdict = (
        "BURN-IN-FUNCTIONAL"
        if bool(grade["burn_in_ok"])
        else "BURN-IN-FAIL"
    )
    burn_verdict = prefix_preliminary_verdict(
        burn_verdict, seeds, FULL_SEEDS
    )
    combined, shared_timescales = combined_e4_verdict(
        grade, stream_tasks_path
    )
    if combined != "PENDING":
        combined = prefix_preliminary_verdict(combined, seeds, FULL_SEEDS)

    quantiles = summarize_quantile_records(quantile_records)
    deviations = protocol_deviations(seeds)
    lines = [
        "# E4a report — long-silence memory test",
        "",
        "## Protocol header",
        "",
        (
            "Forced deviations: none."
            if not deviations
            else "Forced deviations: " + "; ".join(deviations) + "."
        ),
        f"Seed set run: {tuple(seeds)}.",
        (
            "Preregistered five-seed set complete: "
            f"{len(seeds) == len(FULL_SEEDS) and set(seeds) == set(FULL_SEEDS)}."
        ),
        (
            "Settings: MLP300(784,10); lr=1e-3; betas=(0.9,0.999); "
            "eps=1e-8; wd=0.01; batch=128; k=3e-5; T_s=2000; "
            "T_a=2000; protection=box; decay_mode=blend; "
            "track_bind_stats=True; release T_m=10000; c refreshed every "
            "100 optimizer steps from the global protected-weight imp median."
        ),
        (
            "Protected-weight imp quantiles at end_A, averaged over runs: "
            f"median={quantiles['median']:.6g}; p99={quantiles['p99']:.6g}; "
            f"max={quantiles['max']:.6g}."
        ),
        "",
        "## Criteria and measured values",
        "",
        (
            "- E4A-0 BURN-IN-FUNCTIONAL: release mean early_a_bind_rate "
            f"={float(grade['burn_in_mean']):.6g}; threshold < 0.001; "
            f"passes={grade['burn_in_ok']}; verdict={burn_verdict}. "
            "All policy means: "
            + ", ".join(
                f"{policy}={float(value):.6g}"
                for policy, value in grade["burn_in_means_by_policy"].items()
            )
            + "."
        ),
        (
            "- Control: leak A(1000) − A(20000) by construction = "
            + ", ".join(
                f"{construction}:{float(value):.4f}"
                for construction, value in grade["control_margins"].items()
            )
            + f"; threshold ≥ {GRADE_DELTA:.2f} for at least one; "
            f"passes={grade['control_ok']}."
        ),
        (
            "- Bug trap: maximum release drop for any L_hi > L_lo "
            f"={float(grade['maximum_release_drop']):.4f}; bug threshold "
            f"> {GRADE_DELTA:.2f}; triggered={grade['release_bug']}."
        ),
        (
            "- Treatment: A(20000) − A(1000) margins = "
            + ", ".join(
                f"{key}:{float(value):.4f}"
                for key, value in grade["treatment_margins"].items()
            )
            + f"; every margin threshold ≥ -{GRADE_DELTA:.2f}; "
            + (
                f"passes={grade['treatment_ok']}."
                if grade["control_ok"]
                else "reported descriptively because the control failed."
            )
        ),
        f"- E4a verdict: **{report_verdict}**.",
    ]
    if str(grade["verdict"]) == "SILENCE-CONTROL-ABSENT":
        lines.append("silence created no vulnerability at these settings")

    combined_text = (
        "pending E4b completion"
        if combined == "PENDING"
        else (
            f"{combined}; shared passing T_m values={shared_timescales}"
            if shared_timescales
            else f"{combined}; no T_m value passed both treatments"
        )
    )
    lines.extend(
        [
            f"- Combined E4 verdict: **{combined_text}**.",
            "",
            "## Final A-test accuracy by cell",
            "",
            "| Policy | Construction | L | a_final mean ± std | ok seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        for construction in CONSTRUCTIONS:
            for silence_length in SILENCE_LENGTHS:
                values = [
                    float(row["a_final"])
                    for row in rows
                    if str(row["status"]) == "ok"
                    and str(row["policy"]) == policy
                    and str(row["construction"]) == construction
                    and int(float(row["L"])) == silence_length
                ]
                lines.append(
                    f"| {policy} | {construction} | {silence_length} | "
                    f"{format_mean_std(values)} | {len(values)} |"
                )

    lines.extend(
        [
            "",
            "## Silence mechanism",
            "",
            "| Policy | Construction | L | Checkpoint | median_imp mean | "
            "median_radius mean | A accuracy mean | rows |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    mechanism_groups: dict[
        tuple[str, str, int, str], list[Mapping]
    ] = defaultdict(list)
    for row in mechanism_rows:
        mechanism_groups[
            (
                str(row["policy"]),
                str(row["construction"]),
                int(float(row["L"])),
                str(row["checkpoint"]),
            )
        ].append(row)
    for policy in POLICIES:
        for construction in CONSTRUCTIONS:
            for silence_length in SILENCE_LENGTHS:
                for checkpoint in ("end_A", "end_silence"):
                    group_rows = mechanism_groups.get(
                        (policy, construction, silence_length, checkpoint), []
                    )
                    if not group_rows:
                        continue
                    lines.append(
                        f"| {policy} | {construction} | {silence_length} | "
                        f"{checkpoint} | "
                        f"{mean(float(row['median_imp']) for row in group_rows):.6g} | "
                        f"{mean(float(row['median_radius']) for row in group_rows):.6g} | "
                        f"{mean(float(row['a_test_accuracy']) for row in group_rows):.4f} | "
                        f"{len(group_rows)} |"
                    )

    suspect_count = sum(
        str(row["status"]) == "CONFIG-SUSPECT" for row in rows
    )
    lines.extend(
        [
            "",
            (
                "Burn-in line: release mean early_a_bind_rate "
                f"={float(grade['burn_in_mean']):.6g}; threshold < 0.001; "
                f"{burn_verdict}."
            ),
            f"CONFIG-SUSPECT runs: {suspect_count}.",
            f"Runtime: {runtime_seconds:.1f} seconds ({runtime_seconds / 3600.0:.3f} hours).",
            "",
        ]
    )
    return "\n".join(lines)


def serializable_grade(grade: Mapping) -> dict:
    return {
        "control_margins": dict(grade["control_margins"]),
        "control_ok": bool(grade["control_ok"]),
        "release_bug": bool(grade["release_bug"]),
        "release_bug_cells": list(grade["release_bug_cells"]),
        "maximum_release_drop": float(grade["maximum_release_drop"]),
        "treatment_margins": dict(grade["treatment_margins"]),
        "treatment_ok": bool(grade["treatment_ok"]),
        "burn_in_means_by_policy": dict(grade["burn_in_means_by_policy"]),
        "burn_in_mean": float(grade["burn_in_mean"]),
        "burn_in_ok": bool(grade["burn_in_ok"]),
        "verdict": str(grade["verdict"]),
    }


def base_run_config(args: argparse.Namespace, seeds: Sequence[int]) -> dict:
    return {
        "experiment": "E4a",
        "implementation_procedure": "e4-implementation-procedure.md section 1",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seeds": list(seeds),
        "full_seeds": list(FULL_SEEDS),
        "compute_floor_seeds": list(COMPUTE_FLOOR_SEEDS),
        "scientific_settings": {
            "policies": list(POLICIES),
            "constructions": list(CONSTRUCTIONS),
            "silence_lengths": list(SILENCE_LENGTHS),
            "model": "MLP300(input_units=784, output_units=10)",
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "k": K,
            "T_s": T_S,
            "T_a": T_A,
            "release_T_m": T_M_RELEASE,
            "protection": PROTECTION,
            "decay_mode": DECAY_MODE,
            "track_bind_stats": TRACK_BIND_STATS,
            "c_refresh_steps": C_REFRESH_STEPS,
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "early_a_bind_steps": EARLY_A_BIND_STEPS,
            "b_evaluate_every": B_EVALUATE_EVERY,
            "evaluation_subset_size": EVALUATION_SUBSET_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "a_test_subset_seed": A_TEST_SUBSET_SEED,
            "b_permutation_seed": B_PERMUTATION_SEED,
            "config_accuracy_floor": CONFIG_ACCURACY_FLOOR,
            "grade_delta": GRADE_DELTA,
        },
        "forced_deviations": protocol_deviations(seeds),
        "end_a_imp_quantiles": [],
        "runtime_seconds": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FULL_SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e4a"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if not args.seeds:
        parser.error("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def validate_coverage(rows: Sequence[Mapping], seeds: Sequence[int]) -> None:
    expected = {
        (policy, construction, silence_length, seed)
        for policy in POLICIES
        for construction in CONSTRUCTIONS
        for silence_length in SILENCE_LENGTHS
        for seed in seeds
    }
    observed = {
        (
            str(row["policy"]),
            str(row["construction"]),
            int(float(row["L"])),
            int(float(row["seed"])),
        )
        for row in rows
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"incomplete E4a result coverage; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )


def cell_key(row: Mapping) -> tuple[str, str, int, int]:
    return (
        str(row["policy"]),
        str(row["construction"]),
        int(float(row["L"])),
        int(float(row["seed"])),
    )


def repair_incomplete_resume_cells(
    results: list[dict],
    mechanism_rows: list[dict],
    trace_rows: list[dict],
    quantile_records: list[dict],
) -> None:
    """Drop narrowly interrupted cells so resume recomputes them atomically."""
    quantile_keys = {cell_key(row) for row in quantile_records}
    result_by_key = {cell_key(row): row for row in results}
    invalid = set()
    for key, result in result_by_key.items():
        mechanisms = [
            row for row in mechanism_rows if cell_key(row) == key
        ]
        traces = [row for row in trace_rows if cell_key(row) == key]
        expected_mechanisms = (
            1
            if str(result["status"]) == "CONFIG-SUSPECT" or key[2] == 0
            else 2
        )
        expected_traces = (
            0
            if str(result["status"]) == "CONFIG-SUSPECT"
            else PHASE_B_STEPS // B_EVALUATE_EVERY
        )
        if (
            len(mechanisms) != expected_mechanisms
            or len(traces) != expected_traces
            or key not in quantile_keys
        ):
            invalid.add(key)
    if not invalid:
        return
    results[:] = [row for row in results if cell_key(row) not in invalid]
    mechanism_rows[:] = [
        row for row in mechanism_rows if cell_key(row) not in invalid
    ]
    trace_rows[:] = [
        row for row in trace_rows if cell_key(row) not in invalid
    ]
    quantile_records[:] = [
        row for row in quantile_records if cell_key(row) not in invalid
    ]


def refresh_default_e4b_report(e4a_results_path: Path) -> None:
    """Update E4b's combined-verdict line when its default artifacts exist."""
    stream_dir = Path("results/stream_a")
    required = (
        stream_dir / "stream_scalars.csv",
        stream_dir / "stream_tasks.csv",
        stream_dir / "run_config.json",
    )
    if not all(path.exists() for path in required):
        return
    from experiment_e4b import build_report as build_e4b_report

    config = json.loads(
        (stream_dir / "run_config.json").read_text(encoding="utf-8")
    )
    report = build_e4b_report(
        read_csv(stream_dir / "stream_tasks.csv"),
        read_csv(stream_dir / "stream_scalars.csv"),
        config.get("final_imp_quantiles", []),
        tuple(int(seed) for seed in config["seeds"]),
        float(config.get("runtime_seconds", 0.0)),
        e4a_results_path=e4a_results_path,
    )
    (stream_dir / "report_e4b.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    verify_preconditions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "e4a_results.csv"
    mechanism_path = args.output_dir / "e4a_mechanism.csv"
    trace_path = args.output_dir / "e4a_trace.csv"
    report_path = args.output_dir / "report.md"
    config_path = args.output_dir / "run_config.json"

    existing_config = read_optional_json(config_path)
    if args.summarize_only:
        if not existing_config:
            raise FileNotFoundError(f"required E4a config does not exist: {config_path}")
        seeds = tuple(int(seed) for seed in existing_config["seeds"])
    else:
        seeds = args.seeds

    if args.resume and existing_config:
        configured_seeds = tuple(int(seed) for seed in existing_config["seeds"])
        if set(configured_seeds) != set(seeds):
            raise ValueError(
                "resume seed set does not match run_config.json: "
                f"{seeds} vs {configured_seeds}"
            )
        config = existing_config
    else:
        config = base_run_config(args, seeds)

    prior_runtime = float(config.get("runtime_seconds", 0.0))
    invocation_start = time.perf_counter()

    if args.summarize_only:
        results = read_csv(result_path)
        mechanism_rows = read_csv(mechanism_path)
        trace_rows = read_csv(trace_path)
    else:
        if args.resume:
            results = read_optional_csv(result_path)
            mechanism_rows = read_optional_csv(mechanism_path)
            trace_rows = read_optional_csv(trace_path)
        else:
            results = []
            mechanism_rows = []
            trace_rows = []
            config["end_a_imp_quantiles"] = []

        quantile_records = list(config.get("end_a_imp_quantiles", []))
        if args.resume:
            repair_incomplete_resume_cells(
                results, mechanism_rows, trace_rows, quantile_records
            )
            config["end_a_imp_quantiles"] = quantile_records
            write_csv(result_path, results, RESULT_FIELDS)
            write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
            write_csv(trace_path, trace_rows, TRACE_FIELDS)
            atomic_write_json(config_path, config)
        completed = {
            (
                str(row["policy"]),
                str(row["construction"]),
                int(float(row["L"])),
                int(float(row["seed"])),
            )
            for row in results
        }

        mnist_train, mnist_test = load_mnist(args.data_dir)
        device = torch.device(args.device)
        a_test_loader = make_loader(
            fixed_test_subset(mnist_test, A_TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )

        for policy in POLICIES:
            for construction in CONSTRUCTIONS:
                for silence_length in SILENCE_LENGTHS:
                    for seed in seeds:
                        key = (policy, construction, silence_length, seed)
                        if key in completed:
                            print(f"Skipping completed E4a cell {key}")
                            continue
                        print(f"Starting E4a cell {key}")
                        result, mechanisms, traces, quantiles = run_one(
                            policy=policy,
                            construction=construction,
                            silence_length=silence_length,
                            seed=seed,
                            mnist_train=mnist_train,
                            a_test_loader=a_test_loader,
                            device=device,
                            num_workers=args.num_workers,
                        )
                        results.append(result)
                        mechanism_rows.extend(mechanisms)
                        trace_rows.extend(traces)
                        quantile_records = [
                            row
                            for row in quantile_records
                            if (
                                str(row["policy"]),
                                str(row["construction"]),
                                int(row["L"]),
                                int(row["seed"]),
                            )
                            != key
                        ]
                        quantile_records.append(quantiles)
                        config["end_a_imp_quantiles"] = quantile_records
                        config["updated_at_utc"] = utc_now()
                        write_csv(result_path, results, RESULT_FIELDS)
                        write_csv(mechanism_path, mechanism_rows, MECHANISM_FIELDS)
                        write_csv(trace_path, trace_rows, TRACE_FIELDS)
                        atomic_write_json(config_path, config)
                        completed.add(key)

    validate_coverage(results, seeds)
    total_runtime = prior_runtime + (time.perf_counter() - invocation_start)
    quantile_records = list(config.get("end_a_imp_quantiles", []))
    report = build_report(
        results,
        mechanism_rows,
        quantile_records,
        seeds,
        total_runtime,
    )
    report_path.write_text(report, encoding="utf-8")
    grade = grade_e4a(results)
    config["updated_at_utc"] = utc_now()
    config["runtime_seconds"] = total_runtime
    config["grade"] = serializable_grade(grade)
    atomic_write_json(config_path, config)
    refresh_default_e4b_report(result_path)
    print(report)
    print(f"E4a artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
