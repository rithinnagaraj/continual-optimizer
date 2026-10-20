"""E4b: lifetime-rigidity test on the unified 200-task MNIST stream.

The four E4b arms are banked.  P1-A7 extends the same engine with three v4
primary comparators and the tuned primary CAP release arm.  The single frozen
retention-focused CAP endpoint is admitted only by a grading-owned conditional
request.  Completed core cells remain byte-for-byte data sources and are
skipped on ``--resume``; mandatory 50,000-step checkpoints retain deterministic
resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiment_e2 import (
    protected_importance_quantiles,
    read_csv,
    write_csv,
)
from experiment_e3 import (
    fixed_permutation,
    load_mnist,
    make_loader,
    repeat_batches,
    seed_everything,
)
from experiment_e4a import (
    ADAM_EPS,
    BATCH_SIZE,
    BETAS,
    COMPUTE_FLOOR_SEEDS,
    C_REFRESH_STEPS,
    DECAY_MODE,
    EVALUATION_BATCH_SIZE,
    EVALUATION_SUBSET_SIZE,
    FROZEN_RADIUS_THRESHOLD,
    FULL_SEEDS,
    K,
    LEARNING_RATE,
    PROTECTION,
    T_A,
    T_S,
    TRACK_BIND_STATS,
    WEIGHT_DECAY,
    atomic_write_json,
    fixed_test_subset,
    protected_bind_counts,
    protocol_deviations,
    read_optional_csv,
    read_optional_json,
    refresh_global_c,
    utc_now,
    verify_preconditions,
)
from model import MLP300
from phase1_arms import (
    CONDITIONAL_RETUNE_ARM,
    FINAL_STREAM_AMENDMENT,
    P1_A3_DISCLOSURE,
    P1_A4_DISCLOSURE,
    P1_A5_DISCLOSURE,
    P1_A6_DISCLOSURE,
    P1_A7_DISCLOSURE,
    UPGD_AMENDMENT,
    build_arm,
    load_conditional_retune_request,
    load_tuned_configs,
    resolved_config_for_arm,
    cap_release_matches_core,
    upgd_provenance,
    validate_p1_a7_artifact,
)
from cap_adam_experimental import CAPAdam, protection_radius


TASK_COUNT = 200
STEPS_PER_TASK = 5_000
TOTAL_STEPS = TASK_COUNT * STEPS_PER_TASK
SCALAR_LOG_EVERY = 100
CHECKPOINT_EVERY = 50_000
STEPS_TO_THRESHOLD_WINDOW = 200
STEPS_TO_THRESHOLD = 0.85
TEST_SUBSET_SEED = 999
PERMUTATION_SEED_BASE = 100_000
PRELIMINARY_PREFIX = "PRELIMINARY-"
E5_NORM_BLEND_ARM = "cap_release_10000"
E5_NORM_BASELINE_STEP = 100_000
E5_NORM_FINAL_STEP = 1_000_000
E5_NORM_MULTIPLIER = 1.5
REDUCED_ARM_DEVIATION = (
    "User-directed reduced arm matrix: only adamw, cap_leak, cap_ratchet, "
    "and cap_release_10000 were run. The preregistered release T_m=2000 and "
    "T_m=50000 arms and the ungraded E5-norm zero/anchor trajectory arms "
    "were omitted; E4b treatment grading is therefore limited to T_m=10000."
)

CORE_ARMS = (
    "adamw",
    "cap_leak",
    "cap_ratchet",
    "cap_release_10000",
)
PHASE1_APPEND_ARMS = (
    "adam_replay",
    "upgd",
    "ewc_oracle",
)
ARMS = (*CORE_ARMS, *PHASE1_APPEND_ARMS)
ARM_SETTINGS = {
    "adamw": {"kind": "adamw"},
    "cap_leak": {
        "kind": "cap",
        "retention": "leak",
        "T_m": 10_000.0,
        "decay_mode": "blend",
    },
    "cap_ratchet": {
        "kind": "cap",
        "retention": "ratchet",
        "T_m": 10_000.0,
        "decay_mode": "blend",
    },
    "cap_release_10000": {
        "kind": "cap",
        "retention": "release",
        "T_m": 10_000.0,
        "decay_mode": "blend",
    },
}
RELEASE_TIMESCALE_ARMS = {
    10_000: "cap_release_10000",
}

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


def e4b_protocol_deviations(seeds: Sequence[int]) -> list[str]:
    """Return seed deviations plus the explicit reduced-arm deviation."""
    return [*protocol_deviations(seeds), REDUCED_ARM_DEVIATION]


def phase1_run_arms(
    tuned_configs: Mapping,
    include_conditional_retune: bool = False,
) -> tuple[str, ...]:
    """Return the four banked arms plus all required Phase-1 append arms."""
    appended = list(ARMS)
    if not cap_release_matches_core(tuned_configs):
        appended.append("cap_release_tuned")
    if include_conditional_retune:
        appended.append(CONDITIONAL_RETUNE_ARM)
    return tuple(appended)


def phase1_arm_seeds(
    active_arms: Sequence[str], primary_seeds: Sequence[int]
) -> dict[str, tuple[int, ...]]:
    return {
        arm: tuple(int(seed) for seed in primary_seeds)
        for arm in active_arms
    }


def create_optimizer(
    model: nn.Module,
    arm: str,
    tuned_configs: Mapping,
    seed: int,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, Mapping]:
    # Core cells retain their exact banked E4b construction.  Every appended
    # arm reads the canonical artifact; no tuned value is copied here.
    configs = None if arm in CORE_ARMS else tuned_configs
    return build_arm(arm, model, configs, seed, device)


@torch.no_grad()
def cap_stream_statistics(optimizer: CAPAdam) -> dict:
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


def permute_batch(inputs: Tensor, permutation: Tensor) -> Tensor:
    batch_size = inputs.shape[0]
    return inputs.view(batch_size, 784)[:, permutation].view(
        batch_size, 1, 28, 28
    )


@torch.no_grad()
def evaluate_current_task(
    model: nn.Module,
    loader,
    permutation: Tensor,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = permute_batch(inputs, permutation)
        logits = model(inputs)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += targets.numel()
    if count == 0:
        raise ValueError("cannot evaluate an empty test subset")
    accuracy = correct / count
    model.train()
    return accuracy


def checkpoint_path(checkpoint_dir: Path, arm: str, seed: int, step: int) -> Path:
    return checkpoint_dir / f"{arm}_seed{seed}_step{step}.pt"


def checkpoint_step_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_step", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed E4b checkpoint filename: {path}") from error


def latest_checkpoint(
    checkpoint_dir: Path, arm: str, seed: int
) -> Optional[Path]:
    paths = list(checkpoint_dir.glob(f"{arm}_seed{seed}_step*.pt"))
    return max(paths, key=checkpoint_step_from_path) if paths else None


def load_torch_checkpoint(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scalar_row_count: int,
    task_row_count: int,
    *,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "row_counts": {
            "stream_scalars": scalar_row_count,
            "stream_tasks": task_row_count,
        },
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


def rows_for_cell(rows: Sequence[Mapping], arm: str, seed: int) -> list[Mapping]:
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


def trim_rows_to_checkpoint(
    scalar_rows: list[dict],
    task_rows: list[dict],
    checkpoint: Mapping,
) -> None:
    row_counts = checkpoint.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("E4b checkpoint is missing row_counts")
    scalar_count = int(row_counts["stream_scalars"])
    task_count = int(row_counts["stream_tasks"])
    if scalar_count > len(scalar_rows) or task_count > len(task_rows):
        raise ValueError(
            "checkpoint row counts exceed the rows currently present in CSVs"
        )
    del scalar_rows[scalar_count:]
    del task_rows[task_count:]


def fast_forward_batches(batches, count: int) -> None:
    """Discard exactly count CPU batches without transferring any tensor."""
    for skipped in range(count):
        next(batches)
        if (skipped + 1) % 100_000 == 0:
            print(f"  fast-forwarded {skipped + 1}/{count} CPU batches")


def final_quantile_record(
    arm: str, seed: int, optimizer: torch.optim.Optimizer, is_cap: bool
) -> Optional[dict]:
    if not is_cap:
        return None
    quantiles = protected_importance_quantiles(optimizer)
    return {"arm": arm, "seed": seed, **quantiles}


def recover_final_quantile_from_checkpoint(
    checkpoint_dir: Path,
    arm: str,
    seed: int,
    tuned_configs: Mapping,
    device: torch.device,
) -> Optional[dict]:
    if not arm.startswith("cap_"):
        return None
    path = checkpoint_path(checkpoint_dir, arm, seed, TOTAL_STEPS)
    if not path.exists():
        return None
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = create_optimizer(
        model, arm, tuned_configs, seed, device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return final_quantile_record(
        arm, seed, optimizer, bool(hooks["is_cap"])
    )


def run_cell(
    *,
    arm: str,
    seed: int,
    mnist_train,
    test_loader,
    permutations_cpu: Mapping[int, Tensor],
    scalar_rows: list[dict],
    task_rows: list[dict],
    scalar_path: Path,
    task_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
    num_workers: int,
    resume: bool,
    tuned_configs: Mapping,
    optimizer_factory: Optional[Callable[[nn.Module], tuple[Any, Mapping]]] = None,
    checkpoint_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    checkpoint_loaded_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
    checkpoint_trim_callback: Optional[
        Callable[[Mapping[str, Any], list[dict], list[dict]], None]
    ] = None,
    step_callback: Optional[
        Callable[[nn.Module, Any, int, int, int], None]
    ] = None,
    checkpoint_extra_factory: Optional[
        Callable[[nn.Module, Any, int], Mapping[str, Any]]
    ] = None,
) -> Optional[dict]:
    """Run or resume one arm-seed stream without optimizer boundary signals."""
    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = (
        optimizer_factory(model)
        if optimizer_factory is not None
        else create_optimizer(model, arm, tuned_configs, seed, device)
    )
    is_cap = bool(hooks["is_cap"])
    model.train()

    train_loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    batches = repeat_batches(train_loader)

    start_step = 0
    checkpoint = None
    checkpoint_file = latest_checkpoint(checkpoint_dir, arm, seed) if resume else None
    if checkpoint_file is not None:
        checkpoint = load_torch_checkpoint(checkpoint_file, map_location="cpu")
        if checkpoint_validator is not None:
            checkpoint_validator(checkpoint)
        start_step = int(checkpoint.get("step", -1))
        if (
            start_step < 0
            or start_step > TOTAL_STEPS
            or start_step % CHECKPOINT_EVERY != 0
        ):
            raise ValueError(
                f"invalid checkpoint step {start_step} in {checkpoint_file}"
            )
        if checkpoint_trim_callback is not None:
            checkpoint_trim_callback(checkpoint, scalar_rows, task_rows)
        else:
            trim_rows_to_checkpoint(scalar_rows, task_rows, checkpoint)
        write_csv(scalar_path, scalar_rows, SCALAR_FIELDS)
        write_csv(task_path, task_rows, TASK_FIELDS)
        print(
            f"Resuming {arm} seed={seed} from step={start_step}; "
            "rebuilding and fast-forwarding the CPU loader"
        )
        if start_step < TOTAL_STEPS:
            fast_forward_batches(batches, start_step)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "cpu_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if checkpoint_loaded_callback is not None:
            checkpoint_loaded_callback(checkpoint)
    else:
        remove_cell_rows(scalar_rows, arm, seed)
        remove_cell_rows(task_rows, arm, seed)
        write_csv(scalar_path, scalar_rows, SCALAR_FIELDS)
        write_csv(task_path, task_rows, TASK_FIELDS)

    if start_step == TOTAL_STEPS:
        return final_quantile_record(arm, seed, optimizer, is_cap)

    permutations_device = {
        task: permutation.to(device)
        for task, permutation in permutations_cpu.items()
    }
    accuracy_window: deque[float] = deque(maxlen=STEPS_TO_THRESHOLD_WINDOW)
    window_sum = 0.0
    steps_to_085: Optional[int] = None

    for step in range(start_step + 1, TOTAL_STEPS + 1):
        task = ((step - 1) // STEPS_PER_TASK) + 1
        within_task_step = ((step - 1) % STEPS_PER_TASK) + 1
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = permute_batch(inputs, permutations_device[task])

        incoming_accuracy = None
        if hooks["batch_mixer"] is not None:
            # P is measured on the full incoming current-task batch, before
            # stored examples replace any of it. This forward has no gradient.
            with torch.no_grad():
                incoming_accuracy = float(
                    (model(inputs).argmax(dim=1) == targets).float().mean().item()
                )
            inputs, targets = hooks["batch_mixer"](inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite loss for {arm} seed={seed} step={step}"
            )
        loss.backward()
        if hooks["grad_hook"] is not None:
            hooks["grad_hook"](model)
        optimizer.step()
        if is_cap and step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        if step_callback is not None:
            step_callback(model, optimizer, step, task, within_task_step)

        batch_accuracy = incoming_accuracy if incoming_accuracy is not None else float(
            (logits.argmax(dim=1) == targets).float().mean().item()
        )
        if len(accuracy_window) == STEPS_TO_THRESHOLD_WINDOW:
            window_sum -= accuracy_window[0]
        accuracy_window.append(batch_accuracy)
        window_sum += batch_accuracy
        if (
            steps_to_085 is None
            and within_task_step >= STEPS_TO_THRESHOLD_WINDOW
            and len(accuracy_window) == STEPS_TO_THRESHOLD_WINDOW
            and window_sum / STEPS_TO_THRESHOLD_WINDOW >= STEPS_TO_THRESHOLD
        ):
            steps_to_085 = within_task_step

        if step % SCALAR_LOG_EVERY == 0:
            cap_values = (
                cap_stream_statistics(optimizer)
                if is_cap
                else {
                    "median_radius": "",
                    "frozen_fraction": "",
                    "bind_rate": "",
                    "median_imp": "",
                    "theta_norm": "",
                }
            )
            scalar_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "batch_loss": float(loss.item()),
                    "batch_accuracy": batch_accuracy,
                    **cap_values,
                }
            )

        if within_task_step == STEPS_PER_TASK:
            task_end_accuracy = evaluate_current_task(
                model,
                test_loader,
                permutations_device[task],
                device,
            )
            task_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "task": task,
                    "task_end_accuracy": task_end_accuracy,
                    "steps_to_085": (
                        steps_to_085 if steps_to_085 is not None else ""
                    ),
                }
            )
            if hooks["on_boundary"] is not None:
                hooks["on_boundary"](model)
            write_csv(scalar_path, scalar_rows, SCALAR_FIELDS)
            write_csv(task_path, task_rows, TASK_FIELDS)

            accuracy_window.clear()
            window_sum = 0.0
            steps_to_085 = None

            if step % CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    checkpoint_path(checkpoint_dir, arm, seed, step),
                    model,
                    optimizer,
                    step,
                    len(scalar_rows),
                    len(task_rows),
                    extra_payload=(
                        checkpoint_extra_factory(model, optimizer, step)
                        if checkpoint_extra_factory is not None
                        else None
                    ),
                )
            print(
                f"  {arm} seed={seed} task={task}/{TASK_COUNT} "
                f"accuracy={task_end_accuracy:.4f}"
            )

    return final_quantile_record(arm, seed, optimizer, is_cap)


def task_accuracy_means(rows: Sequence[Mapping]) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["arm"]), int(float(row["task"])))
        ].append(float(row["task_end_accuracy"]))
    return {key: mean(values) for key, values in grouped.items()}


def arm_window(
    task_means: Mapping[tuple[str, int], float],
    arm: str,
    start: int,
    end: int,
) -> float:
    values = [
        task_means[(arm, task)]
        for task in range(start, end + 1)
        if (arm, task) in task_means
    ]
    return mean(values) if len(values) == end - start + 1 else math.nan


def grade_e4b(rows: Sequence[Mapping]) -> dict:
    task_means = task_accuracy_means(rows)
    ratchet_early = arm_window(task_means, "cap_ratchet", 1, 20)
    ratchet_late = arm_window(task_means, "cap_ratchet", 181, 200)
    ratchet_decline = ratchet_early - ratchet_late
    control_ok = (
        math.isfinite(ratchet_decline) and ratchet_decline >= 0.05
    )

    leak_late = arm_window(task_means, "cap_leak", 181, 200)
    treatment_margins = {}
    passing_timescales = []
    for timescale, arm in RELEASE_TIMESCALE_ARMS.items():
        release_late = arm_window(task_means, arm, 181, 200)
        margin = release_late - leak_late
        treatment_margins[str(timescale)] = margin
        if math.isfinite(margin) and margin >= -0.02:
            passing_timescales.append(timescale)
    treatment_ok = bool(passing_timescales)

    if not control_ok:
        verdict = "RIGIDITY-CONTROL-ABSENT"
    elif treatment_ok:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "task_means": task_means,
        "ratchet_early": ratchet_early,
        "ratchet_late": ratchet_late,
        "ratchet_decline": ratchet_decline,
        "control_ok": control_ok,
        "leak_late": leak_late,
        "treatment_margins": treatment_margins,
        "passing_timescales": passing_timescales,
        "treatment_ok": treatment_ok,
        "verdict": verdict,
    }


def report_verdict(verdict: str, seeds: Sequence[int]) -> str:
    """Apply hardening 1.6b explicitly to every displayed verdict."""
    full_seed_set_ran = (
        len(seeds) == len(FULL_SEEDS) and set(seeds) == set(FULL_SEEDS)
    )
    if full_seed_set_ran or verdict.startswith(PRELIMINARY_PREFIX):
        return verdict
    return f"{PRELIMINARY_PREFIX}{verdict}"


def e4a_regrade_verdict(e4a_results_path: Path) -> Optional[str]:
    """Read the discriminator-owned E4a verdict beside its result CSV."""
    regrade_path = e4a_results_path.parent / "regrade.json"
    if not regrade_path.exists():
        return None
    payload = json.loads(regrade_path.read_text(encoding="utf-8"))
    verdict = payload.get("verdict")
    if verdict is None and isinstance(payload.get("grade"), Mapping):
        verdict = payload["grade"].get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        raise ValueError(
            f"E4a re-grade artifact {regrade_path} must contain a verdict string"
        )
    verdict = verdict.strip()
    if verdict.startswith(PRELIMINARY_PREFIX):
        verdict = verdict[len(PRELIMINARY_PREFIX) :]
    if verdict == "CONTROL-ABSENT":
        return "SILENCE-CONTROL-ABSENT"
    return verdict


def e4a_controlled_treatment_status(path: Path) -> str:
    if not path.exists():
        return "PENDING"
    from experiment_e4a import grade_e4a

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
    expected = {
        (policy, construction, length, seed)
        for policy in ("leak", "ratchet", "release")
        for construction in ("synthetic", "realistic")
        for length in (0, 1_000, 4_000, 20_000)
        for seed in configured_seeds
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
    if not configured_seeds or not expected.issubset(observed):
        return "PENDING"
    grade = grade_e4a(rows)
    regrade_verdict = e4a_regrade_verdict(path)
    if regrade_verdict is not None:
        return regrade_verdict
    if not bool(grade["control_ok"]):
        return "SILENCE-CONTROL-ABSENT"
    if bool(grade["release_bug"]) or not bool(grade["treatment_ok"]):
        return "PENDING-E4A-REGRADE"
    return "PASS"


def combined_e4_verdict(
    grade: Mapping, e4a_results_path: Path
) -> tuple[str, list[int]]:
    e4a_status = e4a_controlled_treatment_status(e4a_results_path)
    if e4a_status != "PASS":
        return e4a_status, []
    if not bool(grade["control_ok"]):
        return "RIGIDITY-CONTROL-ABSENT", []
    if not bool(grade["treatment_ok"]):
        return "FAIL", []
    shared = [
        timescale
        for timescale in grade["passing_timescales"]
        if int(timescale) == 10_000
    ]
    return ("PASS" if shared else "FAIL"), shared


def theta_norm_values_at_step(
    scalar_rows: Sequence[Mapping],
    arm: str,
    step: int,
    seeds: Sequence[int],
) -> list[float]:
    """Return one exact theta-norm value per requested seed at one step."""
    values_by_seed: dict[int, float] = {}
    requested_seeds = set(int(seed) for seed in seeds)
    for row in scalar_rows:
        if str(row["arm"]) != arm or int(float(row["step"])) != step:
            continue
        seed = int(float(row["seed"]))
        if seed not in requested_seeds or row["theta_norm"] == "":
            continue
        if seed in values_by_seed:
            raise ValueError(
                f"duplicate theta_norm row for arm={arm}, seed={seed}, step={step}"
            )
        values_by_seed[seed] = float(row["theta_norm"])
    return [values_by_seed[seed] for seed in seeds if seed in values_by_seed]


def grade_e5_norm(
    scalar_rows: Sequence[Mapping], seeds: Sequence[int]
) -> dict:
    """Grade the preregistered blend norm bound using means over seeds."""
    baseline_values = theta_norm_values_at_step(
        scalar_rows, E5_NORM_BLEND_ARM, E5_NORM_BASELINE_STEP, seeds
    )
    final_values = theta_norm_values_at_step(
        scalar_rows, E5_NORM_BLEND_ARM, E5_NORM_FINAL_STEP, seeds
    )
    complete = len(baseline_values) == len(seeds) and len(final_values) == len(
        seeds
    )
    baseline_mean = mean(baseline_values) if baseline_values else math.nan
    final_mean = mean(final_values) if final_values else math.nan
    limit = E5_NORM_MULTIPLIER * baseline_mean
    ratio = (
        final_mean / baseline_mean
        if math.isfinite(baseline_mean) and baseline_mean > 0.0
        else math.nan
    )
    passes = complete and math.isfinite(final_mean) and final_mean <= limit
    return {
        "baseline_step": E5_NORM_BASELINE_STEP,
        "final_step": E5_NORM_FINAL_STEP,
        "baseline_mean": baseline_mean,
        "final_mean": final_mean,
        "limit": limit,
        "ratio": ratio,
        "multiplier": E5_NORM_MULTIPLIER,
        "complete": complete,
        "passes": passes,
        "verdict": "PASS" if passes else ("FAIL" if complete else "INCOMPLETE"),
    }


def mean_optional(values: Sequence[float]) -> str:
    return f"{mean(values):.2f}" if values else "n/a"


def final_quantile_summary(records: Sequence[Mapping]) -> dict[str, dict]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in records:
        grouped[str(row["arm"])].append(row)
    return {
        arm: {
            key: mean(float(row[key]) for row in arm_rows)
            for key in ("median", "p99", "max")
        }
        for arm, arm_rows in grouped.items()
    }


def scalar_window_rows(
    scalar_rows: Sequence[Mapping], arm: str, start_task: int, end_task: int
) -> list[Mapping]:
    start_step = (start_task - 1) * STEPS_PER_TASK + 1
    end_step = end_task * STEPS_PER_TASK
    return [
        row
        for row in scalar_rows
        if str(row["arm"]) == arm
        and start_step <= int(float(row["step"])) <= end_step
    ]


def build_report(
    task_rows: Sequence[Mapping],
    scalar_rows: Sequence[Mapping],
    final_quantiles: Sequence[Mapping],
    seeds: Sequence[int],
    runtime_seconds: float,
    e4a_results_path: Path = Path("results/e4a/e4a_results.csv"),
) -> str:
    grade = grade_e4b(task_rows)
    e4b_report_verdict = report_verdict(str(grade["verdict"]), seeds)
    combined, shared_timescales = combined_e4_verdict(
        grade, e4a_results_path
    )
    combined_report_verdict = report_verdict(combined, seeds)
    e5_norm_grade = grade_e5_norm(scalar_rows, seeds)
    e5_norm_report_verdict = report_verdict(
        str(e5_norm_grade["verdict"]), seeds
    )
    quantile_summary = final_quantile_summary(final_quantiles)
    deviations = e4b_protocol_deviations(seeds)

    lines = [
        "# E4b report — lifetime rigidity on the unified stream",
        "",
        "## Protocol header",
        "",
        (
            f"Forced deviations: {P1_A3_DISCLOSURE}"
            if not deviations
            else "Forced deviations: " + "; ".join(deviations) + "."
        ),
        f"Seed set run: {tuple(seeds)}.",
        (
            "Preregistered five-seed set complete: "
            f"{len(seeds) == len(FULL_SEEDS) and set(seeds) == set(FULL_SEEDS)}."
        ),
        (
            "Settings: 200 tasks × 5000 steps; permutations fixed by "
            "100000+t; MLP300(784,10); lr=1e-3; betas=(0.9,0.999); "
            "eps=1e-8; wd=0.01; batch=128; CAP k=3e-5; T_s=2000; "
            "T_a=2000; protection=box; blend decay for all CAP arms; "
            "c refresh every 100 optimizer steps; no task-boundary "
            "information supplied to any optimizer."
        ),
        "",
        "Final protected-weight imp quantiles averaged over seeds:",
        "",
        "| CAP arm | median | p99 | max |",
        "|---|---:|---:|---:|",
    ]
    for arm in CORE_ARMS:
        if arm == "adamw":
            continue
        summary = quantile_summary.get(arm)
        if summary is None:
            lines.append(f"| {arm} | unavailable | unavailable | unavailable |")
        else:
            lines.append(
                f"| {arm} | {summary['median']:.6g} | "
                f"{summary['p99']:.6g} | {summary['max']:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Criteria and measured values",
            "",
            (
                "- Control: E(cap_ratchet,1..20) "
                f"={float(grade['ratchet_early']):.4f}; "
                "E(cap_ratchet,181..200) "
                f"={float(grade['ratchet_late']):.4f}; measured decline "
                f"={float(grade['ratchet_decline']):.4f}; threshold ≥ 0.05; "
                f"passes={grade['control_ok']}."
            ),
            (
                "- Treatment margins E(release_Tm,181..200) − "
                f"E(cap_leak,181..200), leak={float(grade['leak_late']):.4f}: "
                + ", ".join(
                    f"T_m={timescale}:{float(margin):.4f}"
                    for timescale, margin in grade["treatment_margins"].items()
                )
                + "; threshold ≥ -0.02 for the retained T_m=10000 arm; "
                + (
                    f"passing T_m values={grade['passing_timescales']}."
                    if grade["control_ok"]
                    else "reported descriptively because the control failed."
                )
            ),
            f"- E4b verdict: **{e4b_report_verdict}**.",
        ]
    )
    if str(grade["verdict"]) == "RIGIDITY-CONTROL-ABSENT":
        lines.append(
            "pure ratchet does not freeze within 200 tasks at k = 3e-5"
        )
    if str(grade["verdict"]) == "FAIL":
        lines.append(
            "Preregistered redesign after this treatment failure: replace the "
            "per-step bind in the release rule with the fraction of clipped "
            "steps over a 100-step window, run one rerun, and let a second "
            "failure stand."
        )

    if combined == "PENDING":
        combined_detail = "E4a artifacts are incomplete or unavailable"
    elif combined == "PENDING-E4A-REGRADE":
        combined_detail = "waiting for the discriminator re-grade artifact"
    elif shared_timescales:
        combined_detail = f"shared passing T_m values={shared_timescales}"
    elif combined in (
        "SILENCE-CONTROL-ABSENT",
        "RIGIDITY-CONTROL-ABSENT",
    ):
        combined_detail = "validity control absent; treatment not graded"
    else:
        combined_detail = "no T_m value passed both controlled treatments"
    lines.extend(
        [
            f"- Combined E4 verdict: **{combined_report_verdict}**; "
            f"{combined_detail}.",
            "",
            "## E5-norm criterion and decay-mode trajectories",
            "",
            (
                "- Graded blend arm cap_release_10000: mean theta_norm at "
                f"step {int(e5_norm_grade['final_step']):,} "
                f"={float(e5_norm_grade['final_mean']):.6g}; threshold <= "
                f"{float(e5_norm_grade['multiplier']):.1f} x mean theta_norm "
                f"at step {int(e5_norm_grade['baseline_step']):,} "
                f"({float(e5_norm_grade['baseline_mean']):.6g}), measured "
                f"limit={float(e5_norm_grade['limit']):.6g}, measured ratio="
                f"{float(e5_norm_grade['ratio']):.6g}; "
                f"passes={e5_norm_grade['passes']}; verdict="
                f"**{e5_norm_report_verdict}**."
            ),
            (
                "- The preregistered zero and anchor trajectories are unavailable "
                "because those ungraded comparator arms were omitted under the "
                "forced reduced-arm deviation."
            ),
            "",
            "| Decay arm | Step | theta_norm mean over seeds | seeds | Graded |",
            "|---|---:|---:|---:|---|",
        ]
    )
    e5_trajectory_arms = ((E5_NORM_BLEND_ARM, "yes"),)
    for arm, graded in e5_trajectory_arms:
        for step in range(E5_NORM_BASELINE_STEP, E5_NORM_FINAL_STEP + 1, 100_000):
            values = theta_norm_values_at_step(scalar_rows, arm, step, seeds)
            value_text = f"{mean(values):.6g}" if values else "n/a"
            lines.append(
                f"| {arm} | {step} | {value_text} | {len(values)} | {graded} |"
            )

    lines.extend(
        [
            "",
            "## Frozen-fraction and median-importance trajectories",
            "",
            "| Arm | Tasks | frozen_fraction mean | median_imp mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in CORE_ARMS:
        for start_task in range(1, TASK_COUNT + 1, 20):
            end_task = start_task + 19
            rows = scalar_window_rows(
                scalar_rows, arm, start_task, end_task
            )
            frozen_values = [
                float(row["frozen_fraction"])
                for row in rows
                if row["frozen_fraction"] != ""
            ]
            importance_values = [
                float(row["median_imp"])
                for row in rows
                if row["median_imp"] != ""
            ]
            lines.append(
                f"| {arm} | {start_task}–{end_task} | "
                f"{mean(frozen_values):.6f} | {mean(importance_values):.6g} |"
                if frozen_values and importance_values
                else f"| {arm} | {start_task}–{end_task} | n/a | n/a |"
            )

    lines.extend(
        [
            "",
            "## Learning-speed support",
            "",
            "| Arm | mean steps_to_085 tasks 1–20 | reached | "
            "mean steps_to_085 tasks 181–200 | reached |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in CORE_ARMS:
        early_rows = [
            row
            for row in task_rows
            if str(row["arm"]) == arm
            and 1 <= int(float(row["task"])) <= 20
        ]
        late_rows = [
            row
            for row in task_rows
            if str(row["arm"]) == arm
            and 181 <= int(float(row["task"])) <= 200
        ]
        early_values = [
            float(row["steps_to_085"])
            for row in early_rows
            if row["steps_to_085"] != ""
        ]
        late_values = [
            float(row["steps_to_085"])
            for row in late_rows
            if row["steps_to_085"] != ""
        ]
        lines.append(
            f"| {arm} | {mean_optional(early_values)} | {len(early_values)} | "
            f"{mean_optional(late_values)} | {len(late_values)} |"
        )

    lines.extend(
        [
            "",
            f"Runtime: {runtime_seconds:.1f} seconds ({runtime_seconds / 3600.0:.3f} hours).",
            "",
        ]
    )
    return "\n".join(lines)


def validate_coverage(
    scalar_rows: Sequence[Mapping],
    task_rows: Sequence[Mapping],
    seeds: Sequence[int],
    arms: Sequence[str] = ARMS,
    arm_seeds: Optional[Mapping[str, Sequence[int]]] = None,
) -> None:
    resolved_seeds = {
        arm: tuple(
            int(seed)
            for seed in (arm_seeds[arm] if arm_seeds is not None else seeds)
        )
        for arm in arms
    }
    expected_task_keys = {
        (arm, int(seed), task)
        for arm in arms
        for seed in resolved_seeds[arm]
        for task in range(1, TASK_COUNT + 1)
    }
    expected_scalar_keys = {
        (arm, int(seed), step)
        for arm in arms
        for seed in resolved_seeds[arm]
        for step in range(SCALAR_LOG_EVERY, TOTAL_STEPS + 1, SCALAR_LOG_EVERY)
    }
    expected_task_count = len(expected_task_keys)
    expected_scalar_count = len(expected_scalar_keys)
    if len(task_rows) != expected_task_count:
        raise RuntimeError(
            f"stream_tasks.csv has {len(task_rows)} rows; "
            f"expected {expected_task_count}"
        )
    if len(scalar_rows) != expected_scalar_count:
        raise RuntimeError(
            f"stream_scalars.csv has {len(scalar_rows)} rows; "
            f"expected {expected_scalar_count}"
        )
    task_keys = {
        (
            str(row["arm"]),
            int(float(row["seed"])),
            int(float(row["task"])),
        )
        for row in task_rows
    }
    scalar_keys = {
        (
            str(row["arm"]),
            int(float(row["seed"])),
            int(float(row["step"])),
        )
        for row in scalar_rows
    }
    if len(task_keys) != len(task_rows) or task_keys != expected_task_keys:
        raise RuntimeError(
            "stream_tasks.csv contains duplicate, omitted-arm, or invalid cells"
        )
    if len(scalar_keys) != len(scalar_rows) or scalar_keys != expected_scalar_keys:
        raise RuntimeError(
            "stream_scalars.csv contains duplicate, omitted-arm, or invalid cells"
        )


def serializable_grade(grade: Mapping) -> dict:
    return {
        "ratchet_early": float(grade["ratchet_early"]),
        "ratchet_late": float(grade["ratchet_late"]),
        "ratchet_decline": float(grade["ratchet_decline"]),
        "control_ok": bool(grade["control_ok"]),
        "leak_late": float(grade["leak_late"]),
        "treatment_margins": dict(grade["treatment_margins"]),
        "passing_timescales": list(grade["passing_timescales"]),
        "treatment_ok": bool(grade["treatment_ok"]),
        "verdict": str(grade["verdict"]),
    }


def serializable_e5_norm_grade(grade: Mapping) -> dict:
    return {
        "baseline_step": int(grade["baseline_step"]),
        "final_step": int(grade["final_step"]),
        "baseline_mean": float(grade["baseline_mean"]),
        "final_mean": float(grade["final_mean"]),
        "limit": float(grade["limit"]),
        "ratio": float(grade["ratio"]),
        "multiplier": float(grade["multiplier"]),
        "complete": bool(grade["complete"]),
        "passes": bool(grade["passes"]),
        "verdict": str(grade["verdict"]),
    }


def serializable_arguments(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def phase1_append_configs(
    tuned_configs: Mapping, active_arms: Sequence[str]
) -> dict[str, Mapping]:
    return {
        arm: resolved_config_for_arm(arm, tuned_configs)
        for arm in active_arms
        if arm not in CORE_ARMS
    }


def migrate_resume_config(
    config: Mapping,
    args: argparse.Namespace,
    seeds: Sequence[int],
    active_arms: Sequence[str],
    tuned_configs: Mapping,
    conditional_request: Optional[Mapping] = None,
) -> dict:
    """Migrate the banked E4b config to the Phase-1 append matrix."""
    migrated = dict(config)
    settings = dict(migrated.get("scientific_settings", {}))
    previous_arms = list(settings.get("arms", []))
    active_arms = list(active_arms)
    settings["arms"] = active_arms
    settings["release_timescales"] = list(RELEASE_TIMESCALE_ARMS)
    settings["e5_norm_trajectory_arms"] = [E5_NORM_BLEND_ARM]
    migrated["scientific_settings"] = settings
    migrated["forced_deviations"] = [
        *e4b_protocol_deviations(seeds),
        P1_A3_DISCLOSURE,
        P1_A4_DISCLOSURE,
        P1_A5_DISCLOSURE,
        P1_A6_DISCLOSURE,
        P1_A7_DISCLOSURE,
    ]
    migrated["final_imp_quantiles"] = [
        row
        for row in migrated.get("final_imp_quantiles", [])
        if str(row.get("arm")) in active_arms
    ]
    migrated["last_resume_arguments"] = serializable_arguments(args)
    migrated["phase1_append"] = {
        "amendment": FINAL_STREAM_AMENDMENT,
        "p1_a3_disclosure": P1_A3_DISCLOSURE,
        "core_arms": list(CORE_ARMS),
        "append_arms": [arm for arm in active_arms if arm not in CORE_ARMS],
        "arm_seeds": {
            arm: list(values)
            for arm, values in phase1_arm_seeds(active_arms, seeds).items()
        },
        "tuned_config_path": str(args.tuned_config_path),
        "tuned_config_sha256": hashlib.sha256(
            args.tuned_config_path.read_bytes()
        ).hexdigest(),
        "upgd": upgd_provenance(),
        "ewc_disclosure": (
            "ewc_oracle receives task-boundary signals unavailable to every "
            "other arm; its results are boundary-privileged."
        ),
        "adam_replay_label": (
            "out-of-class reference (stores data; violates the streaming "
            "constraint)"
        ),
        "resolved_append_configs": phase1_append_configs(
            tuned_configs, active_arms
        ),
        "conditional_retune": {
            "enabled": CONDITIONAL_RETUNE_ARM in active_arms,
            "request_path": (
                str(args.conditional_retune_request)
                if CONDITIONAL_RETUNE_ARM in active_arms
                else None
            ),
            "request": dict(conditional_request) if conditional_request else None,
        },
    }

    if previous_arms and previous_arms != active_arms:
        migrations = list(migrated.get("arm_set_migrations", []))
        migration = {
            "migrated_at_utc": utc_now(),
            "previous_arms": previous_arms,
            "active_arms": active_arms,
            "reason": (
                "Phase-1 P1-A7 unconditional append or grading-authorized "
                "conditional re-tune after E4b completion; banked core rows "
                "and report remain unchanged."
            ),
        }
        if not any(
            list(item.get("previous_arms", [])) == previous_arms
            and list(item.get("active_arms", [])) == active_arms
            for item in migrations
            if isinstance(item, Mapping)
        ):
            migrations.append(migration)
        migrated["arm_set_migrations"] = migrations

    for stale_key in ("grade", "e5_norm_grade", "combined_e4"):
        migrated.pop(stale_key, None)
    migrated["updated_at_utc"] = utc_now()
    return migrated


def base_run_config(
    args: argparse.Namespace,
    seeds: Sequence[int],
    active_arms: Sequence[str],
    tuned_configs: Mapping,
    conditional_request: Optional[Mapping] = None,
) -> dict:
    return {
        "experiment": "E4b",
        "implementation_procedure": "e4-implementation-procedure.md section 2",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "arguments": serializable_arguments(args),
        "seeds": list(seeds),
        "full_seeds": list(FULL_SEEDS),
        "compute_floor_seeds": list(COMPUTE_FLOOR_SEEDS),
        "scientific_settings": {
            "arms": list(active_arms),
            "tasks": TASK_COUNT,
            "steps_per_task": STEPS_PER_TASK,
            "total_steps": TOTAL_STEPS,
            "permutation_seed_formula": "100000 + t",
            "model": "MLP300(input_units=784, output_units=10)",
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "k": K,
            "T_s": T_S,
            "T_a": T_A,
            "protection": PROTECTION,
            "default_decay_mode": DECAY_MODE,
            "track_bind_stats": TRACK_BIND_STATS,
            "c_refresh_steps": C_REFRESH_STEPS,
            "scalar_log_every": SCALAR_LOG_EVERY,
            "checkpoint_every": CHECKPOINT_EVERY,
            "evaluation_subset_size": EVALUATION_SUBSET_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "test_subset_seed": TEST_SUBSET_SEED,
            "steps_to_085_window": STEPS_TO_THRESHOLD_WINDOW,
            "steps_to_085_threshold": STEPS_TO_THRESHOLD,
            "release_timescales": list(RELEASE_TIMESCALE_ARMS),
            "e5_norm_blend_arm": E5_NORM_BLEND_ARM,
            "e5_norm_trajectory_arms": [E5_NORM_BLEND_ARM],
            "e5_norm_baseline_step": E5_NORM_BASELINE_STEP,
            "e5_norm_final_step": E5_NORM_FINAL_STEP,
            "e5_norm_multiplier": E5_NORM_MULTIPLIER,
        },
        "forced_deviations": [
            *e4b_protocol_deviations(seeds),
            P1_A3_DISCLOSURE,
            P1_A4_DISCLOSURE,
            P1_A5_DISCLOSURE,
            P1_A6_DISCLOSURE,
            P1_A7_DISCLOSURE,
        ],
        "phase1_append": {
            "amendment": FINAL_STREAM_AMENDMENT,
            "p1_a3_disclosure": P1_A3_DISCLOSURE,
            "core_arms": list(CORE_ARMS),
            "append_arms": [arm for arm in active_arms if arm not in CORE_ARMS],
            "arm_seeds": {
                arm: list(values)
                for arm, values in phase1_arm_seeds(active_arms, seeds).items()
            },
            "tuned_config_path": str(args.tuned_config_path),
            "tuned_config_sha256": hashlib.sha256(
                args.tuned_config_path.read_bytes()
            ).hexdigest(),
            "upgd": upgd_provenance(),
            "ewc_disclosure": (
                "ewc_oracle receives task-boundary signals unavailable to "
                "every other arm; its results are boundary-privileged."
            ),
            "adam_replay_label": (
                "out-of-class reference (stores data; violates the streaming "
                "constraint)"
            ),
            "resolved_append_configs": phase1_append_configs(
                tuned_configs, active_arms
            ),
            "conditional_retune": {
                "enabled": CONDITIONAL_RETUNE_ARM in active_arms,
                "request_path": (
                    str(args.conditional_retune_request)
                    if CONDITIONAL_RETUNE_ARM in active_arms
                    else None
                ),
                "request": (
                    dict(conditional_request) if conditional_request else None
                ),
            },
        },
        "final_imp_quantiles": [],
        "runtime_seconds": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FULL_SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/stream_a")
    )
    parser.add_argument(
        "--e4a-results-path",
        type=Path,
        default=Path("results/e4a/e4a_results.csv"),
        help=(
            "E4a result CSV used for combined grading; regrade.json is read "
            "from the same directory."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--tuned-config-path",
        type=Path,
        default=Path("results/phase1/phase1_tuned_configs.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--conditional-retune",
        action="store_true",
        help=(
            "Append the frozen CAP retention-focused arm after "
            "phase1_grade.py has authorized the failure branch."
        ),
    )
    parser.add_argument(
        "--conditional-retune-request",
        type=Path,
        default=Path("results/phase1/conditional_retune_request.json"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if not args.seeds:
        parser.error("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if set(args.seeds) != set(FULL_SEEDS):
        parser.error("P1-A7 requires the full five-seed set")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def refresh_default_e4a_report(
    stream_tasks_path: Path, e4a_results_path: Path
) -> bool:
    """Refresh only an unregraded E4a run whose original grade already passed.

    A discriminator-owned re-grade, a pending re-grade, or either validity
    outcome leaves E4a's report untouched.  This prevents E4b from replacing
    the preregistered E4a record with a diagnosis based on the superseded
    original criterion.
    """
    if not e4a_results_path.exists():
        return False
    status = e4a_controlled_treatment_status(e4a_results_path)
    regrade_path = e4a_results_path.parent / "regrade.json"
    if status != "PASS" or regrade_path.exists():
        return False

    e4a_dir = e4a_results_path.parent
    required = (
        e4a_results_path,
        e4a_dir / "e4a_mechanism.csv",
        e4a_dir / "run_config.json",
    )
    if not all(path.exists() for path in required):
        return False
    from experiment_e4a import build_report as build_e4a_report

    config = json.loads((e4a_dir / "run_config.json").read_text(encoding="utf-8"))
    report = build_e4a_report(
        read_csv(e4a_results_path),
        read_csv(e4a_dir / "e4a_mechanism.csv"),
        config.get("end_a_imp_quantiles", []),
        tuple(int(seed) for seed in config["seeds"]),
        float(config.get("runtime_seconds", 0.0)),
        stream_tasks_path=stream_tasks_path,
    )
    (e4a_dir / "report.md").write_text(report, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    verify_preconditions()
    tuned_configs = load_tuned_configs(args.tuned_config_path)
    validate_p1_a7_artifact(tuned_configs)
    tuned_config_sha256 = hashlib.sha256(
        args.tuned_config_path.read_bytes()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    scalar_path = args.output_dir / "stream_scalars.csv"
    task_path = args.output_dir / "stream_tasks.csv"
    report_path = args.output_dir / "report_e4b.md"
    config_path = args.output_dir / "run_config.json"

    existing_config = read_optional_json(config_path)
    previous_tuned_hash = existing_config.get("phase1_append", {}).get(
        "tuned_config_sha256"
    )
    previous_amendment = existing_config.get("phase1_append", {}).get("amendment")
    if (
        previous_tuned_hash
        and previous_tuned_hash != tuned_config_sha256
        and previous_amendment == FINAL_STREAM_AMENDMENT
    ):
        raise RuntimeError(
            "phase1_tuned_configs.json changed during the Stream-A append"
        )
    conditional_enabled = bool(
        args.conditional_retune
        or existing_config.get("phase1_append", {})
        .get("conditional_retune", {})
        .get("enabled", False)
    )
    conditional_request: Optional[Mapping] = None
    if conditional_enabled:
        conditional_request = load_conditional_retune_request(
            args.conditional_retune_request,
            tuned_config_path=args.tuned_config_path,
            tuned_configs=tuned_configs,
        )
    active_arms = phase1_run_arms(tuned_configs, conditional_enabled)
    if args.summarize_only:
        if not existing_config:
            raise FileNotFoundError(f"required E4b config does not exist: {config_path}")
        seeds = tuple(int(seed) for seed in existing_config["seeds"])
    else:
        seeds = args.seeds
    if len(seeds) != len(FULL_SEEDS) or set(seeds) != set(FULL_SEEDS):
        raise ValueError("P1-A7 Stream-A append requires the full five-seed set")

    if args.summarize_only and existing_config:
        config = dict(existing_config)
    elif args.resume and existing_config:
        configured_seeds = tuple(int(seed) for seed in existing_config["seeds"])
        if set(configured_seeds) != set(seeds):
            raise ValueError(
                "resume seed set does not match run_config.json: "
                f"{seeds} vs {configured_seeds}"
            )
        config = migrate_resume_config(
            existing_config,
            args,
            seeds,
            active_arms,
            tuned_configs,
            conditional_request,
        )
        atomic_write_json(config_path, config)
    else:
        config = base_run_config(
            args,
            seeds,
            active_arms,
            tuned_configs,
            conditional_request,
        )

    prior_runtime = float(config.get("runtime_seconds", 0.0))
    invocation_start = time.perf_counter()

    if args.summarize_only:
        scalar_rows = read_csv(scalar_path)
        task_rows = read_csv(task_path)
    else:
        scalar_rows = read_optional_csv(scalar_path)
        task_rows = read_optional_csv(task_path)
        if (scalar_rows or task_rows) and not args.resume:
            raise RuntimeError(
                "banked Stream-A outputs exist; the Phase-1 append requires --resume"
            )
        if not args.resume:
            scalar_rows = []
            task_rows = []
            config["final_imp_quantiles"] = []
            write_csv(scalar_path, scalar_rows, SCALAR_FIELDS)
            write_csv(task_path, task_rows, TASK_FIELDS)
        else:
            active_scalar_rows = [
                row for row in scalar_rows if str(row["arm"]) in active_arms
            ]
            active_task_rows = [
                row for row in task_rows if str(row["arm"]) in active_arms
            ]
            removed_scalar_count = len(scalar_rows) - len(active_scalar_rows)
            removed_task_count = len(task_rows) - len(active_task_rows)
            if removed_scalar_count or removed_task_count:
                raise RuntimeError(
                    "Stream-A contains rows outside the frozen P1-A7 roster: "
                    f"scalars={removed_scalar_count}, tasks={removed_task_count}"
                )
            scalar_rows = active_scalar_rows
            task_rows = active_task_rows

        device = torch.device(args.device)
        mnist_train, mnist_test = load_mnist(args.data_dir)
        test_loader = make_loader(
            fixed_test_subset(mnist_test, TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )
        permutations_cpu = {
            task: fixed_permutation(784, PERMUTATION_SEED_BASE + task)
            for task in range(1, TASK_COUNT + 1)
        }
        quantile_records = list(config.get("final_imp_quantiles", []))

        completed = {
            (str(row["arm"]), int(float(row["seed"])))
            for row in task_rows
            if int(float(row["task"])) == TASK_COUNT
        }
        active_arm_seeds = phase1_arm_seeds(active_arms, seeds)
        for arm in active_arms:
            for seed in active_arm_seeds[arm]:
                key = (arm, seed)
                if key in completed:
                    print(f"Skipping completed E4b cell {key}")
                    if (
                        arm.startswith("cap_")
                        and not any(
                            str(row["arm"]) == arm
                            and int(row["seed"]) == seed
                            for row in quantile_records
                        )
                    ):
                        recovered = recover_final_quantile_from_checkpoint(
                            checkpoint_dir,
                            arm,
                            seed,
                            tuned_configs,
                            device,
                        )
                        if recovered is not None:
                            quantile_records.append(recovered)
                            config["final_imp_quantiles"] = quantile_records
                            atomic_write_json(config_path, config)
                    continue
                print(f"Starting E4b cell {key}")
                quantiles = run_cell(
                    arm=arm,
                    seed=seed,
                    mnist_train=mnist_train,
                    test_loader=test_loader,
                    permutations_cpu=permutations_cpu,
                    scalar_rows=scalar_rows,
                    task_rows=task_rows,
                    scalar_path=scalar_path,
                    task_path=task_path,
                    checkpoint_dir=checkpoint_dir,
                    device=device,
                    num_workers=args.num_workers,
                    resume=args.resume,
                    tuned_configs=tuned_configs,
                )
                quantile_records = [
                    row
                    for row in quantile_records
                    if not (
                        str(row["arm"]) == arm
                        and int(row["seed"]) == seed
                    )
                ]
                if quantiles is not None:
                    quantile_records.append(quantiles)
                config["final_imp_quantiles"] = quantile_records
                config["updated_at_utc"] = utc_now()
                atomic_write_json(config_path, config)
                completed.add(key)

    validate_coverage(
        scalar_rows,
        task_rows,
        seeds,
        active_arms,
        arm_seeds=phase1_arm_seeds(active_arms, seeds),
    )
    total_runtime = prior_runtime + (time.perf_counter() - invocation_start)
    final_quantiles = list(config.get("final_imp_quantiles", []))
    report = build_report(
        task_rows,
        scalar_rows,
        final_quantiles,
        seeds,
        total_runtime,
        e4a_results_path=args.e4a_results_path,
    )
    # E4b's verdict was banked before the append.  Preserve that report byte
    # for byte; Phase-1 comparisons are written by phase1_grade.py.
    if not report_path.exists():
        report_path.write_text(report, encoding="utf-8")
    else:
        report = report_path.read_text(encoding="utf-8")
    grade = grade_e4b(task_rows)
    e5_norm_grade = grade_e5_norm(scalar_rows, seeds)
    combined, shared_timescales = combined_e4_verdict(
        grade, args.e4a_results_path
    )
    config["updated_at_utc"] = utc_now()
    config["runtime_seconds"] = total_runtime
    config["grade"] = serializable_grade(grade)
    config["grade"]["raw_verdict"] = str(grade["verdict"])
    config["grade"]["verdict"] = report_verdict(str(grade["verdict"]), seeds)
    config["e5_norm_grade"] = serializable_e5_norm_grade(e5_norm_grade)
    config["e5_norm_grade"]["raw_verdict"] = str(e5_norm_grade["verdict"])
    config["e5_norm_grade"]["verdict"] = report_verdict(
        str(e5_norm_grade["verdict"]), seeds
    )
    config["combined_e4"] = {
        "verdict": report_verdict(combined, seeds),
        "raw_verdict": combined,
        "shared_passing_T_m": list(shared_timescales),
        "e4a_results_path": str(args.e4a_results_path),
        "e4a_regrade_path": str(args.e4a_results_path.parent / "regrade.json"),
    }
    atomic_write_json(config_path, config)
    refresh_default_e4a_report(task_path, args.e4a_results_path)
    print(report)
    print(f"E4b artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
