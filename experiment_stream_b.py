"""Phase-1 Stream B: recurring ten-permutation MNIST retention stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from experiment_e2 import read_csv, write_csv
from experiment_e3 import (
    fixed_permutation,
    load_mnist,
    make_loader,
    repeat_batches,
    seed_everything,
)
from experiment_e4a import (
    BATCH_SIZE,
    C_REFRESH_STEPS,
    EVALUATION_BATCH_SIZE,
    FULL_SEEDS,
    atomic_write_json,
    fixed_test_subset,
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
    P1_A7_FAILURE_BRANCH_STATEMENT,
    P1_A7_FREEZE_STATEMENT,
    RETENTION_FOCUSED_LABEL,
    RETENTION_FOCUSED_SUFFIX,
    build_arm,
    l2_descope_guard,
    load_conditional_retune_request,
    load_tuned_configs,
    method_for_arm,
    resolved_config_for_arm,
    upgd_provenance,
    validate_p1_a7_artifact,
)
from phase1_stream_common import (
    BLANK_CAP_STATISTICS,
    CHECKPOINT_EVERY,
    POOL_FIELDS,
    SCALAR_FIELDS,
    SCALAR_LOG_EVERY,
    TASK_FIELDS,
    StepsToThreshold,
    checkpoint_path,
    evaluate_permutation,
    fast_forward_batches,
    final_quantile_record,
    latest_checkpoint,
    load_torch_checkpoint,
    permute_batch,
    remove_cell_rows,
    restore_rng_state,
    cap_stream_statistics,
    save_checkpoint,
    training_step,
    trim_rows_to_checkpoint,
    validate_coverage,
)


TASK_COUNT = 200
STEPS_PER_TASK = 5_000
TOTAL_STEPS = TASK_COUNT * STEPS_PER_TASK
POOL_SIZE = 10
POOL_SEED_BASE = 200_001
SCHEDULE_SEED = 555
TEST_SUBSET_SEED = 999
PRELIMINARY_PREFIX = "PRELIMINARY-"

ARMS = (
    "adamw",
    "adam_replay",
    "upgd",
    "ewc_oracle",
    "cap_release",
    "cap_ratchet",
    "cap_leak",
)


def stream_b_arms(include_conditional_retune: bool = False) -> tuple[str, ...]:
    return (*ARMS, CONDITIONAL_RETUNE_ARM) if include_conditional_retune else ARMS


def stream_b_arm_seeds(
    active_arms: Sequence[str], primary_seeds: Sequence[int]
) -> dict[str, tuple[int, ...]]:
    return {
        arm: tuple(int(seed) for seed in primary_seeds)
        for arm in active_arms
    }

EWC_DISCLOSURE = (
    "ewc_oracle receives task-boundary signals unavailable to every other "
    "arm; its results are boundary-privileged."
)
REPLAY_LABEL = (
    "out-of-class reference (stores data; violates the streaming constraint)"
)


def stream_b_schedule() -> list[int]:
    return torch.randint(
        0,
        POOL_SIZE,
        (TASK_COUNT,),
        generator=torch.Generator().manual_seed(SCHEDULE_SEED),
    ).tolist()


def revisit_gap_distribution(schedule: Sequence[int]) -> dict[str, Any]:
    """Describe every realized same-pool revisit interval in tasks and steps."""
    per_pool: dict[str, Any] = {}
    all_gaps: list[int] = []
    for pool_index in range(POOL_SIZE):
        visits = [
            task
            for task, observed in enumerate(schedule, start=1)
            if int(observed) == pool_index
        ]
        gaps = [right - left for left, right in zip(visits, visits[1:])]
        all_gaps.extend(gaps)
        per_pool[str(pool_index)] = {
            "visit_tasks": visits,
            "gap_tasks": gaps,
            "gap_steps": [gap * STEPS_PER_TASK for gap in gaps],
        }
    if not all_gaps:
        raise RuntimeError("Stream-B schedule contains no revisit gaps")
    counts = dict(sorted(Counter(all_gaps).items()))
    return {
        "expected_gap_tasks": POOL_SIZE,
        "expected_gap_steps": POOL_SIZE * STEPS_PER_TASK,
        "count": len(all_gaps),
        "min_gap_tasks": min(all_gaps),
        "max_gap_tasks": max(all_gaps),
        "mean_gap_tasks": mean(all_gaps),
        "median_gap_tasks": median(all_gaps),
        "mean_gap_steps": mean(all_gaps) * STEPS_PER_TASK,
        "median_gap_steps": median(all_gaps) * STEPS_PER_TASK,
        "gap_task_counts": {str(key): value for key, value in counts.items()},
        "per_pool": per_pool,
    }


def display_arm(arm: str) -> str:
    if arm == "adam_replay":
        return f"adam_replay — {REPLAY_LABEL}"
    if arm.endswith(RETENTION_FOCUSED_SUFFIX):
        label = f"{arm} — {RETENTION_FOCUSED_LABEL}"
        if method_for_arm(arm) == "adam_replay":
            label += f"; {REPLAY_LABEL}"
        return label
    return arm


def _write_tables(
    scalar_path: Path,
    task_path: Path,
    pool_path: Path,
    scalar_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(scalar_path, scalar_rows, SCALAR_FIELDS)
    write_csv(task_path, task_rows, TASK_FIELDS)
    write_csv(pool_path, pool_rows, POOL_FIELDS)


def recover_final_quantile(
    *,
    checkpoint_dir: Path,
    arm: str,
    seed: int,
    tuned_configs: Mapping[str, Any],
    device: torch.device,
) -> Optional[dict[str, Any]]:
    path = checkpoint_path(checkpoint_dir, arm, seed, TOTAL_STEPS)
    if not path.exists():
        return None
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = build_arm(arm, model, tuned_configs, seed, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return final_quantile_record(arm, seed, optimizer, bool(hooks["is_cap"]))


def run_cell(
    *,
    arm: str,
    seed: int,
    tuned_configs: Mapping[str, Any],
    mnist_train,
    test_loader,
    permutations_cpu: Mapping[int, torch.Tensor],
    schedule: Sequence[int],
    scalar_rows: list[dict],
    task_rows: list[dict],
    pool_rows: list[dict],
    scalar_path: Path,
    task_path: Path,
    pool_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
    num_workers: int,
    resume: bool,
    pool_size: int = POOL_SIZE,
    optimizer_factory: Optional[Callable[[torch.nn.Module], tuple[Any, Mapping]]] = None,
    checkpoint_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    checkpoint_loaded_callback: Optional[
        Callable[[Mapping[str, Any]], None]
    ] = None,
    checkpoint_trim_callback: Optional[
        Callable[
            [Mapping[str, Any], list[dict], list[dict], list[dict]], None
        ]
    ] = None,
    step_callback: Optional[
        Callable[[torch.nn.Module, Any, int, int, int], None]
    ] = None,
    checkpoint_extra_factory: Optional[
        Callable[[torch.nn.Module, Any, int], Mapping[str, Any]]
    ] = None,
) -> Optional[dict[str, Any]]:
    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = (
        optimizer_factory(model)
        if optimizer_factory is not None
        else build_arm(arm, model, tuned_configs, seed, device)
    )
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
            raise ValueError(f"invalid checkpoint step in {checkpoint_file}")
        if checkpoint_trim_callback is not None:
            checkpoint_trim_callback(
                checkpoint, scalar_rows, task_rows, pool_rows
            )
        else:
            trim_rows_to_checkpoint(
                {
                    "streamb_scalars": scalar_rows,
                    "streamb_tasks": task_rows,
                    "streamb_pool": pool_rows,
                },
                checkpoint,
            )
        _write_tables(
            scalar_path, task_path, pool_path, scalar_rows, task_rows, pool_rows
        )
        print(
            f"Resuming {arm} seed={seed} from step={start_step}; "
            "fast-forwarding the CPU loader"
        )
        if start_step < TOTAL_STEPS:
            fast_forward_batches(batches, start_step)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint)
        if checkpoint_loaded_callback is not None:
            checkpoint_loaded_callback(checkpoint)
    else:
        for rows in (scalar_rows, task_rows, pool_rows):
            remove_cell_rows(rows, arm, seed)
        _write_tables(
            scalar_path, task_path, pool_path, scalar_rows, task_rows, pool_rows
        )

    if start_step == TOTAL_STEPS:
        return final_quantile_record(
            arm, seed, optimizer, bool(hooks["is_cap"])
        )

    permutations = {
        index: permutation.to(device)
        for index, permutation in permutations_cpu.items()
    }
    threshold = StepsToThreshold()
    for step in range(start_step + 1, TOTAL_STEPS + 1):
        task = ((step - 1) // STEPS_PER_TASK) + 1
        within_task_step = ((step - 1) % STEPS_PER_TASK) + 1
        pool_index = int(schedule[task - 1])
        inputs, targets = next(batches)
        inputs = permute_batch(
            inputs.to(device, non_blocking=True), permutations[pool_index]
        )
        targets = targets.to(device, non_blocking=True)
        loss, batch_accuracy = training_step(
            model=model,
            optimizer=optimizer,
            hooks=hooks,
            inputs=inputs,
            targets=targets,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite loss for {arm} seed={seed} step={step}"
            )
        if bool(hooks["is_cap"]) and step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        if step_callback is not None:
            step_callback(model, optimizer, step, task, within_task_step)
        threshold.update(batch_accuracy, within_task_step)

        if step % SCALAR_LOG_EVERY == 0:
            diagnostics = (
                cap_stream_statistics(optimizer)
                if bool(hooks["is_cap"])
                else dict(BLANK_CAP_STATISTICS)
            )
            scalar_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "batch_loss": float(loss.item()),
                    "batch_accuracy": batch_accuracy,
                    **diagnostics,
                }
            )

        if within_task_step == STEPS_PER_TASK:
            current_accuracy = None
            for evaluated_index in range(pool_size):
                accuracy = evaluate_permutation(
                    model,
                    test_loader,
                    permutations[evaluated_index],
                    device,
                )
                is_current = evaluated_index == pool_index
                if is_current:
                    current_accuracy = accuracy
                pool_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "task": task,
                        "pool_index": evaluated_index,
                        "accuracy": accuracy,
                        "is_current": is_current,
                    }
                )
            if current_accuracy is None:
                raise RuntimeError("current pool permutation was not evaluated")
            task_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "task": task,
                    "task_end_accuracy": current_accuracy,
                    "steps_to_085": (
                        threshold.reached if threshold.reached is not None else ""
                    ),
                }
            )
            # The boundary signal is disclosed and delivered only to EWC,
            # after all ten boundary evaluations, including after task 200.
            if hooks["on_boundary"] is not None:
                hooks["on_boundary"](model)
            _write_tables(
                scalar_path,
                task_path,
                pool_path,
                scalar_rows,
                task_rows,
                pool_rows,
            )
            threshold.reset()
            if step % CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    checkpoint_path(checkpoint_dir, arm, seed, step),
                    model,
                    optimizer,
                    step,
                    {
                        "streamb_scalars": len(scalar_rows),
                        "streamb_tasks": len(task_rows),
                        "streamb_pool": len(pool_rows),
                    },
                    extra_payload=(
                        checkpoint_extra_factory(model, optimizer, step)
                        if checkpoint_extra_factory is not None
                        else None
                    ),
                )
            print(
                f"  {arm} seed={seed} task={task}/{TASK_COUNT} "
                f"current_accuracy={current_accuracy:.4f}"
            )

    return final_quantile_record(arm, seed, optimizer, bool(hooks["is_cap"]))


def end_retention_by_seed(
    pool_rows: Sequence[Mapping[str, Any]],
    *,
    task_count: int = TASK_COUNT,
    pool_size: int = POOL_SIZE,
) -> dict[str, dict[int, float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in pool_rows:
        if int(float(row["task"])) != task_count:
            continue
        current = str(row["is_current"]).lower() in {"true", "1"}
        if not current:
            grouped[(str(row["arm"]), int(float(row["seed"])))].append(
                float(row["accuracy"])
            )
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (arm, seed), values in grouped.items():
        if len(values) != pool_size - 1:
            raise RuntimeError(f"invalid end-retention pool coverage for {arm}/{seed}")
        result[arm][seed] = mean(values)
    return dict(result)


def grade_controls(pool_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed = end_retention_by_seed(pool_rows)
    arm_means = {
        arm: mean(seed_values.values()) for arm, seed_values in by_seed.items()
    }
    replay_absent = arm_means["adam_replay"] <= arm_means["adamw"] + 0.01
    ewc_suspect = arm_means["ewc_oracle"] <= arm_means["adamw"] + 0.01
    return {
        "end_retention_by_seed": by_seed,
        "end_retention": arm_means,
        "replay_control_absent": replay_absent,
        "replay_verdict": "REPLAY-CONTROL-ABSENT" if replay_absent else "PASS",
        "ewc_integration_suspect": ewc_suspect,
        "ewc_verdict": "EWC-INTEGRATION-SUSPECT" if ewc_suspect else "PASS",
    }


def _report_verdict(value: str, seeds: Sequence[int]) -> str:
    complete = len(seeds) == len(FULL_SEEDS) and set(seeds) == set(FULL_SEEDS)
    return value if complete else PRELIMINARY_PREFIX + value


def relearning_summary(
    task_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[int],
    arms: Sequence[str] = ARMS,
) -> dict[str, tuple[Optional[float], Optional[float]]]:
    by_cell_pool: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        task = int(float(row["task"]))
        by_cell_pool[
            (str(row["arm"]), int(float(row["seed"])), int(schedule[task - 1]))
        ].append(row)
    first: dict[str, list[float]] = defaultdict(list)
    last: dict[str, list[float]] = defaultdict(list)
    for (arm, _seed, _pool), visits in by_cell_pool.items():
        visits.sort(key=lambda row: int(float(row["task"])))
        if visits[0]["steps_to_085"] != "":
            first[arm].append(float(visits[0]["steps_to_085"]))
        if visits[-1]["steps_to_085"] != "":
            last[arm].append(float(visits[-1]["steps_to_085"]))
    return {
        arm: (
            mean(first[arm]) if first[arm] else None,
            mean(last[arm]) if last[arm] else None,
        )
        for arm in arms
    }


def build_report(
    pool_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    schedule: Sequence[int],
    runtime_seconds: float,
    deployed_configs: Mapping[str, Mapping[str, Any]],
    tuned_configs: Mapping[str, Any],
    revisit_gaps: Mapping[str, Any],
    active_arms: Sequence[str] = ARMS,
    arm_seeds: Optional[Mapping[str, Sequence[int]]] = None,
) -> str:
    controls = grade_controls(pool_rows)
    l2_guard = l2_descope_guard(tuned_configs)
    relearning = relearning_summary(task_rows, schedule, active_arms)
    lines = [
        "# Phase 1 Stream B report",
        "",
        "## Protocol header",
        "",
        (
            "Forced deviations: "
            f"{P1_A3_DISCLOSURE} {P1_A4_DISCLOSURE} {P1_A5_DISCLOSURE} "
            f"{P1_A6_DISCLOSURE} {P1_A7_DISCLOSURE}"
        ),
        f"Seed set run: {tuple(seeds)}.",
        P1_A7_FREEZE_STATEMENT,
        (
            "Settings: 10 fixed input-permutation pool entries (seeds "
            "200001-200010); schedule torch.randint seed 555; 200 tasks x "
            "5,000 steps; MLP-300; batch 128; subset-999 evaluation of all "
            "10 pool entries at every boundary; configs read exclusively "
            "from phase1_tuned_configs.json."
        ),
        (
            "Shared optimizer settings: AdamW/CAP betas=(0.9,0.999), "
            "eps=1e-8, matrix-weight decay=0.01; CAP T_s=2000, box, blend, "
            "c refreshed every 100 steps; tuned lr/k/T_a/T_m values are "
            "printed in the resolved-configuration table."
        ),
        EWC_DISCLOSURE,
        f"adam_replay: {REPLAY_LABEL}.",
        (
            "Tuning disclosure: seed 42 used the deployment permutation "
            "formula in an online stream with no held-out notion; all methods "
            "received identical tuning budgets, and seed 42 remains in final "
            "aggregates."
        ),
        "",
        "## P1-A7 pre-launch freeze",
        "",
        f"- Unconditional roster: `{ARMS}`.",
        (
            "- Primary CAP configuration: `"
            + json.dumps(
                resolved_config_for_arm("cap_release", tuned_configs),
                sort_keys=True,
            )
            + "`."
        ),
        (
            "- Conditional CAP configuration: `"
            + json.dumps(
                resolved_config_for_arm(CONDITIONAL_RETUNE_ARM, tuned_configs),
                sort_keys=True,
            )
            + "`."
        ),
        f"- {P1_A7_FAILURE_BRANCH_STATEMENT}",
        "",
        "## Resolved configurations",
        "",
        "| Arm | Configuration read from tuned artifact |",
        "|---|---|",
    ]
    for arm in active_arms:
        lines.append(
            f"| {display_arm(arm)} | "
            f"`{json.dumps(deployed_configs[arm], sort_keys=True)}` |"
        )
    lines.extend(
        [
        "",
        "## End retention",
        "",
        "| Arm | Mean pool_retention(200) | SD over seeds | Role |",
        "|---|---:|---:|---|",
        ]
    )
    for arm in active_arms:
        role = (
            RETENTION_FOCUSED_LABEL
            if arm.endswith(RETENTION_FOCUSED_SUFFIX)
            else REPLAY_LABEL if arm == "adam_replay"
            else "boundary-privileged" if arm == "ewc_oracle"
            else "in-class"
        )
        seed_values = list(controls["end_retention_by_seed"][arm].values())
        deviation = stdev(seed_values) if len(seed_values) > 1 else 0.0
        lines.append(
            f"| {display_arm(arm)} | {controls['end_retention'][arm]:.6f} | "
            f"{deviation:.6f} | {role} |"
        )
    lines.extend(
        [
            "",
            (
                "- Replay validity: adam_replay retention "
                f"={controls['end_retention']['adam_replay']:.6f}; adamw+0.01 "
                f"={controls['end_retention']['adamw'] + 0.01:.6f}; verdict "
                f"**{_report_verdict(controls['replay_verdict'], seeds)}**."
            ),
            (
                "- EWC integration: ewc_oracle retention "
                f"={controls['end_retention']['ewc_oracle']:.6f}; adamw+0.01 "
                f"={controls['end_retention']['adamw'] + 0.01:.6f}; verdict "
                f"**{_report_verdict(controls['ewc_verdict'], seeds)}**."
            ),
            (
                "- l2_init descope guard: l2_init primary validation retention "
                f"={l2_guard['l2_init_primary_validation_retention']:.7f}; "
                f"max(adamw={l2_guard['opponents']['adamw']:.7f}, "
                f"upgd={l2_guard['opponents']['upgd']:.7f})="
                f"{l2_guard['binding_value']:.7f}; verdict "
                f"**{l2_guard['verdict']}**."
            ),
            (
                "The predeclared validation-scale L2 descope guard passed."
                if l2_guard["passed"]
                else "L2INIT-DESCOPE-INVALID: l2_init must run before clause "
                "(ii) can be graded."
            ),
            "",
            "## Realized revisit gaps",
            "",
            (
                "The frozen schedule produced "
                f"{int(revisit_gaps['count'])} consecutive revisit gaps: "
                f"mean={float(revisit_gaps['mean_gap_tasks']):.3f} tasks "
                f"({float(revisit_gaps['mean_gap_steps']):.1f} steps), "
                f"median={float(revisit_gaps['median_gap_tasks']):.3f} tasks, "
                f"range={int(revisit_gaps['min_gap_tasks'])}-"
                f"{int(revisit_gaps['max_gap_tasks'])} tasks. Expected gap="
                f"{int(revisit_gaps['expected_gap_tasks'])} tasks/"
                f"{int(revisit_gaps['expected_gap_steps'])} steps."
            ),
            "Gap histogram (tasks): `"
            + json.dumps(revisit_gaps["gap_task_counts"], sort_keys=True)
            + "`.",
            "",
            "## Relearning trend (descriptive)",
            "",
            "| Arm | First visit mean steps_to_085 | Last visit mean steps_to_085 |",
            "|---|---:|---:|",
        ]
    )
    for arm in active_arms:
        first, last = relearning[arm]
        lines.append(
            f"| {display_arm(arm)} | {first:.1f} | {last:.1f} |"
            if first is not None and last is not None
            else f"| {display_arm(arm)} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Final CAP diagnostics (descriptive only)",
            "",
            "| Arm | median radius | frozen fraction | bind rate | median importance | theta norm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in [item for item in active_arms if item.startswith("cap_")]:
        final_rows = [
            row
            for row in scalar_rows
            if str(row["arm"]) == arm
            and int(float(row["step"])) == TOTAL_STEPS
        ]
        values = {
            field: mean(float(row[field]) for row in final_rows)
            for field in (
                "median_radius",
                "frozen_fraction",
                "bind_rate",
                "median_imp",
                "theta_norm",
            )
        }
        lines.append(
            f"| {display_arm(arm)} | {values['median_radius']:.6g} | "
            f"{values['frozen_fraction']:.6f} | {values['bind_rate']:.6f} | "
            f"{values['median_imp']:.6g} | {values['theta_norm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Per-arm seed sets: `"
            + json.dumps(
                {
                    arm: list(values)
                    for arm, values in (arm_seeds or {}).items()
                },
                sort_keys=True,
            )
            + "`.",
            "",
            f"Runtime: {runtime_seconds:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FULL_SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/stream_b")
    )
    parser.add_argument(
        "--tuned-config-path",
        type=Path,
        default=Path("results/phase1/phase1_tuned_configs.json"),
    )
    parser.add_argument("--num-workers", type=int, default=8)
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
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be a nonempty unique list")
    if set(args.seeds) != set(FULL_SEEDS):
        parser.error("P1-A7 requires the full five-seed set")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


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
    scalar_path = args.output_dir / "streamb_scalars.csv"
    task_path = args.output_dir / "streamb_tasks.csv"
    pool_path = args.output_dir / "streamb_pool.csv"
    report_path = args.output_dir / "report.md"
    config_path = args.output_dir / "run_config.json"
    schedule = stream_b_schedule()
    existing_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    stream_started = (
        (scalar_path.exists() and bool(read_csv(scalar_path)))
        or (task_path.exists() and bool(read_csv(task_path)))
        or (checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")))
    )
    if stream_started and not (args.resume or args.summarize_only):
        raise RuntimeError("Stream-B outputs exist; use --resume")
    if (
        existing_config
        and existing_config.get("tuned_config_sha256")
        and existing_config["tuned_config_sha256"] != tuned_config_sha256
    ):
        if stream_started:
            raise RuntimeError(
                "phase1_tuned_configs.json changed after Stream B began; the "
                "freeze clause forbids continuing with different configs"
            )
        existing_config = {}
    existing_roster = list(
        existing_config.get("scientific_settings", {}).get("arms", [])
    )
    allowed_existing_rosters = (
        [],
        list(ARMS),
        list(stream_b_arms(True)),
    )
    if existing_roster not in allowed_existing_rosters:
        if stream_started:
            raise RuntimeError(
                "existing Stream-B arm roster is incompatible with P1-A7: "
                f"{existing_roster}"
            )
        # A manifest without any scientific rows is not a started stream.
        # Replace it so the required P1-A7 freeze is recorded before cell one.
        existing_config = {}
    if args.summarize_only and not existing_config:
        raise FileNotFoundError(f"missing Stream-B run config: {config_path}")
    conditional_enabled = bool(
        args.conditional_retune
        or existing_config.get("conditional_retune", {}).get("enabled", False)
    )
    conditional_request: Optional[dict[str, Any]] = None
    if conditional_enabled:
        conditional_request = load_conditional_retune_request(
            args.conditional_retune_request,
            tuned_config_path=args.tuned_config_path,
            tuned_configs=tuned_configs,
        )
    active_arms = stream_b_arms(conditional_enabled)
    deployed_configs = {
        arm: resolved_config_for_arm(arm, tuned_configs) for arm in active_arms
    }
    seeds = (
        tuple(int(seed) for seed in existing_config["seeds"])
        if args.summarize_only
        else args.seeds
    )
    if len(seeds) != len(FULL_SEEDS) or set(seeds) != set(FULL_SEEDS):
        raise ValueError("P1-A7 Stream B requires the full five-seed set")
    if args.resume and existing_config:
        configured = tuple(int(seed) for seed in existing_config["seeds"])
        if set(configured) != set(seeds):
            raise ValueError("resume seeds do not match run_config.json")

    arm_seeds = stream_b_arm_seeds(active_arms, seeds)
    visit_counts = dict(Counter(schedule))
    first_appearance = {
        index: schedule.index(index) + 1 for index in range(POOL_SIZE)
    }
    revisit_gaps = revisit_gap_distribution(schedule)
    freeze_record = {
        "recorded_at_utc": existing_config.get(
            "pre_stream_b_freeze_record", {}
        ).get("recorded_at_utc", utc_now()),
        "tuned_config_sha256": tuned_config_sha256,
        "unconditional_roster": list(ARMS),
        "primary_cap_config": resolved_config_for_arm(
            "cap_release", tuned_configs
        ),
        "conditional_retune_arm": CONDITIONAL_RETUNE_ARM,
        "conditional_retune_config": resolved_config_for_arm(
            CONDITIONAL_RETUNE_ARM, tuned_configs
        ),
        "selection_timing_statement": P1_A7_FREEZE_STATEMENT,
        "failure_branch_statement": P1_A7_FAILURE_BRANCH_STATEMENT,
    }

    config = existing_config or {
        "experiment": "phase1_stream_b",
        "created_at_utc": utc_now(),
        "seeds": list(seeds),
        "upgd": upgd_provenance(),
        "ewc_disclosure": EWC_DISCLOSURE,
        "adam_replay_label": REPLAY_LABEL,
        "final_imp_quantiles": [],
        "runtime_seconds": 0.0,
    }
    previous_arms = list(config.get("scientific_settings", {}).get("arms", []))
    if previous_arms not in ([], list(ARMS), list(active_arms)):
        raise RuntimeError(
            "existing Stream-B arm roster is incompatible with P1-A7: "
            f"{previous_arms}"
        )
    config.update(
        {
            "amendment": FINAL_STREAM_AMENDMENT,
            "p1_a3_disclosure": P1_A3_DISCLOSURE,
            "p1_a4_disclosure": P1_A4_DISCLOSURE,
            "p1_a5_disclosure": P1_A5_DISCLOSURE,
            "p1_a6_disclosure": P1_A6_DISCLOSURE,
            "p1_a7_disclosure": P1_A7_DISCLOSURE,
            "tuned_config_path": str(args.tuned_config_path),
            "tuned_config_sha256": tuned_config_sha256,
            "deployed_configs": deployed_configs,
            "pre_stream_b_freeze_record": freeze_record,
            "conditional_retune": {
                "enabled": conditional_enabled,
                "request_path": (
                    str(args.conditional_retune_request)
                    if conditional_enabled
                    else None
                ),
                "request": conditional_request,
            },
            "scientific_settings": {
                "arms": list(active_arms),
                "unconditional_arms": list(ARMS),
                "arm_seeds": {
                    arm: list(values) for arm, values in arm_seeds.items()
                },
                "tasks": TASK_COUNT,
                "steps_per_task": STEPS_PER_TASK,
                "pool_permutation_seeds": list(
                    range(POOL_SEED_BASE, POOL_SEED_BASE + POOL_SIZE)
                ),
                "schedule_seed": SCHEDULE_SEED,
                "schedule": schedule,
                "schedule_visit_counts": visit_counts,
                "schedule_first_appearance_tasks": first_appearance,
                "revisit_gap_distribution": revisit_gaps,
                "checkpoint_every": CHECKPOINT_EVERY,
                "scalar_log_every": SCALAR_LOG_EVERY,
            },
        }
    )
    atomic_write_json(config_path, config)
    prior_runtime = float(config.get("runtime_seconds", 0.0))
    started = time.perf_counter()

    if args.summarize_only:
        scalar_rows = read_csv(scalar_path)
        task_rows = read_csv(task_path)
        pool_rows = read_csv(pool_path)
    else:
        scalar_rows = read_csv(scalar_path) if args.resume and scalar_path.exists() else []
        task_rows = read_csv(task_path) if args.resume and task_path.exists() else []
        pool_rows = read_csv(pool_path) if args.resume and pool_path.exists() else []
        if not args.resume:
            _write_tables(scalar_path, task_path, pool_path, [], [], [])
        device = torch.device(args.device)
        mnist_train, mnist_test = load_mnist(args.data_dir)
        test_loader = make_loader(
            fixed_test_subset(mnist_test, TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )
        permutations = {
            index: fixed_permutation(784, POOL_SEED_BASE + index)
            for index in range(POOL_SIZE)
        }
        quantile_records = list(config.get("final_imp_quantiles", []))
        completed = {
            (str(row["arm"]), int(float(row["seed"])))
            for row in task_rows
            if int(float(row["task"])) == TASK_COUNT
        }
        for arm in active_arms:
            for seed in arm_seeds[arm]:
                key = (arm, seed)
                if key in completed:
                    print(f"Skipping completed Stream-B cell {key}")
                    if arm.startswith("cap_") and not any(
                        str(row["arm"]) == arm and int(row["seed"]) == seed
                        for row in quantile_records
                    ):
                        recovered = recover_final_quantile(
                            checkpoint_dir=checkpoint_dir,
                            arm=arm,
                            seed=seed,
                            tuned_configs=tuned_configs,
                            device=device,
                        )
                        if recovered is not None:
                            quantile_records.append(recovered)
                    continue
                print(f"Starting Stream-B cell {key}")
                quantiles = run_cell(
                    arm=arm,
                    seed=seed,
                    tuned_configs=tuned_configs,
                    mnist_train=mnist_train,
                    test_loader=test_loader,
                    permutations_cpu=permutations,
                    schedule=schedule,
                    scalar_rows=scalar_rows,
                    task_rows=task_rows,
                    pool_rows=pool_rows,
                    scalar_path=scalar_path,
                    task_path=task_path,
                    pool_path=pool_path,
                    checkpoint_dir=checkpoint_dir,
                    device=device,
                    num_workers=args.num_workers,
                    resume=args.resume,
                )
                quantile_records = [
                    row
                    for row in quantile_records
                    if not (
                        str(row["arm"]) == arm and int(row["seed"]) == seed
                    )
                ]
                if quantiles is not None:
                    quantile_records.append(quantiles)
                config["final_imp_quantiles"] = quantile_records
                config["updated_at_utc"] = utc_now()
                atomic_write_json(config_path, config)

    observed_arms = {
        str(row["arm"]) for rows in (scalar_rows, task_rows, pool_rows) for row in rows
    }
    unexpected_arms = observed_arms - set(active_arms)
    if unexpected_arms:
        raise RuntimeError(
            "Stream-B contains rows outside the frozen P1-A7 roster: "
            f"{sorted(unexpected_arms)}"
        )
    for arm in active_arms:
        validate_coverage(
            scalar_rows=[row for row in scalar_rows if str(row["arm"]) == arm],
            task_rows=[row for row in task_rows if str(row["arm"]) == arm],
            pool_rows=[row for row in pool_rows if str(row["arm"]) == arm],
            arms=(arm,),
            seeds=arm_seeds[arm],
            task_count=TASK_COUNT,
            total_steps=TOTAL_STEPS,
            pool_size=POOL_SIZE,
        )
    runtime = prior_runtime + (time.perf_counter() - started)
    controls = grade_controls(pool_rows)
    config["runtime_seconds"] = runtime
    config["updated_at_utc"] = utc_now()
    config["controls"] = controls
    atomic_write_json(config_path, config)
    report = build_report(
        pool_rows,
        task_rows,
        scalar_rows,
        seeds,
        schedule,
        runtime,
        deployed_configs,
        tuned_configs,
        revisit_gaps,
        active_arms,
        arm_seeds,
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
