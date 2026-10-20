"""Run the Post-Phase-1 liveness preflight and recurrent-validation grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

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
    fixed_test_subset,
    refresh_global_c,
)
from model import MLP300
from phase1_arms import build_arm
from phase1_retention_validation import (
    STEPS_PER_TASK,
    TEST_SUBSET_SEED,
)
from phase1_stream_common import (
    StepsToThreshold,
    evaluate_permutation,
    permute_batch,
    cap_stream_statistics,
    training_step,
)
from post_phase1_common import (
    GRID_DIR,
    GRID_RESULTS,
    GRID_TASKS,
    LIVENESS_RECORD,
    METHODS,
    UPGD_CORRECTED_DIR,
    V2_ISOLATION,
    assert_post_output,
    atomic_write_json,
    canonical_json,
    first_config,
    post_guard,
    grid_knobs,
    parameter_vector,
    post_arm_config,
    retention_grids,
    validation_protocol,
    values_for_knob,
)


GRID_POOL = GRID_DIR / "post_grid_pool.csv"
GRID_RUN_CONFIG = GRID_DIR / "post_run_config.json"
REPLAY_SHORT_BATCH_AMENDMENT = (
    Path(__file__).resolve().parent
    / "post_phase1_amendment_POST-P1-A1.md"
)
CAP_LIVENESS_AMENDMENT = (
    Path(__file__).resolve().parent
    / "post_phase1_amendment_POST-P1-A2.md"
)
CORRECTED_RESULTS = UPGD_CORRECTED_DIR / "post_grid_results.csv"
CORRECTED_TASKS = UPGD_CORRECTED_DIR / "post_grid_tasks.csv"
CORRECTED_POOL = UPGD_CORRECTED_DIR / "post_grid_pool.csv"
CORRECTED_RUN_CONFIG = UPGD_CORRECTED_DIR / "post_run_config.json"
EDGE_PLAN = GRID_DIR / "post_edge_extension_plan.json"

CELL_BASE_FIELDS = (
    "method",
    "config_index",
    "config_json",
    "provenance",
    "validation_retention_metric",
    "validation_plasticity",
    "diverged",
    "status",
    "runtime_seconds",
)
CONFIG_FIELDS = (
    "lr",
    "weight_decay",
    "replay_count",
    "buffer_capacity",
    "lambda",
    "gamma",
    "sigma",
    "beta_utility",
    "k",
    "T_a",
    "T_s",
    "T_m",
    "retention",
)
CELL_FIELDS = CELL_BASE_FIELDS + CONFIG_FIELDS
TASK_FIELDS = (
    "method",
    "config_index",
    "config_json",
    "task",
    "current_pool_index",
    "task_end_accuracy",
    "steps_to_085",
    "cap_diagnostics_json",
)
POOL_FIELDS = (
    "method",
    "config_index",
    "task",
    "pool_index",
    "accuracy",
    "is_current",
)


def corrected_upgd_grid() -> list[dict[str, Any]]:
    return [
        {
            "lr": 0.03,
            "weight_decay": wd,
            "sigma": 0.01,
            "beta_utility": 0.9999,
        }
        for wd in (0.01, 1e-3, 1e-4, 3.3e-4)
    ]


def _construct(method: str, model: MLP300, config: Mapping[str, Any]):
    arm, direct = post_arm_config(method, config)
    return build_arm(arm, model, direct, 42, torch.device("cpu"))


def _liveness_distance(
    method: str,
    low: Mapping[str, Any],
    high: Mapping[str, Any],
    steps: int,
    *,
    prime_cap_state: bool = False,
) -> float:
    seed_everything(42)
    template = MLP300(input_units=784, output_units=10)
    initial = template.state_dict()
    batches = []
    generator = torch.Generator().manual_seed(91_337)
    for _ in range(steps):
        batches.append(
            (
                torch.randn(BATCH_SIZE, 1, 28, 28, generator=generator),
                torch.randint(0, 10, (BATCH_SIZE,), generator=generator),
            )
        )
    vectors = []
    for config in (low, high):
        model = MLP300(input_units=784, output_units=10)
        model.load_state_dict(initial)
        optimizer, hooks = _construct(method, model, config)
        if prime_cap_state:
            if not hooks["is_cap"]:
                raise RuntimeError("CAP state priming requested for a non-CAP arm")
            for group in optimizer.param_groups:
                group["disable_box_clamp"] = True
            prime_inputs = torch.randn(
                BATCH_SIZE, 1, 28, 28,
                generator=torch.Generator().manual_seed(81_337),
            )
            prime_targets = torch.randint(
                0, 10, (BATCH_SIZE,),
                generator=torch.Generator().manual_seed(81_338),
            )
            prime_loss, _ = training_step(
                model=model,
                optimizer=optimizer,
                hooks=hooks,
                inputs=prime_inputs,
                targets=prime_targets,
            )
            if not torch.isfinite(prime_loss):
                raise RuntimeError(f"non-finite CAP priming loss for {method}")
            for group in optimizer.param_groups:
                group.pop("disable_box_clamp", None)
                if not bool(group.get("protect", True)):
                    continue
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter)
                    if not state or "anchor" not in state or "imp" not in state:
                        raise RuntimeError("CAP priming did not initialize state")
                    state["anchor"].copy_(parameter.detach())
                    state["imp"].fill_(1.0)
        for index, (inputs, targets) in enumerate(batches, start=1):
            loss, _ = training_step(
                model=model,
                optimizer=optimizer,
                hooks=hooks,
                inputs=inputs,
                targets=targets,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite preflight loss for {method}")
            if hooks["is_cap"] and index % C_REFRESH_STEPS == 0:
                refresh_global_c(optimizer)
            # EWC's lambda and gamma are live only after consolidation. Two
            # synthetic boundaries expose both accumulation and stiffness.
            if hooks["on_boundary"] is not None and index in (2, 5):
                hooks["on_boundary"](model)
        vectors.append(parameter_vector(model))
    return float(torch.linalg.vector_norm(vectors[0] - vectors[1]).item())


def run_liveness_preflight() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for method in METHODS:
        for knob in grid_knobs(method):
            values = values_for_knob(method, knob)
            low = first_config(method)
            high = dict(low)
            low[knob], high[knob] = values[0], values[-1]
            steps = 10
            distance = _liveness_distance(method, low, high, steps)
            extended = False
            state_primed = False
            if distance <= 0.0 and knob == "weight_decay":
                steps = 200
                extended = True
                distance = _liveness_distance(method, low, high, steps)
            if (
                distance <= 0.0
                and method in {"cap_release", "cap_ratchet"}
                and knob in {"k", "T_a", "T_m"}
            ):
                state_primed = True
                distance = _liveness_distance(
                    method,
                    low,
                    high,
                    steps,
                    prime_cap_state=True,
                )
                if distance <= 0.0:
                    steps = 200
                    extended = True
                    distance = _liveness_distance(
                        method,
                        low,
                        high,
                        steps,
                        prime_cap_state=True,
                    )
            record = {
                "method": method,
                "knob": knob,
                "low": values[0],
                "high": values[-1],
                "steps": steps,
                "extension_used": extended,
                "state_primed": state_primed,
                "state_priming_amendment": (
                    "POST-P1-A2" if state_primed else None
                ),
                "l2_distance": distance,
                "status": "PASS" if distance > 0.0 else "KNOB-INERT",
            }
            results.append(record)
            print(
                f"{record['status']} {method} {knob} "
                f"distance={distance:.12g} steps={steps} "
                f"state_primed={state_primed}"
            )
            if distance <= 0.0:
                raise RuntimeError(f"KNOB-INERT {method} {knob}")
    payload = {
        "schema_version": 3,
        "isolation": V2_ISOLATION,
        "status": "PASS",
        "results": results,
    }
    atomic_write_json(LIVENESS_RECORD, payload)
    return payload


def require_liveness_pass() -> None:
    if not LIVENESS_RECORD.exists():
        raise RuntimeError("knob-liveness preflight has not run")
    payload = json.loads(LIVENESS_RECORD.read_text(encoding="utf-8"))
    expected = {
        (method, knob) for method in METHODS for knob in grid_knobs(method)
    }
    observed = {
        (str(row["method"]), str(row["knob"]))
        for row in payload.get("results", [])
        if row.get("status") == "PASS" and float(row.get("l2_distance", 0)) > 0
    }
    if (
        payload.get("isolation") != V2_ISOLATION
        or payload.get("status") != "PASS"
        or observed != expected
    ):
        raise RuntimeError("KNOB-INERT preflight coverage is incomplete")


def run_validation_cell(
    *,
    method: str,
    config_index: int,
    config: Mapping[str, Any],
    provenance: str,
    mnist_train: Any,
    test_loader: Any,
    permutations_cpu: Mapping[int, torch.Tensor],
    schedule: Sequence[int],
    device: torch.device,
    num_workers: int,
    pool_size: int,
    task_count: int,
    total_steps: int,
    final_block_start_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_everything(42)
    model = MLP300(input_units=784, output_units=10).to(device)
    arm, direct = post_arm_config(method, config)
    optimizer, hooks = build_arm(arm, model, direct, 42, device)
    loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=42,
    )
    batches = repeat_batches(loader)
    permutations = {key: value.to(device) for key, value in permutations_cpu.items()}
    threshold = StepsToThreshold()
    task_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    config_json = canonical_json(config)
    status = "OK"
    started = time.perf_counter()

    for step in range(1, total_steps + 1):
        task = ((step - 1) // STEPS_PER_TASK) + 1
        within_task = ((step - 1) % STEPS_PER_TASK) + 1
        current_index = int(schedule[task - 1])
        inputs, targets = next(batches)
        loss, accuracy = training_step(
            model=model,
            optimizer=optimizer,
            hooks=hooks,
            inputs=permute_batch(
                inputs.to(device, non_blocking=True), permutations[current_index]
            ),
            targets=targets.to(device, non_blocking=True),
        )
        if not bool(torch.isfinite(loss).item()):
            status = "EWC-DIVERGED" if method == "ewc_oracle" else "DIVERGED"
            break
        if hooks["is_cap"] and step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        threshold.update(accuracy, within_task)
        if within_task != STEPS_PER_TASK:
            continue

        current_accuracy = None
        for pool_index in range(pool_size):
            evaluated = evaluate_permutation(
                model, test_loader, permutations[pool_index], device
            )
            is_current = pool_index == current_index
            if is_current:
                current_accuracy = evaluated
            pool_rows.append(
                {
                    "method": method,
                    "config_index": config_index,
                    "task": task,
                    "pool_index": pool_index,
                    "accuracy": evaluated,
                    "is_current": is_current,
                }
            )
        diagnostics = (
            cap_stream_statistics(optimizer) if hooks["is_cap"] else {}
        )
        task_rows.append(
            {
                "method": method,
                "config_index": config_index,
                "config_json": config_json,
                "task": task,
                "current_pool_index": current_index,
                "task_end_accuracy": current_accuracy,
                "steps_to_085": threshold.reached or "",
                "cap_diagnostics_json": json.dumps(
                    diagnostics, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        if hooks["on_boundary"] is not None:
            hooks["on_boundary"](model)
        threshold.reset()
        print(
            f"  {method}[{config_index}] task={task}/{task_count} "
            f"current_accuracy={float(current_accuracy):.4f}"
        )

    retained = [
        float(row["accuracy"])
        for row in pool_rows
        if int(row["task"]) == task_count and not bool(row["is_current"])
    ]
    plasticity = [
        float(row["task_end_accuracy"])
        for row in task_rows
        if int(row["task"]) >= final_block_start_task
    ]
    expected_retained = pool_size - 1
    complete = (
        status == "OK"
        and len(retained) == expected_retained
        and len(plasticity) == pool_size
    )
    row = {
        "method": method,
        "config_index": config_index,
        "config_json": config_json,
        "provenance": provenance,
        "validation_retention_metric": mean(retained) if complete else "",
        "validation_plasticity": mean(plasticity) if complete else "",
        "diverged": not complete,
        "status": status if not complete else "OK",
        "runtime_seconds": time.perf_counter() - started,
        **{field: config.get(field, "") for field in CONFIG_FIELDS},
    }
    del model, optimizer, loader, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return task_rows, pool_rows, row


def _write_campaign(
    cell_path: Path,
    task_path: Path,
    pool_path: Path,
    cells: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(cell_path, cells, CELL_FIELDS)
    write_csv(task_path, tasks, TASK_FIELDS)
    write_csv(pool_path, pool, POOL_FIELDS)


def run_campaign(
    *,
    cells_to_run: Sequence[tuple[str, int, Mapping[str, Any], str]],
    cell_path: Path,
    task_path: Path,
    pool_path: Path,
    run_config_path: Path,
    data_dir: Path,
    device: torch.device,
    num_workers: int,
) -> None:
    for path in (cell_path, task_path, pool_path, run_config_path):
        assert_post_output(path)
    cells = read_csv(cell_path) if cell_path.exists() else []
    tasks = read_csv(task_path) if task_path.exists() else []
    pool = read_csv(pool_path) if pool_path.exists() else []
    existing_run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else {}
    )
    if (cells or tasks or pool) and existing_run_config.get("isolation") != V2_ISOLATION:
        raise RuntimeError(
            "pre-isolation grid output cannot be resumed by PP1-A1(C): "
            f"{run_config_path.parent}"
        )
    complete = {
        (str(row["method"]), str(row["config_json"])) for row in cells
    }
    mnist_train, mnist_test = load_mnist(data_dir)
    test_loader = make_loader(
        fixed_test_subset(mnist_test, TEST_SUBSET_SEED),
        EVALUATION_BATCH_SIZE,
        device,
        num_workers,
        shuffle=False,
    )
    protocol = validation_protocol()
    permutations = {
        index: fixed_permutation(784, protocol["pool_seed_base"] + index)
        for index in range(protocol["pool_size"])
    }
    schedule = protocol["schedule"]
    atomic_write_json(
        run_config_path,
        {
            "schema_version": 3,
            "amendment": "PP1-A1 (2026-08-02, post-results)",
            "isolation": V2_ISOLATION,
            "grading_suppressed": [
                "E4b",
                "combined-E4",
                "E5-norm",
            ],
            "validation_seed": 42,
            "loader_seed": 42,
            "schedule_seed": protocol["schedule_seed"],
            "tasks": protocol["task_count"],
            "steps_per_task": STEPS_PER_TASK,
            "total_steps": protocol["total_steps"],
            "pool_size": protocol["pool_size"],
            "stream_pool_size": protocol["stream_pool_size"],
            "pool_permutation_seeds": list(
                range(
                    protocol["pool_seed_base"],
                    protocol["pool_seed_base"] + protocol["pool_size"],
                )
            ),
            "schedule_type": "two-shuffled-blocks",
            "schedule": schedule,
            "selection_metrics": {
                "retention": "final boundary mean over non-current pool members",
                "plasticity": "mean task_end_accuracy over final block",
            },
            "test_subset_seed": TEST_SUBSET_SEED,
            "implementation_amendments": [
                {
                    "id": "POST-P1-A1",
                    "path": str(REPLAY_SHORT_BATCH_AMENDMENT),
                    "sha256": hashlib.sha256(
                        REPLAY_SHORT_BATCH_AMENDMENT.read_bytes()
                    ).hexdigest(),
                    "summary": (
                        "On a short loader batch, cap effective replay_count "
                        "at actual batch size minus one so a fresh example "
                        "always remains; full batches use the configured count."
                    ),
                },
                {
                    "id": "POST-P1-A2",
                    "path": str(CAP_LIVENESS_AMENDMENT),
                    "sha256": hashlib.sha256(
                        CAP_LIVENESS_AMENDMENT.read_bytes()
                    ).hexdigest(),
                    "summary": (
                        "Retry exactly-zero CAP state-dependent knob checks "
                        "from identical primed protection state."
                    ),
                },
            ],
        },
    )
    for method, index, config, provenance in cells_to_run:
        key = (method, canonical_json(config))
        if key in complete:
            print(f"Skipping completed grid cell {method}[{index}]")
            continue
        new_tasks, new_pool, new_cell = run_validation_cell(
            method=method,
            config_index=index,
            config=config,
            provenance=provenance,
            mnist_train=mnist_train,
            test_loader=test_loader,
            permutations_cpu=permutations,
            schedule=schedule,
            device=device,
            num_workers=num_workers,
            pool_size=protocol["pool_size"],
            task_count=protocol["task_count"],
            total_steps=protocol["total_steps"],
            final_block_start_task=protocol["final_block_start_task"],
        )
        tasks.extend(new_tasks)
        pool.extend(new_pool)
        cells.append(new_cell)
        _write_campaign(cell_path, task_path, pool_path, cells, tasks, pool)
        complete.add(key)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "corrected-upgd",
            "retention-grid",
            "edge-extension",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "preflight":
        with post_guard():
            run_liveness_preflight()
        return 0
    with post_guard():
        require_liveness_pass()
        if args.action == "corrected-upgd":
            cells = [
                ("upgd", index, config, "corrected_upgd")
                for index, config in enumerate(corrected_upgd_grid())
            ]
            run_campaign(
                cells_to_run=cells,
                cell_path=CORRECTED_RESULTS,
                task_path=CORRECTED_TASKS,
                pool_path=CORRECTED_POOL,
                run_config_path=CORRECTED_RUN_CONFIG,
                data_dir=args.data_dir,
                device=torch.device(args.device),
                num_workers=args.num_workers,
            )
            return 0
        if args.action == "edge-extension":
            if not EDGE_PLAN.exists():
                raise RuntimeError("edge-extension plan does not exist")
            plan = json.loads(EDGE_PLAN.read_text(encoding="utf-8"))
            cells = []
            for method in METHODS:
                for index, config in enumerate(
                    plan["methods"][method]["new_configs"], start=10_000
                ):
                    cells.append((method, index, config, "edge_extension"))
            run_campaign(
                cells_to_run=cells,
                cell_path=GRID_RESULTS,
                task_path=GRID_TASKS,
                pool_path=GRID_POOL,
                run_config_path=GRID_RUN_CONFIG,
                data_dir=args.data_dir,
                device=torch.device(args.device),
                num_workers=args.num_workers,
            )
            return 0
        grids = retention_grids()
        cells = [
            (method, index, config, "registered_grid")
            for method in METHODS
            for index, config in enumerate(grids[method])
        ]
        run_campaign(
            cells_to_run=cells,
            cell_path=GRID_RESULTS,
            task_path=GRID_TASKS,
            pool_path=GRID_POOL,
            run_config_path=GRID_RUN_CONFIG,
            data_dir=args.data_dir,
            device=torch.device(args.device),
            num_workers=args.num_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
