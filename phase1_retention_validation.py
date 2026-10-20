"""P1-A4/A5 recurrent validation and v7 dual-endpoint artifact builder.

This program never reruns or rewrites the completed v4 plasticity campaign.
It preserves that campaign byte-for-byte, runs every registered grid cell on
the disjoint recurrent validation stream, and only then replaces the canonical
downstream manifest with the P1-A5/v7 schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
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
    COMPUTE_FLOOR_SEEDS,
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
    P1_A3_DISCLOSURE,
    P1_A4_DISCLOSURE,
    P1_A5_DISCLOSURE,
    PHASE1_AMENDMENT,
    RETENTION_FOCUSED_LABEL,
    RETENTION_FOCUSED_SUFFIX,
    TUNED_METHODS,
    UPGD_AMENDMENT,
    UPGD_DEFAULTS,
    UPGD_TASK_CONFIG_SOURCE,
    build_arm,
    cap_release_matches_core,
)
from phase1_stream_common import (
    StepsToThreshold,
    evaluate_permutation,
    permute_batch,
    training_step,
)
from phase1_tuning import (
    EWC_DISCLOSURE,
    PRE_AMENDMENT_SMOKE,
    REPLAY_DISCLOSURE,
    TUNING_DISCLOSURE,
    registered_grids,
)


VALIDATION_SEED = 42
POOL_SIZE = 5
POOL_SEED_BASE = 300_001
SCHEDULE_SEED = 556
TASK_COUNT = 30
STEPS_PER_TASK = 5_000
TOTAL_STEPS = TASK_COUNT * STEPS_PER_TASK
RETENTION_START_TASK = 11
PLASTICITY_TOLERANCE = 0.02
TEST_SUBSET_SEED = 999

TASK_FIELDS = (
    "method",
    "config_index",
    "config_json",
    "task",
    "current_pool_index",
    "task_end_accuracy",
    "steps_to_085",
)
POOL_FIELDS = (
    "method",
    "config_index",
    "task",
    "pool_index",
    "accuracy",
    "is_current",
)
CELL_FIELDS = (
    "method",
    "config_index",
    "config_json",
    "validation_retention_metric",
    "runtime_seconds",
    "status",
)


def validation_schedule() -> list[int]:
    generator = torch.Generator().manual_seed(SCHEDULE_SEED)
    return torch.cat(
        [torch.randperm(POOL_SIZE, generator=generator) for _ in range(6)]
    ).tolist()


def sha256_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _copy_exact(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(
                f"preserved v4 evidence differs from source: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def preserve_v4_artifacts(
    *, legacy_dir: Path, v4_dir: Path, canonical_path: Path
) -> tuple[Path, dict[str, str]]:
    """Copy the completed v4 campaign once and thereafter only verify it."""
    if legacy_dir.exists():
        v4_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(item for item in legacy_dir.rglob("*") if item.is_file()):
            _copy_exact(source, v4_dir / source.relative_to(legacy_dir))
    elif not v4_dir.exists():
        raise FileNotFoundError(
            "completed v4 tuning directory is missing; expected either "
            f"{legacy_dir} or {v4_dir}"
        )

    artifact_path = v4_dir / "phase1_tuned_configs.json"
    if not artifact_path.exists():
        if not canonical_path.exists():
            raise FileNotFoundError("completed v4 tuned-config artifact is missing")
        _copy_exact(canonical_path, artifact_path)

    for required in (
        "tuning_results.csv",
        "report.md",
        "run_config.json",
        "phase1_tuned_configs.json",
    ):
        if not (v4_dir / required).exists():
            raise FileNotFoundError(f"preserved v4 artifact is incomplete: {required}")

    hashes = sha256_files(v4_dir)
    if not hashes:
        raise RuntimeError("preserved v4 directory contains no files")
    return artifact_path, hashes


def validate_v4_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 2 or payload.get("amendment") != UPGD_AMENDMENT:
        raise ValueError("preserved primary artifact is not the completed P1-A3/v4 run")
    provenance = payload.get("upgd", {})
    if (
        provenance.get("task_config_source") != UPGD_TASK_CONFIG_SOURCE
        or provenance.get("resolved_default_kwargs") != UPGD_DEFAULTS
    ):
        raise ValueError(
            "P1-A4(f): v4 UPGD cells do not use the P1-A3 published task config"
        )
    grids = registered_grids()
    if sum(len(grid) for grid in grids.values()) != 41:
        raise RuntimeError("registered grid no longer contains exactly 41 cells")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping):
        raise KeyError("v4 artifact is missing methods")
    for method in TUNED_METHODS:
        entry = methods.get(method)
        if not isinstance(entry, Mapping):
            raise KeyError(f"v4 artifact is missing method {method!r}")
        tried = entry.get("tried_configs")
        if not isinstance(tried, list) or len(tried) != len(grids[method]):
            raise ValueError(f"v4 artifact has incomplete grid for {method!r}")
        by_index = {int(item["config_index"]): item for item in tried}
        for index, expected in enumerate(grids[method]):
            item = by_index.get(index)
            if item is None or item.get("config") != expected:
                raise ValueError(f"v4 grid mismatch at {method}[{index}]")
            if item.get("status") != "OK" or item.get("selection_metric") is None:
                raise ValueError(f"v4 cell is not a valid completed result: {method}[{index}]")
        winner = int(entry["winner_config_index"])
        if entry.get("config") != grids[method][winner]:
            raise ValueError(f"v4 winner does not match its grid cell for {method!r}")


def assert_stream_b_unseen(stream_b_dir: Path) -> None:
    for name in ("streamb_scalars.csv", "streamb_tasks.csv", "streamb_pool.csv"):
        path = stream_b_dir / name
        if path.exists() and read_csv(path):
            raise RuntimeError(
                "P1-A5 is valid only before any Stream-B cell starts; "
                f"observed rows in {path}"
            )
    checkpoint_dir = stream_b_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        raise RuntimeError("P1-A5 cannot start after a Stream-B checkpoint exists")


def _direct_config(method: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {"methods": {method: {"config": dict(config)}}}


def run_validation_cell(
    *,
    method: str,
    config_index: int,
    config: Mapping[str, Any],
    mnist_train,
    test_loader,
    permutations_cpu: Mapping[int, torch.Tensor],
    schedule: Sequence[int],
    device: torch.device,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_everything(VALIDATION_SEED)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = build_arm(
        method,
        model,
        _direct_config(method, config),
        VALIDATION_SEED,
        device,
    )
    loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=VALIDATION_SEED,
    )
    batches = repeat_batches(loader)
    permutations = {index: value.to(device) for index, value in permutations_cpu.items()}
    threshold = StepsToThreshold()
    task_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    status = "OK"
    started = time.perf_counter()
    config_json = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))

    for step in range(1, TOTAL_STEPS + 1):
        task = ((step - 1) // STEPS_PER_TASK) + 1
        within_task = ((step - 1) % STEPS_PER_TASK) + 1
        current_index = int(schedule[task - 1])
        inputs, targets = next(batches)
        inputs = permute_batch(
            inputs.to(device, non_blocking=True), permutations[current_index]
        )
        targets = targets.to(device, non_blocking=True)
        loss, accuracy = training_step(
            model=model,
            optimizer=optimizer,
            hooks=hooks,
            inputs=inputs,
            targets=targets,
        )
        if not bool(torch.isfinite(loss).item()):
            status = "EWC-DIVERGED" if method == "ewc_oracle" else "DIVERGED"
            break
        if bool(hooks["is_cap"]) and step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        threshold.update(accuracy, within_task)

        if within_task == STEPS_PER_TASK:
            current_accuracy = None
            for pool_index in range(POOL_SIZE):
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
            task_rows.append(
                {
                    "method": method,
                    "config_index": config_index,
                    "config_json": config_json,
                    "task": task,
                    "current_pool_index": current_index,
                    "task_end_accuracy": current_accuracy,
                    "steps_to_085": threshold.reached if threshold.reached is not None else "",
                }
            )
            if hooks["on_boundary"] is not None:
                hooks["on_boundary"](model)
            threshold.reset()
            print(
                f"  {method}[{config_index}] task={task}/{TASK_COUNT} "
                f"current_accuracy={float(current_accuracy):.4f}"
            )

    runtime = time.perf_counter() - started
    retained = [
        float(row["accuracy"])
        for row in pool_rows
        if int(row["task"]) >= RETENTION_START_TASK and not bool(row["is_current"])
    ]
    expected = (TASK_COUNT - RETENTION_START_TASK + 1) * (POOL_SIZE - 1)
    metric = mean(retained) if status == "OK" and len(retained) == expected else None
    cell = {
        "method": method,
        "config_index": config_index,
        "config_json": config_json,
        "validation_retention_metric": metric if metric is not None else "",
        "runtime_seconds": runtime,
        "status": status,
    }
    del model, optimizer, loader, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return task_rows, pool_rows, cell


def _remove_cell(rows: list[dict], method: str, index: int) -> None:
    rows[:] = [
        row
        for row in rows
        if not (
            str(row["method"]) == method
            and int(float(row["config_index"])) == index
        )
    ]


def _write_validation(
    output_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(output_dir / "validation_tasks.csv", tasks, TASK_FIELDS)
    write_csv(output_dir / "validation_pool.csv", pool, POOL_FIELDS)
    write_csv(output_dir / "validation_cells.csv", cells, CELL_FIELDS)


def validate_validation_results(
    *,
    tasks: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> None:
    """Require exact, nonduplicated evidence for every successful grid cell."""
    grids = registered_grids()
    expected_cells = {
        (method, index)
        for method in TUNED_METHODS
        for index in range(len(grids[method]))
    }
    actual_cells = [
        (str(row["method"]), int(float(row["config_index"]))) for row in cells
    ]
    if len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
        raise RuntimeError(
            "recurrent-validation cell coverage is incomplete or duplicated"
        )

    task_by_cell: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    pool_by_cell: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in tasks:
        key = (str(row["method"]), int(float(row["config_index"])))
        task_by_cell.setdefault(key, []).append(row)
    for row in pool:
        key = (str(row["method"]), int(float(row["config_index"])))
        pool_by_cell.setdefault(key, []).append(row)
    if set(task_by_cell) - expected_cells or set(pool_by_cell) - expected_cells:
        raise RuntimeError("recurrent-validation outputs contain an unknown cell")

    schedule = validation_schedule()
    for cell in cells:
        method = str(cell["method"])
        index = int(float(cell["config_index"]))
        key = (method, index)
        config_json = json.dumps(
            grids[method][index], sort_keys=True, separators=(",", ":")
        )
        if str(cell["config_json"]) != config_json:
            raise RuntimeError(f"validation config mismatch for {method}[{index}]")
        if str(cell["status"]) != "OK":
            continue

        cell_tasks = task_by_cell.get(key, [])
        cell_pool = pool_by_cell.get(key, [])
        task_keys = {int(float(row["task"])) for row in cell_tasks}
        pool_keys = {
            (int(float(row["task"])), int(float(row["pool_index"])))
            for row in cell_pool
        }
        if (
            len(cell_tasks) != TASK_COUNT
            or task_keys != set(range(1, TASK_COUNT + 1))
            or len(cell_pool) != TASK_COUNT * POOL_SIZE
            or pool_keys
            != {
                (task, pool_index)
                for task in range(1, TASK_COUNT + 1)
                for pool_index in range(POOL_SIZE)
            }
        ):
            raise RuntimeError(
                f"successful validation cell has incomplete evidence: {method}[{index}]"
            )
        for row in cell_tasks:
            task = int(float(row["task"]))
            if (
                str(row["config_json"]) != config_json
                or int(float(row["current_pool_index"])) != schedule[task - 1]
            ):
                raise RuntimeError(
                    f"validation task metadata mismatch for {method}[{index}]"
                )
        retained = []
        for row in cell_pool:
            task = int(float(row["task"]))
            pool_index = int(float(row["pool_index"]))
            is_current = str(row["is_current"]).lower() in {"true", "1"}
            expected_current = pool_index == schedule[task - 1]
            if is_current != expected_current:
                raise RuntimeError(
                    f"validation current-pool flag mismatch for {method}[{index}]"
                )
            if task >= RETENTION_START_TASK and not is_current:
                retained.append(float(row["accuracy"]))
        expected_metric = mean(retained)
        observed_metric = float(cell["validation_retention_metric"])
        if abs(observed_metric - expected_metric) > 1e-12:
            raise RuntimeError(
                f"validation retention metric mismatch for {method}[{index}]"
            )


def build_v7_artifact(
    *,
    v4: Mapping[str, Any],
    validation_cells: Sequence[Mapping[str, Any]],
    v4_hashes: Mapping[str, str],
    v4_dir: Path,
) -> dict[str, Any]:
    grids = registered_grids()
    validation = {
        (str(row["method"]), int(float(row["config_index"]))): row
        for row in validation_cells
    }
    methods: dict[str, Any] = {}
    sensitivity_arms: dict[str, Any] = {}
    for method in TUNED_METHODS:
        v4_entry = v4["methods"][method]
        v4_tried = {
            int(item["config_index"]): item for item in v4_entry["tried_configs"]
        }
        tried = []
        best_plasticity = max(
            float(item["selection_metric"]) for item in v4_tried.values()
        )
        for index, config in enumerate(grids[method]):
            row = validation.get((method, index))
            if row is None:
                raise RuntimeError(f"missing recurrent validation cell {method}[{index}]")
            retention = (
                float(row["validation_retention_metric"])
                if row["validation_retention_metric"] != ""
                else None
            )
            plasticity = float(v4_tried[index]["selection_metric"])
            eligible = (
                row["status"] == "OK"
                and retention is not None
                and plasticity >= best_plasticity - PLASTICITY_TOLERANCE
            )
            tried.append(
                {
                    "config_index": index,
                    "config": dict(config),
                    "plasticity_metric": plasticity,
                    "validation_retention_metric": retention,
                    "eligible_for_retention_focused": eligible,
                    "validation_runtime_seconds": float(row["runtime_seconds"]),
                    "validation_status": str(row["status"]),
                }
            )
        eligible_rows = [item for item in tried if item["eligible_for_retention_focused"]]
        if not eligible_rows:
            raise RuntimeError(f"no eligible retention-focused config for {method}")
        retained = min(
            eligible_rows,
            key=lambda item: (
                -float(item["validation_retention_metric"]),
                -float(item["plasticity_metric"]),
                int(item["config_index"]),
            ),
        )
        primary_index = int(v4_entry["winner_config_index"])
        primary_config = dict(v4_entry["config"])
        retained_config = dict(retained["config"])
        coincide = primary_config == retained_config
        methods[method] = {
            # Compatibility aliases always mean the v4-primary endpoint.
            "config": primary_config,
            "selection_metric": float(v4_entry["selection_metric"]),
            "winner_config_index": primary_index,
            "primary": {
                "config": primary_config,
                "plasticity_metric": float(v4_entry["selection_metric"]),
                "tie_break_metric": float(v4_entry["tie_break_metric"]),
                "config_index": primary_index,
            },
            "retention_focused": {
                "config": retained_config,
                "plasticity_metric": float(retained["plasticity_metric"]),
                "validation_retention_metric": float(
                    retained["validation_retention_metric"]
                ),
                "config_index": int(retained["config_index"]),
            },
            "endpoints_coincide": coincide,
            "tried_configs": tried,
        }
        if not coincide:
            arm = method + RETENTION_FOCUSED_SUFFIX
            sensitivity_arms[arm] = {
                "method": method,
                "endpoint": "retention_focused",
                "label": RETENTION_FOCUSED_LABEL,
                "seeds": list(
                    FULL_SEEDS if method == "cap_release" else COMPUTE_FLOOR_SEEDS
                ),
                "config": retained_config,
            }

    primary_cap = methods["cap_release"]["primary"]["config"]
    artifact: dict[str, Any] = {
        "schema_version": 7,
        "created_at_utc": utc_now(),
        "amendment": PHASE1_AMENDMENT,
        "amendments": ["P1-A3", "P1-A4", "P1-A5"],
        "p1_a3_disclosure": P1_A3_DISCLOSURE,
        "p1_a4_disclosure": P1_A4_DISCLOSURE,
        "p1_a5_disclosure": P1_A5_DISCLOSURE,
        "primary_analysis": "completed v4 plasticity-only winners",
        "flexibility_analysis": "descriptive endpoint-matched sensitivity",
        "flexibility_framing": (
            "P1-A5 tests whether CAP offers greater hyperparameter-controlled "
            "plasticity–retention flexibility"
        ),
        "v4_artifact_directory": str(v4_dir),
        "v4_file_sha256": dict(v4_hashes),
        "tuning_seed": VALIDATION_SEED,
        "tuning_disclosure": TUNING_DISCLOSURE,
        "ewc_disclosure": EWC_DISCLOSURE,
        "adam_replay_label": REPLAY_DISCLOSURE,
        "upgd": dict(v4["upgd"]),
        "pre_amendment_upgd_smoke_gate": v4.get(
            "pre_amendment_upgd_smoke_gate", PRE_AMENDMENT_SMOKE
        ),
        "amended_upgd_smoke_gate": v4.get("amended_upgd_smoke_gate", {}),
        "validation_protocol": {
            "fresh_model_per_config": True,
            "seed": VALIDATION_SEED,
            "pool_permutation_seeds": list(
                range(POOL_SEED_BASE, POOL_SEED_BASE + POOL_SIZE)
            ),
            "schedule_seed": SCHEDULE_SEED,
            "schedule": validation_schedule(),
            "balanced_blocks": 6,
            "tasks": TASK_COUNT,
            "steps_per_task": STEPS_PER_TASK,
            "retention_boundaries": [RETENTION_START_TASK, TASK_COUNT],
            "plasticity_tolerance": PLASTICITY_TOLERANCE,
        },
        "methods": methods,
        "sensitivity_arms": sensitivity_arms,
        "derived_configs": {
            "cap_ratchet": {**primary_cap, "retention": "ratchet"},
            "cap_leak": {**primary_cap, "retention": "leak"},
        },
    }
    artifact["stream_a_release_arm"] = (
        "cap_release_10000" if cap_release_matches_core(artifact) else "cap_release_tuned"
    )
    artifact["kill_gate_retune_arm"] = (
        "cap_release_retention_focused"
        if "cap_release_retention_focused" in sensitivity_arms
        else None
    )
    return artifact


def build_report(artifact: Mapping[str, Any], runtime: float) -> str:
    lines = [
        "# Phase 1 recurrent validation and dual-endpoint selection",
        "",
        "## Protocol header",
        "",
        f"Forced deviations: {P1_A3_DISCLOSURE} {P1_A4_DISCLOSURE} {P1_A5_DISCLOSURE}",
        f"Seed set run: ({VALIDATION_SEED},).",
        (
            "Settings: fresh MLP-300 per config; batch 128; five fixed "
            "input-permutation pool entries (seeds 300001-300005); six "
            "balanced shuffled blocks from schedule seed 556; 30 tasks x "
            "5,000 steps; subset-999 evaluation of all five pool entries "
            "at every boundary."
        ),
        EWC_DISCLOSURE,
        f"adam_replay: {REPLAY_DISCLOSURE}.",
        TUNING_DISCLOSURE,
        "The completed v4 winners remain primary; this campaign is post-v4-tuning and pre-Stream-B.",
        f"Validation schedule: `{artifact['validation_protocol']['schedule']}`.",
        "Each five-task block contains every pool member exactly once.",
        "",
        "| Method | Sensitivity designation | Primary config | Retention-focused config | P metric | R validation retention | Coincide |",
        "|---|---|---|---|---:|---:|:---:|",
    ]
    for method in TUNED_METHODS:
        entry = artifact["methods"][method]
        arm = method + RETENTION_FOCUSED_SUFFIX
        designation = (
            "coincident (no sensitivity arm)"
            if entry["endpoints_coincide"]
            else f"{arm} ({RETENTION_FOCUSED_LABEL})"
        )
        lines.append(
            f"| {method} | {designation} | "
            f"`{json.dumps(entry['primary']['config'], sort_keys=True)}` | "
            f"`{json.dumps(entry['retention_focused']['config'], sort_keys=True)}` | "
            f"{entry['primary']['plasticity_metric']:.6f} | "
            f"{entry['retention_focused']['validation_retention_metric']:.6f} | "
            f"{'yes' if entry['endpoints_coincide'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Sensitivity arms: `" + json.dumps(artifact["sensitivity_arms"], sort_keys=True) + "`.",
            f"Measured recurrent-validation runtime: {runtime:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--legacy-v4-dir", type=Path, default=Path("results/phase1/tuning")
    )
    parser.add_argument(
        "--v4-dir", type=Path, default=Path("results/phase1/tuning_v4")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/tuning_v5")
    )
    parser.add_argument(
        "--tuned-config-path",
        type=Path,
        default=Path("results/phase1/phase1_tuned_configs.json"),
    )
    parser.add_argument(
        "--stream-b-dir", type=Path, default=Path("results/phase1/stream_b")
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    verify_preconditions()
    assert_stream_b_unseen(args.stream_b_dir)
    v4_path, v4_hashes = preserve_v4_artifacts(
        legacy_dir=args.legacy_v4_dir,
        v4_dir=args.v4_dir,
        canonical_path=args.tuned_config_path,
    )
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    validate_v4_artifact(v4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_path = args.output_dir / "validation_tasks.csv"
    pool_path = args.output_dir / "validation_pool.csv"
    cell_path = args.output_dir / "validation_cells.csv"
    report_path = args.output_dir / "report.md"
    config_path = args.output_dir / "run_config.json"

    any_existing = any(path.exists() for path in (task_path, pool_path, cell_path))
    if any_existing and not args.resume:
        raise RuntimeError(
            "validation outputs already exist; use --resume to preserve completed cells"
        )
    tasks = read_csv(task_path) if task_path.exists() else []
    pool = read_csv(pool_path) if pool_path.exists() else []
    cells = read_csv(cell_path) if cell_path.exists() else []
    if not any_existing:
        _write_validation(args.output_dir, [], [], [])

    previous_run_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    prior_runtime = float(previous_run_config.get("runtime_seconds", 0.0))
    run_config = {
        "experiment": "phase1_recurrent_validation",
        "amendment": PHASE1_AMENDMENT,
        "amendments": ["P1-A3", "P1-A4", "P1-A5"],
        "created_at_utc": previous_run_config.get("created_at_utc", utc_now()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "v4_artifact_path": str(v4_path),
        "v4_file_sha256": v4_hashes,
        "validation_schedule": validation_schedule(),
        "schedule_counts": dict(Counter(validation_schedule())),
        "runtime_seconds": prior_runtime,
    }
    atomic_write_json(config_path, run_config)
    started = time.perf_counter()
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
    completed = {
        (str(row["method"]), int(float(row["config_index"]))) for row in cells
    }
    grids = registered_grids()
    for method in TUNED_METHODS:
        for index, config in enumerate(grids[method]):
            key = (method, index)
            if key in completed:
                print(f"Skipping completed recurrent-validation cell {method}[{index}]")
                continue
            print(f"Starting recurrent-validation cell {method}[{index}] {config}")
            _remove_cell(tasks, method, index)
            _remove_cell(pool, method, index)
            _remove_cell(cells, method, index)
            new_tasks, new_pool, cell = run_validation_cell(
                method=method,
                config_index=index,
                config=config,
                mnist_train=mnist_train,
                test_loader=test_loader,
                permutations_cpu=permutations,
                schedule=validation_schedule(),
                device=device,
                num_workers=args.num_workers,
            )
            tasks.extend(new_tasks)
            pool.extend(new_pool)
            cells.append(cell)
            _write_validation(args.output_dir, tasks, pool, cells)
            run_config["updated_at_utc"] = utc_now()
            run_config["runtime_seconds"] = (
                prior_runtime + time.perf_counter() - started
            )
            atomic_write_json(config_path, run_config)

    validate_validation_results(tasks=tasks, pool=pool, cells=cells)
    artifact = build_v7_artifact(
        v4=v4,
        validation_cells=cells,
        v4_hashes=v4_hashes,
        v4_dir=args.v4_dir,
    )
    runtime = prior_runtime + time.perf_counter() - started
    atomic_write_json(args.tuned_config_path, artifact)
    atomic_write_json(args.output_dir / "phase1_tuned_configs.json", artifact)
    report = build_report(artifact, runtime)
    report_path.write_text(report, encoding="utf-8")
    run_config["completed_at_utc"] = utc_now()
    run_config["runtime_seconds"] = runtime
    run_config["canonical_tuned_config_path"] = str(args.tuned_config_path)
    atomic_write_json(config_path, run_config)
    print(report)


if __name__ == "__main__":
    main()
