"""Run/resume both final streams for frozen Post-Phase-1 winners.

This runner deliberately imports only the Phase-1 stream *protocol* helpers.
It never calls E4b, combined-E4, E5-norm, or Phase-1 grading functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import experiment_e4b as stream_a
import experiment_stream_b as stream_b
from experiment_e2 import read_csv, write_csv
from experiment_e3 import fixed_permutation, load_mnist, make_loader
from experiment_e4a import fixed_test_subset
from phase1_stream_common import validate_coverage
from post_phase1_common import (
    METHODS,
    POST_DIR,
    SEEDS,
    V2_ISOLATION,
    atomic_write_json,
    block_stream_b_schedule,
    canonical_json,
    post_guard,
    load_winner_artifact,
    load_pool_decision,
    normalize_config,
    post_deployment_artifact,
)


POST_STREAM_A = POST_DIR / "stream_a"
POST_STREAM_B = POST_DIR / "stream_b"
EWC_DIVERGENCE = POST_DIR / "post_ewc_divergence.json"
ISOLATION_MARKER = V2_ISOLATION
EVALUATION_BATCH_SIZE = 128  # Appendix A evaluation size, independent of historical E4 defaults.


def arm_name(method: str) -> str:
    return f"{method}_retention"


def _checkpoint_dir(root: Path, arm: str, config: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]
    return root / "post_checkpoints" / arm / digest


class StreamRuntime:
    def __init__(
        self, data_dir: Path, device: torch.device, num_workers: int,
        pool_size_b: int,
    ):
        self.device = device
        self.num_workers = num_workers
        self.mnist_train, mnist_test = load_mnist(data_dir)
        self.test_loader_a = make_loader(
            fixed_test_subset(mnist_test, stream_a.TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            num_workers,
            shuffle=False,
        )
        self.test_loader_b = make_loader(
            fixed_test_subset(mnist_test, stream_b.TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            num_workers,
            shuffle=False,
        )
        self.permutations_a = {
            task: fixed_permutation(
                784, stream_a.PERMUTATION_SEED_BASE + task
            )
            for task in range(1, stream_a.TASK_COUNT + 1)
        }
        self.permutations_b = {
            index: fixed_permutation(784, 200_001 + index)
            for index in range(pool_size_b)
        }
        self.pool_size_b = pool_size_b
        self.schedule_b = block_stream_b_schedule(pool_size_b)


def _paths_a() -> tuple[Path, Path, Path]:
    return (
        POST_STREAM_A / "stream_scalars.csv",
        POST_STREAM_A / "stream_tasks.csv",
        POST_STREAM_A / "post_run_config.json",
    )


def _paths_b() -> tuple[Path, Path, Path, Path]:
    return (
        POST_STREAM_B / "streamb_scalars.csv",
        POST_STREAM_B / "streamb_tasks.csv",
        POST_STREAM_B / "streamb_pool.csv",
        POST_STREAM_B / "post_run_config.json",
    )


def _clear_arm(
    arm: str,
    scalar_a: list[dict[str, Any]],
    task_a: list[dict[str, Any]],
    scalar_b: list[dict[str, Any]],
    task_b: list[dict[str, Any]],
    pool_b: list[dict[str, Any]],
) -> None:
    for rows in (scalar_a, task_a, scalar_b, task_b, pool_b):
        rows[:] = [row for row in rows if str(row.get("arm")) != arm]
    scalar_path_a, task_path_a, _ = _paths_a()
    scalar_path_b, task_path_b, pool_path_b, _ = _paths_b()
    write_csv(scalar_path_a, scalar_a, stream_a.SCALAR_FIELDS)
    write_csv(task_path_a, task_a, stream_a.TASK_FIELDS)
    write_csv(scalar_path_b, scalar_b, stream_b.SCALAR_FIELDS)
    write_csv(task_path_b, task_b, stream_b.TASK_FIELDS)
    write_csv(pool_path_b, pool_b, stream_b.POOL_FIELDS)


def _run_a(
    *,
    method: str,
    config: Mapping[str, Any],
    runtime: StreamRuntime,
    scalar_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> None:
    arm = arm_name(method)
    scalar_path, task_path, _ = _paths_a()
    checkpoints = _checkpoint_dir(POST_STREAM_A, arm, config)
    direct = post_deployment_artifact(method, config)
    for seed in SEEDS:
        if any(
            str(row["arm"]) == arm
            and int(float(row["seed"])) == seed
            and int(float(row["task"])) == stream_a.TASK_COUNT
            for row in task_rows
        ):
            print(f"Skipping completed Stream-A cell {arm}/{seed}")
            continue
        stream_a.run_cell(
            arm=arm,
            seed=seed,
            mnist_train=runtime.mnist_train,
            test_loader=runtime.test_loader_a,
            permutations_cpu=runtime.permutations_a,
            scalar_rows=scalar_rows,
            task_rows=task_rows,
            scalar_path=scalar_path,
            task_path=task_path,
            checkpoint_dir=checkpoints,
            device=runtime.device,
            num_workers=runtime.num_workers,
            resume=True,
            tuned_configs=direct,
        )


def _run_b(
    *,
    method: str,
    config: Mapping[str, Any],
    runtime: StreamRuntime,
    scalar_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> None:
    arm = arm_name(method)
    scalar_path, task_path, pool_path, _ = _paths_b()
    checkpoints = _checkpoint_dir(POST_STREAM_B, arm, config)
    direct = post_deployment_artifact(method, config)
    for seed in SEEDS:
        if any(
            str(row["arm"]) == arm
            and int(float(row["seed"])) == seed
            and int(float(row["task"])) == stream_b.TASK_COUNT
            for row in task_rows
        ):
            print(f"Skipping completed Stream-B cell {arm}/{seed}")
            continue
        stream_b.run_cell(
            arm=arm,
            seed=seed,
            tuned_configs=direct,
            mnist_train=runtime.mnist_train,
            test_loader=runtime.test_loader_b,
            permutations_cpu=runtime.permutations_b,
            schedule=runtime.schedule_b,
            scalar_rows=scalar_rows,
            task_rows=task_rows,
            pool_rows=pool_rows,
            scalar_path=scalar_path,
            task_path=task_path,
            pool_path=pool_path,
            checkpoint_dir=checkpoints,
            device=runtime.device,
            num_workers=runtime.num_workers,
            resume=True,
            pool_size=runtime.pool_size_b,
        )


def _ewc_candidates(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = normalize_config("ewc_oracle", entry["retention_config"])
    tried = [
        item
        for item in entry["tried_configs"]
        if not bool(item.get("diverged"))
        and item.get("validation_retention_metric") is not None
    ]
    tried.sort(
        key=lambda item: (
            -min(
                float(item["validation_retention_metric"]),
                float(item["validation_plasticity"]),
            ),
            -(
                float(item["validation_retention_metric"])
                + float(item["validation_plasticity"])
            ) / 2,
            int(item["selection_order"]),
        )
    )
    ordered = [selected]
    for item in tried:
        config = normalize_config("ewc_oracle", item["config"])
        if config not in ordered:
            ordered.append(config)
    return ordered


def run_all(data_dir: Path, device: torch.device, num_workers: int) -> None:
    artifact = load_winner_artifact()
    decision = load_pool_decision()
    pool_size_b = int(decision["chosen_pool_size"])
    POST_STREAM_A.mkdir(parents=True, exist_ok=True)
    POST_STREAM_B.mkdir(parents=True, exist_ok=True)
    scalar_path_a, task_path_a, config_path_a = _paths_a()
    scalar_path_b, task_path_b, pool_path_b, config_path_b = _paths_b()
    for config_path, table_paths in (
        (config_path_a, (scalar_path_a, task_path_a)),
        (config_path_b, (scalar_path_b, task_path_b, pool_path_b)),
    ):
        checkpoint_root = config_path.parent / "post_checkpoints"
        has_rows = any(path.exists() and bool(read_csv(path)) for path in table_paths)
        has_checkpoints = checkpoint_root.exists() and any(
            checkpoint_root.rglob("*.pt")
        )
        existing_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        if (has_rows or has_checkpoints) and existing_config.get("isolation") != ISOLATION_MARKER:
            raise RuntimeError(
                "pre-isolation stream output exists and cannot be reused by "
                f"PP1-A1(C): {config_path.parent}"
            )
    initial_config = {
        "schema_version": 3,
        "isolation": ISOLATION_MARKER,
        "arms": [arm_name(method) for method in METHODS],
        "seeds": list(SEEDS),
        "grading_suppressed": ["E4b", "combined-E4", "E5-norm"],
        "checkpoint_every": 50_000,
    }
    atomic_write_json(config_path_a, {**initial_config, "stream": "A"})
    atomic_write_json(config_path_b, {**initial_config, "stream": "B"})
    scalar_a = read_csv(scalar_path_a) if scalar_path_a.exists() else []
    task_a = read_csv(task_path_a) if task_path_a.exists() else []
    scalar_b = read_csv(scalar_path_b) if scalar_path_b.exists() else []
    task_b = read_csv(task_path_b) if task_path_b.exists() else []
    pool_b = read_csv(pool_path_b) if pool_path_b.exists() else []

    runtime = StreamRuntime(data_dir, device, num_workers, pool_size_b)
    deployed: dict[str, Any] = {}
    divergence_records: list[dict[str, Any]] = []
    prior_divergence = (
        json.loads(EWC_DIVERGENCE.read_text(encoding="utf-8"))
        if EWC_DIVERGENCE.exists()
        else {"schema_version": 3, "isolation": ISOLATION_MARKER, "attempts": []}
    )
    if prior_divergence.get("isolation") != ISOLATION_MARKER:
        raise RuntimeError(
            "pre-isolation EWC divergence state cannot be used by PP1-A1(C)"
        )
    known_failed = {
        canonical_json(item["config"])
        for item in prior_divergence.get("attempts", [])
    }

    for method in METHODS:
        entry = artifact[method]
        arm = arm_name(method)

        candidates = (
            _ewc_candidates(entry)
            if method == "ewc_oracle"
            else [normalize_config(method, entry["retention_config"])]
        )
        completed = False
        for candidate in candidates:
            if canonical_json(candidate) in known_failed:
                continue
            if candidate != candidates[0]:
                _clear_arm(
                    arm, scalar_a, task_a, scalar_b, task_b, pool_b
                )
            try:
                _run_a(
                    method=method,
                    config=candidate,
                    runtime=runtime,
                    scalar_rows=scalar_a,
                    task_rows=task_a,
                )
                _run_b(
                    method=method,
                    config=candidate,
                    runtime=runtime,
                    scalar_rows=scalar_b,
                    task_rows=task_b,
                    pool_rows=pool_b,
                )
            except FloatingPointError as error:
                if method != "ewc_oracle":
                    raise
                record = {
                    "verdict": "EWC-RETENTION-DIVERGED",
                    "config": candidate,
                    "error": str(error),
                }
                divergence_records.append(record)
                known_failed.add(canonical_json(candidate))
                atomic_write_json(
                    EWC_DIVERGENCE,
                    {
                        "schema_version": 3,
                        "isolation": ISOLATION_MARKER,
                        "verdict": "EWC-RETENTION-DIVERGED",
                        "attempts": [
                            *prior_divergence.get("attempts", []),
                            *divergence_records,
                        ],
                    },
                )
                continue
            deployed[method] = {
                "config": candidate,
                "status": (
                    "fallback after EWC-RETENTION-DIVERGED"
                    if candidate != candidates[0]
                    else "completed"
                ),
                "stream_a_source": "PP1-A1(C) fresh v2 run",
                "stream_b_source": "PP1-A1 v2 run",
            }
            completed = True
            break
        if not completed:
            raise RuntimeError(
                "EWC-RETENTION-DIVERGED: no candidate completed both streams"
            )

    arms = tuple(arm_name(method) for method in METHODS)
    validate_coverage(
        scalar_rows=scalar_a,
        task_rows=task_a,
        pool_rows=None,
        arms=arms,
        seeds=SEEDS,
        task_count=stream_a.TASK_COUNT,
        total_steps=stream_a.TOTAL_STEPS,
    )
    validate_coverage(
        scalar_rows=scalar_b,
        task_rows=task_b,
        pool_rows=pool_b,
        arms=arms,
        seeds=SEEDS,
        task_count=stream_b.TASK_COUNT,
        total_steps=stream_b.TOTAL_STEPS,
        pool_size=pool_size_b,
    )
    common_config = {
        "schema_version": 3,
        "amendment": "PP1-A1 (2026-08-02, post-results)",
        "arms": list(arms),
        "seeds": list(SEEDS),
        "deployed": deployed,
        "grading_suppressed": ["E4b", "combined-E4", "E5-norm"],
        "checkpoint_every": 50_000,
        "isolation": ISOLATION_MARKER,
    }
    atomic_write_json(config_path_a, {**common_config, "stream": "A"})
    atomic_write_json(
        config_path_b,
        {
            **common_config,
            "stream": "B",
            "pool_size": pool_size_b,
            "pool_permutation_seeds": list(
                range(200_001, 200_001 + pool_size_b)
            ),
            "schedule_type": "block-shuffled",
            "schedule_seed": 555,
            "schedule": runtime.schedule_b,
            "schedule_visit_counts": {
                str(index): runtime.schedule_b.count(index)
                for index in range(pool_size_b)
            },
            "schedule_first_appearance_tasks": {
                str(index): runtime.schedule_b.index(index) + 1
                for index in range(pool_size_b)
            },
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    with post_guard():
        run_all(args.data_dir, torch.device(args.device), args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
