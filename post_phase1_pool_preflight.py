"""Run PP1-A1's immutable Stream-B pool-headroom preflight."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import torch

import experiment_stream_b as stream_b
from experiment_e2 import read_csv
from experiment_e3 import fixed_permutation, load_mnist, make_loader
from experiment_e4a import EVALUATION_BATCH_SIZE, fixed_test_subset
from phase1_stream_common import validate_coverage
from post_phase1_common import (
    POOL_DECISION,
    POOL_PREFLIGHT_DIR,
    V2_ISOLATION,
    assert_post_output,
    atomic_write_json,
    block_stream_b_schedule,
    post_guard,
)


CONFIG = {"lr": 3e-4, "weight_decay": 0.0}
SEEDS = (42, 123)
POOL_SIZES = (20, 40)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(pool_size: int) -> dict[str, Path]:
    root = POOL_PREFLIGHT_DIR / f"pool_{pool_size}"
    return {
        "root": root,
        "scalars": root / "post_streamb_scalars.csv",
        "tasks": root / "post_streamb_tasks.csv",
        "pool": root / "post_streamb_pool.csv",
        "checkpoints": root / "checkpoints",
        "config": root / "post_run_config.json",
    }


def run_pool(
    pool_size: int,
    *,
    data_dir: Path,
    device: torch.device,
    num_workers: int,
    resume: bool,
) -> dict[str, Any]:
    paths = _paths(pool_size)
    for path in paths.values():
        assert_post_output(path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    existing_config = (
        json.loads(paths["config"].read_text(encoding="utf-8"))
        if paths["config"].exists()
        else {}
    )
    has_rows = any(
        path.exists() and bool(read_csv(path))
        for path in (paths["scalars"], paths["tasks"], paths["pool"])
    )
    has_checkpoints = paths["checkpoints"].exists() and any(
        paths["checkpoints"].rglob("*.pt")
    )
    if (has_rows or has_checkpoints) and existing_config.get("isolation") != V2_ISOLATION:
        raise RuntimeError(
            "pre-isolation pool-preflight output cannot be resumed by "
            f"PP1-A1(C): {paths['root']}"
        )
    schedule = block_stream_b_schedule(pool_size)
    visit_counts = dict(sorted(Counter(schedule).items()))
    first_appearance = {index: schedule.index(index) + 1 for index in range(pool_size)}
    atomic_write_json(
        paths["config"],
        {
            "schema_version": 3,
            "amendment": "PP1-A1(B), post-results",
            "isolation": V2_ISOLATION,
            "method": "adamw",
            "config": CONFIG,
            "seeds": list(SEEDS),
            "pool_size": pool_size,
            "pool_permutation_seeds": list(range(200_001, 200_001 + pool_size)),
            "tasks": stream_b.TASK_COUNT,
            "steps_per_task": stream_b.STEPS_PER_TASK,
            "schedule_type": "block-shuffled",
            "schedule_seed": 555,
            "schedule": schedule,
            "schedule_visit_counts": visit_counts,
            "schedule_first_appearance_tasks": first_appearance,
        },
    )
    scalar_rows = read_csv(paths["scalars"]) if resume and paths["scalars"].exists() else []
    task_rows = read_csv(paths["tasks"]) if resume and paths["tasks"].exists() else []
    pool_rows = read_csv(paths["pool"]) if resume and paths["pool"].exists() else []
    if not resume:
        stream_b._write_tables(
            paths["scalars"], paths["tasks"], paths["pool"], [], [], []
        )
    mnist_train, mnist_test = load_mnist(data_dir)
    test_loader = make_loader(
        fixed_test_subset(mnist_test, stream_b.TEST_SUBSET_SEED),
        EVALUATION_BATCH_SIZE,
        device,
        num_workers,
        shuffle=False,
    )
    permutations = {
        index: fixed_permutation(784, 200_001 + index)
        for index in range(pool_size)
    }
    artifact = {"methods": {"adamw": {"config": CONFIG}}}
    completed = {
        int(float(row["seed"]))
        for row in task_rows
        if int(float(row["task"])) == stream_b.TASK_COUNT
    }
    for seed in SEEDS:
        if seed in completed:
            print(f"Skipping completed pool-headroom cell n={pool_size} seed={seed}")
            continue
        stream_b.run_cell(
            arm="adamw",
            seed=seed,
            tuned_configs=artifact,
            mnist_train=mnist_train,
            test_loader=test_loader,
            permutations_cpu=permutations,
            schedule=schedule,
            scalar_rows=scalar_rows,
            task_rows=task_rows,
            pool_rows=pool_rows,
            scalar_path=paths["scalars"],
            task_path=paths["tasks"],
            pool_path=paths["pool"],
            checkpoint_dir=paths["checkpoints"],
            device=device,
            num_workers=num_workers,
            resume=resume,
            pool_size=pool_size,
        )
    validate_coverage(
        scalar_rows=scalar_rows,
        task_rows=task_rows,
        pool_rows=pool_rows,
        arms=("adamw",),
        seeds=SEEDS,
        task_count=stream_b.TASK_COUNT,
        total_steps=stream_b.TOTAL_STEPS,
        pool_size=pool_size,
    )
    values = stream_b.end_retention_by_seed(
        pool_rows, task_count=stream_b.TASK_COUNT, pool_size=pool_size
    )["adamw"]
    ordered = [float(values[seed]) for seed in SEEDS]
    return {
        "pool_size": pool_size,
        "end_retention_by_seed": {str(seed): values[seed] for seed in SEEDS},
        "mean_end_retention": mean(ordered),
        "sd_end_retention": stdev(ordered),
        "qualifies": mean(ordered) <= 0.85,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if POOL_DECISION.exists():
        existing = json.loads(POOL_DECISION.read_text(encoding="utf-8"))
        if existing.get("isolation") != V2_ISOLATION:
            raise RuntimeError(
                "pre-isolation pool-size decision cannot be used by PP1-A1(C): "
                f"{POOL_DECISION}"
            )
        raise RuntimeError(f"pool-size decision is immutable: {POOL_DECISION}")
    with post_guard():
        measurements = [
            run_pool(
                size,
                data_dir=args.data_dir,
                device=torch.device(args.device),
                num_workers=args.num_workers,
                resume=args.resume,
            )
            for size in POOL_SIZES
        ]
        qualifying = [row["pool_size"] for row in measurements if row["qualifies"]]
        chosen = min(qualifying) if qualifying else None
        payload = {
            "schema_version": 3,
            "created_at_utc": utc_now(),
            "amendment": "PP1-A1(B), written post-results on 2026-08-02",
            "isolation": V2_ISOLATION,
            "method": "adamw",
            "config": CONFIG,
            "threshold": 0.85,
            "candidate_pool_sizes": list(POOL_SIZES),
            "measurements": measurements,
            "chosen_pool_size": chosen,
            "status": "SELECTED" if chosen is not None else "POOL-HEADROOM-ABSENT",
        }
        atomic_write_json(POOL_DECISION, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
