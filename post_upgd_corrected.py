"""Run the separate corrected-UPGD mechanism diagnostic.

PP1-A1(C) forbids merging these cells into the isolated v2 selection pool.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

import experiment_stream_b as stream_b
from experiment_e2 import read_csv, write_csv
from experiment_e3 import fixed_permutation, load_mnist, make_loader
from experiment_e4a import EVALUATION_BATCH_SIZE, fixed_test_subset
from phase1_stream_common import validate_coverage
from post_phase1_common import (
    SEEDS,
    UPGD_CORRECTED_DIR,
    UPGD_CORRECTED_STREAM_TASKS,
    V2_ISOLATION,
    atomic_write_json,
    block_stream_b_schedule,
    canonical_json,
    post_guard,
    load_json,
    load_pool_decision,
)
from post_phase1_grid import CORRECTED_RESULTS, require_liveness_pass


CORRECTED_CONFIG = UPGD_CORRECTED_DIR / "post_corrected_config.json"
SCALARS = UPGD_CORRECTED_DIR / "post_streamb_scalars.csv"
TASKS = UPGD_CORRECTED_STREAM_TASKS
POOL = UPGD_CORRECTED_DIR / "post_streamb_pool.csv"
RUN_CONFIG = UPGD_CORRECTED_DIR / "post_streamb_run_config.json"
REPORT = UPGD_CORRECTED_DIR / "post_report.md"
ARM = "upgd_corrected"


def select_corrected() -> dict[str, Any]:
    rows = read_csv(CORRECTED_RESULTS)
    if len(rows) != 4:
        raise RuntimeError("the corrected-UPGD validation grid is incomplete")
    if any(str(row["status"]) != "OK" for row in rows):
        raise RuntimeError("a corrected-UPGD validation cell did not complete")
    best_plasticity = max(float(row["validation_plasticity"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["validation_plasticity"]) >= best_plasticity - 0.02
    ]
    winner = max(
        eligible,
        key=lambda row: (
            float(row["validation_retention_metric"]),
            float(row["validation_plasticity"]),
            -int(float(row["config_index"])),
        ),
    )
    payload = {
        "schema_version": 3,
        "isolation": V2_ISOLATION,
        "label": (
            "post-hoc robustness analysis (post-Stream-B; corrects a "
            "tuning-grid defect in our protocol)"
        ),
        "selection_rule": (
            "highest validation retention among configs within 0.02 of "
            "the corrected grid's own best validation plasticity"
        ),
        "best_validation_plasticity": best_plasticity,
        "winner_config_index": int(float(winner["config_index"])),
        "config": load_json_from_text(str(winner["config_json"])),
        "validation_retention_metric": float(
            winner["validation_retention_metric"]
        ),
        "validation_plasticity": float(winner["validation_plasticity"]),
        "tried_configs": [
            {
                "config_index": int(float(row["config_index"])),
                "config": load_json_from_text(str(row["config_json"])),
                "validation_retention_metric": float(
                    row["validation_retention_metric"]
                ),
                "validation_plasticity": float(row["validation_plasticity"]),
                "eligible": row in eligible,
            }
            for row in rows
        ],
    }
    if CORRECTED_CONFIG.exists() and load_json(CORRECTED_CONFIG) != payload:
        raise RuntimeError("the frozen corrected-UPGD selection changed")
    if not CORRECTED_CONFIG.exists():
        atomic_write_json(CORRECTED_CONFIG, payload)
    return payload


def load_json_from_text(text: str) -> dict[str, Any]:
    import json

    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("configuration JSON must be an object")
    return value


def run_stream(
    selection: Mapping[str, Any],
    data_dir: Path,
    device: torch.device,
    num_workers: int,
) -> None:
    pool_size = int(load_pool_decision()["chosen_pool_size"])
    config = dict(selection["config"])
    direct = {"methods": {"upgd": {"config": config}}}
    UPGD_CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    existing_run_config = load_json(RUN_CONFIG) if RUN_CONFIG.exists() else {}
    existing_paths = (SCALARS, TASKS, POOL)
    checkpoint_root = UPGD_CORRECTED_DIR / "post_checkpoints"
    has_state = any(
        path.exists() and bool(read_csv(path)) for path in existing_paths
    ) or (checkpoint_root.exists() and any(checkpoint_root.rglob("*.pt")))
    if has_state:
        if existing_run_config.get("isolation") != V2_ISOLATION:
            raise RuntimeError(
                "pre-isolation corrected-UPGD stream output cannot be resumed"
            )
    atomic_write_json(
        RUN_CONFIG,
        {
            "schema_version": 3,
            "isolation": V2_ISOLATION,
            "label": selection["label"],
            "status": "RUNNING",
            "seeds": list(SEEDS),
        },
    )
    scalars = read_csv(SCALARS) if SCALARS.exists() else []
    tasks = read_csv(TASKS) if TASKS.exists() else []
    pool = read_csv(POOL) if POOL.exists() else []
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
    schedule = block_stream_b_schedule(pool_size)
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]
    checkpoints = (
        UPGD_CORRECTED_DIR / "post_checkpoints" / ARM / digest
    )
    for seed in SEEDS:
        if any(
            str(row["arm"]) == ARM
            and int(float(row["seed"])) == seed
            and int(float(row["task"])) == stream_b.TASK_COUNT
            for row in tasks
        ):
            print(f"Skipping completed corrected-UPGD cell {seed}")
            continue
        stream_b.run_cell(
            arm=ARM,
            seed=seed,
            tuned_configs=direct,
            mnist_train=mnist_train,
            test_loader=test_loader,
            permutations_cpu=permutations,
            schedule=schedule,
            scalar_rows=scalars,
            task_rows=tasks,
            pool_rows=pool,
            scalar_path=SCALARS,
            task_path=TASKS,
            pool_path=POOL,
            checkpoint_dir=checkpoints,
            device=device,
            num_workers=num_workers,
            resume=True,
            pool_size=pool_size,
        )
    validate_coverage(
        scalar_rows=scalars,
        task_rows=tasks,
        pool_rows=pool,
        arms=(ARM,),
        seeds=SEEDS,
        task_count=stream_b.TASK_COUNT,
        total_steps=stream_b.TOTAL_STEPS,
        pool_size=pool_size,
    )
    by_seed = stream_b.end_retention_by_seed(
        pool, task_count=stream_b.TASK_COUNT, pool_size=pool_size
    )[ARM]
    values = [by_seed[seed] for seed in SEEDS]
    atomic_write_json(
        RUN_CONFIG,
        {
            "schema_version": 3,
            "amendment": "PP1-A1 (2026-08-02, post-results)",
            "isolation": V2_ISOLATION,
            "label": selection["label"],
            "config_path": str(CORRECTED_CONFIG),
            "config_sha256": hashlib.sha256(
                CORRECTED_CONFIG.read_bytes()
            ).hexdigest(),
            "seeds": list(SEEDS),
            "pool_size": pool_size,
            "pool_permutation_seeds": list(
                range(200_001, 200_001 + pool_size)
            ),
            "schedule_type": "block-shuffled",
            "schedule_seed": 555,
            "schedule": schedule,
            "end_retention_by_seed": by_seed,
            "end_retention_mean": mean(values),
            "end_retention_sd": stdev(values),
            "grading_suppressed": ["Phase-1", "E4b", "combined-E4", "E5-norm"],
        },
    )
    REPORT.write_text(
        "# Corrected-UPGD robustness analysis\n\n"
        f"**{selection['label']}**\n\n"
        f"Selected config: `{config}`.\n\n"
        f"Stream-B end retention: {mean(values):.6f} ± {stdev(values):.6f} "
        f"over seeds {list(SEEDS)}.\n\n"
        "PP1-A1(C) isolation: this mechanism diagnostic is not merged into "
        "the v2 selection pool, comparison table, or winner artifact. Every "
        "method in the main analysis is selected symmetrically from its own "
        "fresh v2 grid.\n\n"
        "The Phase-1 primary grading remains frozen and is not replaced.\n",
        encoding="utf-8",
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
        require_liveness_pass()
        selection = select_corrected()
        run_stream(
            selection,
            args.data_dir,
            torch.device(args.device),
            args.num_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
