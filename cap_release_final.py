"""Finalize revised contest-blocked CAP through bounded retuning and reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import experiment_e4b as stream_a
import experiment_cap_v2_postp1 as cap_runner
import experiment_stream_b as stream_b
from experiment_e2 import read_csv, write_csv
from post_phase1_common import POST_DIR, V2_ISOLATION
from post_phase1_report import stream_a_plasticity
from post_phase1_streams import StreamRuntime
from cap_v2_postp1_common import (
    DEFAULT_OUTPUT_ROOT,
    FULL_SEEDS,
    POST_P1_POOL_SIZE,
    assert_authoritative_protocol,
    atomic_write_json,
    block_stream_b_schedule,
    build_run_config,
    canonical_json,
    complete_cap_config,
    lock_run_config,
    protocol_hash,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT_VERSION = "cap-revised-release-final-v1"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "cap_release_final"
K_GRID = (5e-5, 7e-5, 1e-4)
TM_GRID = (500_000.0, 1_000_000.0, 2_000_000.0)
CENTER = {"k": 7e-5, "T_m": 1_000_000.0}
PRIMARY_DEVELOPMENT_SEED = 42
CONFIRMATION_SEEDS = (31_415, 27_182)
DEVELOPMENT_SEEDS = (PRIMARY_DEVELOPMENT_SEED, *CONFIRMATION_SEEDS)
FINAL_SEEDS = FULL_SEEDS
if set(CONFIRMATION_SEEDS) & set(FINAL_SEEDS):
    raise RuntimeError("confirmation seeds must be development-only")
TEMPORAL_WINDOWS = (
    ("1-100k", 1, 100_000),
    ("101-250k", 100_001, 250_000),
    ("251-500k", 250_001, 500_000),
    ("501-750k", 500_001, 750_000),
    ("751k-1M", 750_001, 1_000_000),
)
WITHIN_TASK_STEPS = (100, 200, 300, 500, 1_000, 1_500, 2_000, 5_000)
GAP_BUCKETS = (
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41-80", 41, 80),
)
RETUNE_RUN_FIELDS = (
    "config_id",
    "stage",
    "seed",
    "stream",
    "k",
    "T_m",
    "plasticity",
    "retention",
    "balanced_score",
    "current_task_accuracy",
    "bind_rate",
    "median_imp",
    "median_radius",
    "frozen_fraction",
    "theta_norm",
)
RETUNE_SUMMARY_FIELDS = (
    "config_id",
    "k",
    "T_m",
    "development_seeds",
    "seed_count",
    "mean_plasticity",
    "std_plasticity",
    "mean_retention",
    "std_retention",
    "mean_seedwise_min",
    "min_of_means",
    "mean_plasticity_retention",
    "pareto_stage1",
    "stage1_rank",
    "confirmation_candidate",
    "selection_rank",
    "boundary_selected",
)
FINAL_METRIC_FIELDS = (
    "method",
    "seed",
    "plasticity",
    "retention",
    "seedwise_min",
)
PAIRED_DELTA_FIELDS = (
    "seed",
    "delta_plasticity",
    "delta_retention",
    "delta_seedwise_min",
)
GAP_FIELDS = ("method", "seed", "gap_bucket", "mean_retention", "count")
TEMPORAL_FIELDS = (
    "method",
    "seed",
    "summary_type",
    "window_or_step",
    "mean_accuracy",
    "count",
)
DIAGNOSTIC_SUMMARY_FIELDS = (
    "method",
    "seed",
    "bind_rate",
    "median_imp",
    "median_radius",
    "frozen_fraction",
    "theta_norm",
    "current_task_accuracy",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--retune", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--all-seeds", action="store_true")
    seeds.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.retune and (args.all_seeds or args.seeds):
        parser.error("--retune uses the fixed development seeds")
    if args.train:
        requested = (
            FINAL_SEEDS
            if args.all_seeds or args.seeds is None
            else tuple(args.seeds)
        )
        if len(requested) != len(FINAL_SEEDS) or set(requested) != set(FINAL_SEEDS):
            parser.error(f"--train requires the fixed final seeds {FINAL_SEEDS}")
    if args.analyze and (args.all_seeds or args.seeds):
        parser.error("--analyze uses the fixed final reporting seeds")
    return args


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _config_id(config: Mapping[str, float]) -> str:
    k_label = f"{float(config['k']):.0e}".replace("+", "")
    tm_label = f"{int(float(config['T_m']))}"
    return f"k{k_label}_tm{tm_label}"


def _grid() -> list[dict[str, float]]:
    return [{"k": k, "T_m": tm} for k in K_GRID for tm in TM_GRID]


def _full_cap_config(config: Mapping[str, float]) -> dict[str, Any]:
    return {
        **complete_cap_config(
            {"k": float(config["k"]), "T_m": float(config["T_m"])}
        ),
        "release_update": "contest_blocked",
        "momentum_response": "none",
        "box_geometry": "coordinate",
        "protection": "box",
        "importance_evidence": "raw_g_squared",
        "anchor_update": "moving",
        "amsgrad": False,
        "maximize": False,
        "freeze_stats": False,
    }


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, payload)


def _retune_config() -> dict[str, Any]:
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": "retune",
        "grid": _grid(),
        "primary_development_seed": PRIMARY_DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "final_reporting_seeds": list(FINAL_SEEDS),
        "selection_metric": "mean seedwise min(plasticity, retention)",
        "stage1_rule": "Pareto nondominated then top three by seedwise minimum",
        "stage1_tie_breaks": [
            "higher mean of plasticity and retention",
            "prefer center configuration",
        ],
        "tie_breaks": [
            "higher mean seedwise minimum",
            "higher minimum of mean plasticity and mean retention",
            "higher mean of plasticity and retention",
            "lower sum of plasticity and retention sample standard deviations",
            "prefer center configuration",
        ],
        "fixed_cap_config": _full_cap_config(CENTER),
        "protocol_hash": protocol_hash(),
    }
    payload["config_hash"] = _hash_payload(payload)
    return payload


def _runtime(args: argparse.Namespace) -> StreamRuntime:
    return StreamRuntime(
        args.data_dir,
        torch.device(args.device),
        args.num_workers,
        POST_P1_POOL_SIZE,
    )


def _run_configuration(
    *,
    args: argparse.Namespace,
    runtime: StreamRuntime,
    root: Path,
    config: Mapping[str, float],
    seeds: Sequence[int],
    phase: str,
    label: str,
) -> None:
    config_id = _config_id(config)
    cap_overrides = {"k": float(config["k"]), "T_m": float(config["T_m"])}
    cap_config = _full_cap_config(config)
    config_hash = _hash_payload(cap_config)
    arm = f"cap_release_final_{label}"
    context = {
        "experiment_version": EXPERIMENT_VERSION,
        "phase": phase,
        "label": label,
        "config_id": config_id,
        "config_hash": config_hash,
    }
    for stream in ("a", "b"):
        stream_root = root / f"stream_{stream}"
        stream_root.mkdir(parents=True, exist_ok=True)
        paths = cap_runner.stream_paths(stream_root, stream)
        proposed = build_run_config(
            stream=stream,
            variant="baseline",
            device=runtime.device,
            runner_path=Path(__file__).resolve(),
            cap_overrides=cap_overrides,
            experiment_context=context,
        )
        proposed["experiment_version"] = EXPERIMENT_VERSION
        proposed["mode"] = phase
        proposed["configuration_hash"] = config_hash
        proposed["final_reporting_seeds"] = list(FINAL_SEEDS)
        proposed["mandatory_prelaunch_test"] = (
            "python -m unittest tests.test_cap_v2_postp1 "
            "tests.test_cap_release_final"
        )
        lock_run_config(
            paths["config"],
            proposed,
            requested_seeds=seeds,
            resume=args.resume,
            device=runtime.device,
        )
        for seed in seeds:
            cap_runner.run_cell(
                stream=stream,
                variant="baseline",
                seed=int(seed),
                runtime=runtime,
                paths=paths,
                resume=args.resume,
                cap_overrides=cap_overrides,
                arm_override=arm,
                experiment_context=context,
            )


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def _sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _arm_metrics(root: Path, arm: str) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = {}
    scalar_path = root / "stream_a" / "stream_scalars.csv"
    if scalar_path.exists():
        plasticity = stream_a_plasticity(read_csv(scalar_path))
        if arm in plasticity:
            metrics["plasticity"] = plasticity[arm]
    pool_path = root / "stream_b" / "streamb_pool.csv"
    if pool_path.exists():
        retention = stream_b.end_retention_by_seed(
            read_csv(pool_path), pool_size=POST_P1_POOL_SIZE
        )
        if arm in retention:
            metrics["retention"] = retention[arm]
    task_path = root / "stream_b" / "streamb_tasks.csv"
    if task_path.exists():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in read_csv(task_path):
            if str(row["arm"]) == arm:
                grouped[int(float(row["seed"]))].append(float(row["task_end_accuracy"]))
        metrics["current_task_accuracy"] = {
            seed: _mean(values) for seed, values in grouped.items()
        }
    return metrics


def _diagnostic_means(
    root: Path,
    arm: str,
    streams: Sequence[str] = ("stream_a", "stream_b"),
) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for stream in streams:
        if stream not in ("stream_a", "stream_b"):
            raise ValueError(f"unknown diagnostic stream: {stream}")
        path = root / stream / "cap_v2_diagnostics.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            seed = int(float(row["seed"]))
            for field in ("bind_rate", "median_imp", "median_radius", "theta_norm"):
                if row.get(field, "") != "":
                    value = float(row[field])
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"non-finite {field} diagnostic for seed{seed}"
                        )
                    grouped[seed][field].append(value)
        scalar_name = (
            "stream_scalars.csv"
            if stream == "stream_a"
            else "streamb_scalars.csv"
        )
        scalar_path = root / stream / scalar_name
        if scalar_path.exists():
            for row in read_csv(scalar_path):
                if str(row["arm"]) != arm or row.get("frozen_fraction", "") == "":
                    continue
                seed = int(float(row["seed"]))
                value = float(row["frozen_fraction"])
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"non-finite frozen_fraction diagnostic for seed{seed}"
                    )
                grouped[seed]["frozen_fraction"].append(value)
    return {
        seed: {field: _mean(values) for field, values in fields.items()}
        for seed, fields in grouped.items()
    }


def _retune_rows(output_dir: Path, candidates: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in _grid():
        config_id = _config_id(config)
        root = output_dir / "retune" / "runs" / config_id
        arm = f"cap_release_final_retune_{config_id}"
        metrics = _arm_metrics(root, arm)
        diagnostics_a = _diagnostic_means(root, arm, ("stream_a",))
        diagnostics_b = _diagnostic_means(root, arm, ("stream_b",))
        seeds = (
            DEVELOPMENT_SEEDS if config_id in candidates else (PRIMARY_DEVELOPMENT_SEED,)
        )
        for seed in seeds:
            plasticity = metrics.get("plasticity", {}).get(seed)
            retention = metrics.get("retention", {}).get(seed)
            required_diagnostics = {
                "bind_rate",
                "median_imp",
                "median_radius",
                "frozen_fraction",
                "theta_norm",
            }
            if (
                not required_diagnostics <= set(diagnostics_a.get(seed, {}))
                or not required_diagnostics <= set(diagnostics_b.get(seed, {}))
                or seed not in metrics.get("current_task_accuracy", {})
            ):
                raise RuntimeError(
                    f"retune diagnostics are incomplete for {config_id}/seed{seed}"
                )
            common = {
                "config_id": config_id,
                "stage": 1 if seed == PRIMARY_DEVELOPMENT_SEED else 2,
                "seed": seed,
                "k": config["k"],
                "T_m": config["T_m"],
                "balanced_score": (
                    min(plasticity, retention)
                    if plasticity is not None and retention is not None
                    else ""
                ),
            }
            rows.append(
                {
                    **common,
                    "stream": "a",
                    "plasticity": "" if plasticity is None else plasticity,
                    "retention": "",
                    "current_task_accuracy": "",
                    **diagnostics_a.get(seed, {}),
                }
            )
            rows.append(
                {
                    **common,
                    "stream": "b",
                    "plasticity": "",
                    "retention": "" if retention is None else retention,
                    "current_task_accuracy": metrics.get(
                        "current_task_accuracy", {}
                    ).get(seed, ""),
                    **diagnostics_b.get(seed, {}),
                }
            )
    return rows


def _paired_scores(
    output_dir: Path, config: Mapping[str, float], seeds: Sequence[int]
) -> dict[int, tuple[float, float]]:
    config_id = _config_id(config)
    root = output_dir / "retune" / "runs" / config_id
    metrics = _arm_metrics(root, f"cap_release_final_retune_{config_id}")
    scores = {
        seed: (metrics["plasticity"][seed], metrics["retention"][seed])
        for seed in seeds
        if seed in metrics.get("plasticity", {})
        and seed in metrics.get("retention", {})
    }
    for seed, values in scores.items():
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite metric for {config_id}/seed{seed}")
    return scores


def _pareto_ids(stage1: Mapping[str, tuple[float, float]]) -> set[str]:
    return {
        config_id
        for config_id, (plasticity, retention) in stage1.items()
        if not any(
            other_id != config_id
            and other_plasticity >= plasticity
            and other_retention >= retention
            and (other_plasticity > plasticity or other_retention > retention)
            for other_id, (other_plasticity, other_retention) in stage1.items()
        )
    }


def _stage1_top_three(output_dir: Path) -> tuple[list[str], set[str]]:
    stage1: dict[str, tuple[float, float]] = {}
    for config in _grid():
        scores = _paired_scores(output_dir, config, (PRIMARY_DEVELOPMENT_SEED,))
        if PRIMARY_DEVELOPMENT_SEED in scores:
            stage1[_config_id(config)] = scores[PRIMARY_DEVELOPMENT_SEED]
    if len(stage1) != len(_grid()):
        raise RuntimeError("Stage 1 is incomplete; all nine configurations need both streams")
    pareto = _pareto_ids(stage1)
    ranked_pareto = sorted(
        pareto,
        key=lambda config_id: (
            -min(stage1[config_id]),
            -(sum(stage1[config_id]) / 2.0),
            config_id != _config_id(CENTER),
            config_id,
        ),
    )
    remaining = sorted(
        set(stage1) - set(ranked_pareto),
        key=lambda config_id: (
            -min(stage1[config_id]),
            -(sum(stage1[config_id]) / 2.0),
            config_id != _config_id(CENTER),
            config_id,
        ),
    )
    return (ranked_pareto + remaining)[:3], pareto


def _candidate_key(
    output_dir: Path, config: Mapping[str, float]
) -> tuple[float, float, float, float, bool]:
    scores = _paired_scores(output_dir, config, DEVELOPMENT_SEEDS)
    if len(scores) != len(DEVELOPMENT_SEEDS):
        raise RuntimeError(f"confirmation is incomplete for {_config_id(config)}")
    plasticity = [scores[seed][0] for seed in DEVELOPMENT_SEEDS]
    retention = [scores[seed][1] for seed in DEVELOPMENT_SEEDS]
    seedwise_min = [min(*scores[seed]) for seed in DEVELOPMENT_SEEDS]
    mean_p = _mean(plasticity)
    mean_r = _mean(retention)
    return (
        _mean(seedwise_min),
        min(mean_p, mean_r),
        (mean_p + mean_r) / 2.0,
        -(_sd(plasticity) + _sd(retention)),
        _config_id(config) == _config_id(CENTER),
    )


def _summary_rows(
    output_dir: Path, candidate_ids: Sequence[str], pareto: set[str]
) -> list[dict[str, Any]]:
    candidate_set = set(candidate_ids)
    stage1_scores = {
        _config_id(config): _paired_scores(
            output_dir, config, (PRIMARY_DEVELOPMENT_SEED,)
        )[PRIMARY_DEVELOPMENT_SEED]
        for config in _grid()
    }
    stage1_order = {
        config_id: rank
        for rank, config_id in enumerate(
            sorted(
                stage1_scores,
                key=lambda cid: (
                    -min(stage1_scores[cid]),
                    -(sum(stage1_scores[cid]) / 2.0),
                    cid != _config_id(CENTER),
                    cid,
                ),
            ),
            start=1,
        )
    }
    candidate_order = {
        config_id: rank
        for rank, config_id in enumerate(
            sorted(
                candidate_ids,
                key=lambda cid: _candidate_key(
                    output_dir,
                    next(config for config in _grid() if _config_id(config) == cid),
                ),
                reverse=True,
            ),
            start=1,
        )
    }
    rows: list[dict[str, Any]] = []
    for config in _grid():
        config_id = _config_id(config)
        seeds = DEVELOPMENT_SEEDS if config_id in candidate_set else (PRIMARY_DEVELOPMENT_SEED,)
        scores = _paired_scores(output_dir, config, seeds)
        plasticity = [scores[seed][0] for seed in seeds]
        retention = [scores[seed][1] for seed in seeds]
        seedwise_min = [min(*scores[seed]) for seed in seeds]
        mean_p = _mean(plasticity)
        mean_r = _mean(retention)
        rows.append(
            {
                "config_id": config_id,
                "k": config["k"],
                "T_m": config["T_m"],
                "development_seeds": json.dumps(list(seeds)),
                "seed_count": len(scores),
                "mean_plasticity": mean_p,
                "std_plasticity": _sd(plasticity),
                "mean_retention": mean_r,
                "std_retention": _sd(retention),
                "mean_seedwise_min": _mean(seedwise_min),
                "min_of_means": min(mean_p, mean_r),
                "mean_plasticity_retention": (mean_p + mean_r) / 2.0,
                "pareto_stage1": config_id in pareto,
                "stage1_rank": stage1_order[config_id],
                "confirmation_candidate": config_id in candidate_set,
                "selection_rank": candidate_order.get(config_id, ""),
                "boundary_selected": (
                    config["k"] in (min(K_GRID), max(K_GRID))
                    or config["T_m"] in (min(TM_GRID), max(TM_GRID))
                ),
            }
        )
    return rows


def run_retune(args: argparse.Namespace) -> None:
    assert_authoritative_protocol()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retune_root = args.output_dir / "retune"
    retune_root.mkdir(parents=True, exist_ok=True)
    _immutable_json(retune_root / "retune_config.json", _retune_config())
    runtime = _runtime(args)
    for config in _grid():
        config_id = _config_id(config)
        _run_configuration(
            args=args,
            runtime=runtime,
            root=retune_root / "runs" / config_id,
            config=config,
            seeds=(PRIMARY_DEVELOPMENT_SEED,),
            phase="retune",
            label=f"retune_{config_id}",
        )
    candidate_ids, pareto = _stage1_top_three(args.output_dir)
    by_id = {_config_id(config): config for config in _grid()}
    for config_id in candidate_ids:
        _run_configuration(
            args=args,
            runtime=runtime,
            root=retune_root / "runs" / config_id,
            config=by_id[config_id],
            seeds=CONFIRMATION_SEEDS,
            phase="retune",
            label=f"retune_{config_id}",
        )
    run_rows = _retune_rows(args.output_dir, set(candidate_ids))
    write_csv(retune_root / "retune_runs.csv", run_rows, RETUNE_RUN_FIELDS)
    summary_rows = _summary_rows(args.output_dir, candidate_ids, pareto)
    write_csv(retune_root / "retune_summary.csv", summary_rows, RETUNE_SUMMARY_FIELDS)
    gap_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for config in _grid():
        config_id = _config_id(config)
        root = retune_root / "runs" / config_id
        arm = f"cap_release_final_retune_{config_id}"
        gap_rows.extend(
            _retention_gap_rows(
                config_id, read_csv(root / "stream_b" / "streamb_pool.csv"), arm
            )
        )
        temporal_rows.extend(
            _temporal_rows(
                config_id, read_csv(root / "stream_a" / "stream_scalars.csv"), arm
            )
        )
    write_csv(retune_root / "retention_by_gap.csv", gap_rows, GAP_FIELDS)
    write_csv(retune_root / "temporal_plasticity.csv", temporal_rows, TEMPORAL_FIELDS)
    selected_id = min(
        candidate_ids,
        key=lambda cid: int(
            next(row["selection_rank"] for row in summary_rows if row["config_id"] == cid)
        ),
    )
    selected = by_id[selected_id]
    selected_row = next(row for row in summary_rows if row["config_id"] == selected_id)
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "selected_config_id": selected_id,
        "selected_k": selected["k"],
        "selected_T_m": selected["T_m"],
        "cap_config": _full_cap_config(selected),
        "development_selection_score": selected_row["mean_seedwise_min"],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "final_reporting_seeds": list(FINAL_SEEDS),
        "selection_rank": 1,
        "boundary_selected": selected_row["boundary_selected"],
        "protocol_hash": protocol_hash(),
    }
    payload["config_hash"] = _hash_payload(payload)
    _immutable_json(args.output_dir / "selected_config.json", payload)
    print(f"Selected {selected_id}; wrote retune artifacts under {retune_root}")


def _load_selected(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"run --retune first; missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("config_hash")
    observed = _hash_payload(payload)
    payload["config_hash"] = claimed
    if claimed != observed:
        raise RuntimeError("selected_config.json hash mismatch")
    if payload["experiment_version"] != EXPERIMENT_VERSION:
        raise RuntimeError("selected_config.json experiment version mismatch")
    selected_config = {
        "k": float(payload["selected_k"]),
        "T_m": float(payload["selected_T_m"]),
    }
    if (
        selected_config not in _grid()
        or payload.get("selected_config_id") != _config_id(selected_config)
        or payload.get("cap_config") != _full_cap_config(selected_config)
        or tuple(payload.get("development_seeds", ())) != DEVELOPMENT_SEEDS
        or tuple(payload.get("confirmation_seeds", ())) != CONFIRMATION_SEEDS
        or tuple(payload.get("final_reporting_seeds", ())) != FINAL_SEEDS
        or payload.get("protocol_hash") != protocol_hash()
    ):
        raise RuntimeError("selected_config.json scientific identity mismatch")
    return payload


def _validate_retune_summary(output_dir: Path, selected: Mapping[str, Any]) -> None:
    path = output_dir / "retune" / "retune_summary.csv"
    rows = read_csv(path)
    matching = [
        row
        for row in rows
        if str(row["config_id"]) == str(selected["selected_config_id"])
    ]
    if len(rows) != len(_grid()) or len(matching) != 1:
        raise RuntimeError("retune_summary.csv coverage/selection mismatch")
    row = matching[0]
    if (
        int(float(row["selection_rank"])) != 1
        or float(row["k"]) != float(selected["selected_k"])
        or float(row["T_m"]) != float(selected["selected_T_m"])
    ):
        raise RuntimeError("selected_config.json disagrees with retune_summary.csv")


def _existing_v25_center() -> tuple[Path, str] | None:
    root = DEFAULT_OUTPUT_ROOT / "v25_baseline"
    arm = "cap_release_v25_baseline"
    expected_cap = complete_cap_config(CENTER)
    for stream in ("a", "b"):
        config_path = root / f"stream_{stream}" / "run_config.json"
        if not config_path.exists():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            config.get("variant") != "baseline"
            or config.get("cap_config") != expected_cap
            or config.get("release_update") != "contest_blocked"
            or config.get("box_geometry") != "coordinate"
            or config.get("momentum_response") != "none"
            or config.get("post_phase1_protocol_hash") != protocol_hash()
        ):
            return None
    metrics = _arm_metrics(root, arm)
    if not all(
        seed in metrics.get("plasticity", {})
        and seed in metrics.get("retention", {})
        and math.isfinite(metrics["plasticity"][seed])
        and math.isfinite(metrics["retention"][seed])
        for seed in FINAL_SEEDS
    ):
        return None
    return root, arm


def run_train(args: argparse.Namespace) -> None:
    assert_authoritative_protocol()
    selected = _load_selected(args.output_dir)
    selected_config = {
        "k": float(selected["selected_k"]),
        "T_m": float(selected["selected_T_m"]),
    }
    runtime = _runtime(args)
    _run_configuration(
        args=args,
        runtime=runtime,
        root=args.output_dir / "train" / "tuned",
        config=selected_config,
        seeds=FINAL_SEEDS,
        phase="final_train",
        label="tuned",
    )
    if _config_id(selected_config) != _config_id(CENTER):
        existing_center = _existing_v25_center()
        if existing_center is None:
            _run_configuration(
                args=args,
                runtime=runtime,
                root=args.output_dir / "train" / "center_control",
                config=CENTER,
                seeds=FINAL_SEEDS,
                phase="center_control",
                label="center_control",
            )
        else:
            root, arm = existing_center
            reference = {
                "experiment_version": EXPERIMENT_VERSION,
                "source_root": str(root.resolve()),
                "source_arm": arm,
                "cap_config": _full_cap_config(CENTER),
                "protocol_hash": protocol_hash(),
            }
            reference["reference_hash"] = _hash_payload(reference)
            _immutable_json(
                args.output_dir / "train" / "center_control_reference.json",
                reference,
            )
    print(f"Completed final CAP training under {args.output_dir / 'train'}")


def _final_method_roots(output_dir: Path) -> dict[str, tuple[Path, str]]:
    result = {
        "cap_tuned": (
            output_dir / "train" / "tuned",
            "cap_release_final_tuned",
        )
    }
    center = output_dir / "train" / "center_control"
    if center.exists():
        result["cap_center"] = (center, "cap_release_final_center_control")
    else:
        reference_path = output_dir / "train" / "center_control_reference.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            claimed = reference.pop("reference_hash")
            observed = _hash_payload(reference)
            if claimed != observed:
                raise RuntimeError("center-control reference hash mismatch")
            existing_center = _existing_v25_center()
            if existing_center is None:
                raise RuntimeError("referenced v2.5 center control is no longer valid")
            validated_root, validated_arm = existing_center
            if (
                Path(str(reference["source_root"])).resolve()
                != validated_root.resolve()
                or str(reference["source_arm"]) != validated_arm
            ):
                raise RuntimeError("center-control reference identity mismatch")
            result["cap_center"] = (
                validated_root,
                validated_arm,
            )
    return result


def _external_metrics() -> dict[str, dict[str, dict[int, float]]]:
    scalar_path, pool_path = _external_result_paths()
    if not scalar_path.exists() or not pool_path.exists():
        raise FileNotFoundError(
            "authoritative post_phase1_v2 baseline outputs are required for analysis"
        )
    _validate_external_protocol()
    plasticity = stream_a_plasticity(read_csv(scalar_path))
    retention = stream_b.end_retention_by_seed(
        read_csv(pool_path), pool_size=POST_P1_POOL_SIZE
    )
    result: dict[str, dict[str, dict[int, float]]] = {}
    for method in ("adamw", "upgd", "ewc_oracle", "adam_replay"):
        arm = _external_arm(method)
        if arm in plasticity and arm in retention:
            result[method] = {
                "plasticity": plasticity[arm],
                "retention": retention[arm],
            }
    required = {"adamw", "upgd", "ewc_oracle"}
    if not required <= set(result):
        raise RuntimeError("authoritative baseline outputs lack AdamW, UPGD, or EWC")
    for method, metrics in result.items():
        if set(metrics["plasticity"]) != set(FINAL_SEEDS) or set(
            metrics["retention"]
        ) != set(FINAL_SEEDS):
            raise RuntimeError(f"incomplete five-seed baseline metrics for {method}")
    return result


def _external_result_paths() -> tuple[Path, Path]:
    return (
        POST_DIR / "stream_a" / "stream_scalars.csv",
        POST_DIR / "stream_b" / "streamb_pool.csv",
    )


def _external_arm(method: str) -> str:
    return f"{method}_retention"


def _validate_external_protocol() -> None:
    config_a_path = POST_DIR / "stream_a" / "post_run_config.json"
    config_b_path = POST_DIR / "stream_b" / "post_run_config.json"
    if not config_a_path.exists() or not config_b_path.exists():
        raise FileNotFoundError(
            "authoritative post_phase1_v2 run configurations are required"
        )
    config_a = json.loads(config_a_path.read_text(encoding="utf-8"))
    config_b = json.loads(config_b_path.read_text(encoding="utf-8"))
    for name, config, stream in (
        ("Stream A", config_a, "A"),
        ("Stream B", config_b, "B"),
    ):
        if (
            config.get("isolation") != V2_ISOLATION
            or config.get("stream") != stream
            or tuple(config.get("seeds", ())) != FINAL_SEEDS
        ):
            raise RuntimeError(f"{name} baseline protocol identity mismatch")
    if (
        int(config_b.get("pool_size", -1)) != POST_P1_POOL_SIZE
        or config_b.get("schedule") != block_stream_b_schedule(POST_P1_POOL_SIZE)
    ):
        raise RuntimeError("Stream B baseline pool/schedule protocol mismatch")


def _retention_gap_rows(
    method: str, pool_rows: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    schedule = block_stream_b_schedule(POST_P1_POOL_SIZE)
    gaps: dict[tuple[int, int], int] = {}
    last_visit: dict[int, int] = {}
    for task, current in enumerate(schedule, start=1):
        for index, previous in last_visit.items():
            if index != current:
                gaps[(task, index)] = task - previous
        last_visit[int(current)] = task
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in pool_rows:
        if str(row["arm"]) != arm or str(row["is_current"]).lower() in ("true", "1"):
            continue
        gap = gaps.get((int(float(row["task"])), int(float(row["pool_index"]))))
        if gap is None:
            continue
        for label, lower, upper in GAP_BUCKETS:
            if lower <= gap <= upper:
                grouped[(int(float(row["seed"])), label)].append(float(row["accuracy"]))
                break
    return [
        {
            "method": method,
            "seed": seed,
            "gap_bucket": bucket,
            "mean_retention": _mean(values),
            "count": len(values),
        }
        for (seed, bucket), values in sorted(grouped.items())
    ]


def _temporal_rows(
    method: str, scalar_rows: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in scalar_rows:
        if str(row["arm"]) != arm:
            continue
        seed = int(float(row["seed"]))
        step = int(float(row["step"]))
        accuracy = float(row["batch_accuracy"])
        for label, lower, upper in TEMPORAL_WINDOWS:
            if lower <= step <= upper:
                grouped[(seed, "stream_window", label)].append(accuracy)
                break
        local_step = ((step - 1) % stream_a.STEPS_PER_TASK) + 1
        if local_step in WITHIN_TASK_STEPS:
            grouped[(seed, "within_task_step", str(local_step))].append(accuracy)
            if step > stream_a.TOTAL_STEPS // 2:
                grouped[(seed, "late_within_task_step", str(local_step))].append(
                    accuracy
                )
    return [
        {
            "method": method,
            "seed": seed,
            "summary_type": summary_type,
            "window_or_step": label,
            "mean_accuracy": _mean(values),
            "count": len(values),
        }
        for (seed, summary_type, label), values in sorted(grouped.items())
    ]


def _plot_frontier(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for the frontier figure") from error
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(
            (float(row["retention"]), float(row["plasticity"]))
        )
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for method, values in sorted(grouped.items()):
        retention = [value[0] for value in values]
        plasticity = [value[1] for value in values]
        axis.errorbar(
            _mean(retention),
            _mean(plasticity),
            xerr=_sd(retention),
            yerr=_sd(plasticity),
            marker="o",
            capsize=3,
            linestyle="none",
            label=method,
        )
    axis.set_xlabel("Stream-B final non-current retention")
    axis.set_ylabel("Stream-A second-half plasticity")
    axis.set_title("CAP stability-plasticity frontier")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analyze(args: argparse.Namespace) -> None:
    selected = _load_selected(args.output_dir)
    _validate_retune_summary(args.output_dir, selected)
    analysis_root = args.output_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    method_roots = _final_method_roots(args.output_dir)
    final_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for method, (root, arm) in method_roots.items():
        metrics = _arm_metrics(root, arm)
        for seed in FINAL_SEEDS:
            if seed not in metrics.get("plasticity", {}) or seed not in metrics.get(
                "retention", {}
            ):
                raise RuntimeError(f"final training is incomplete for {method}/seed{seed}")
            plasticity = metrics["plasticity"][seed]
            retention = metrics["retention"][seed]
            if not math.isfinite(plasticity) or not math.isfinite(retention):
                raise RuntimeError(f"non-finite final metric for {method}/seed{seed}")
            final_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "plasticity": plasticity,
                    "retention": retention,
                    "seedwise_min": min(plasticity, retention),
                }
            )
        pool = read_csv(root / "stream_b" / "streamb_pool.csv")
        scalars = read_csv(root / "stream_a" / "stream_scalars.csv")
        gap_rows.extend(_retention_gap_rows(method, pool, arm))
        temporal_rows.extend(_temporal_rows(method, scalars, arm))
        diagnostics = _diagnostic_means(root, arm)
        for seed in FINAL_SEEDS:
            seed_diagnostics = diagnostics.get(seed, {})
            required_diagnostics = {
                "bind_rate",
                "median_imp",
                "median_radius",
                "frozen_fraction",
                "theta_norm",
            }
            if not required_diagnostics <= set(seed_diagnostics) or seed not in metrics.get(
                "current_task_accuracy", {}
            ):
                raise RuntimeError(f"final diagnostics are incomplete for {method}/seed{seed}")
            diagnostic_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **seed_diagnostics,
                    "current_task_accuracy": metrics.get(
                        "current_task_accuracy", {}
                    ).get(seed, ""),
                }
            )

    external = _external_metrics()
    for method, metrics in external.items():
        for seed in FINAL_SEEDS:
            if seed in metrics["plasticity"] and seed in metrics["retention"]:
                plasticity = metrics["plasticity"][seed]
                retention = metrics["retention"][seed]
                if not math.isfinite(plasticity) or not math.isfinite(retention):
                    raise RuntimeError(
                        f"non-finite external metric for {method}/seed{seed}"
                    )
                final_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "plasticity": plasticity,
                        "retention": retention,
                        "seedwise_min": min(plasticity, retention),
                    }
                )
    scalar_path, pool_path = _external_result_paths()
    if external and scalar_path.exists() and pool_path.exists():
        scalar_rows = read_csv(scalar_path)
        pool_rows = read_csv(pool_path)
        for method in external:
            arm = _external_arm(method)
            gap_rows.extend(_retention_gap_rows(method, pool_rows, arm))
            temporal_rows.extend(_temporal_rows(method, scalar_rows, arm))

    paired_rows: list[dict[str, Any]] = []
    by_method_seed = {
        (str(row["method"]), int(row["seed"])): row for row in final_rows
    }
    if "cap_center" in method_roots:
        for seed in FINAL_SEEDS:
            tuned = by_method_seed[("cap_tuned", seed)]
            center = by_method_seed[("cap_center", seed)]
            paired_rows.append(
                {
                    "seed": seed,
                    "delta_plasticity": float(tuned["plasticity"]) - float(center["plasticity"]),
                    "delta_retention": float(tuned["retention"]) - float(center["retention"]),
                    "delta_seedwise_min": float(tuned["seedwise_min"]) - float(center["seedwise_min"]),
                }
            )

    write_csv(analysis_root / "final_metrics.csv", final_rows, FINAL_METRIC_FIELDS)
    write_csv(analysis_root / "paired_retune_deltas.csv", paired_rows, PAIRED_DELTA_FIELDS)
    write_csv(analysis_root / "retention_by_gap.csv", gap_rows, GAP_FIELDS)
    write_csv(analysis_root / "temporal_plasticity.csv", temporal_rows, TEMPORAL_FIELDS)
    write_csv(
        analysis_root / "diagnostic_summary.csv",
        diagnostic_rows,
        DIAGNOSTIC_SUMMARY_FIELDS,
    )
    _plot_frontier(final_rows, analysis_root / "stability_plasticity_frontier.png")

    grouped_final: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped_final[str(row["method"])].append(row)
    lines = [
        "# Final revised-Release CAP analysis",
        "",
        f"Selected configuration: `{selected['selected_config_id']}` ",
        f"(`k={selected['selected_k']}`, `T_m={selected['selected_T_m']}`).",
        f"Grid-boundary selection: **{'yes' if selected['boundary_selected'] else 'no'}**. ",
        "The bounded grid was not expanded.",
        "",
        "| Method | P mean +/- SD | R mean +/- SD | B mean +/- SD | min(mean P, mean R) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, rows in sorted(grouped_final.items()):
        plasticity = [float(row["plasticity"]) for row in rows]
        retention = [float(row["retention"]) for row in rows]
        balanced = [float(row["seedwise_min"]) for row in rows]
        lines.append(
            f"| {method} | {_mean(plasticity):.6f} +/- {_sd(plasticity):.6f} | "
            f"{_mean(retention):.6f} +/- {_sd(retention):.6f} | "
            f"{_mean(balanced):.6f} +/- {_sd(balanced):.6f} | "
            f"{min(_mean(plasticity), _mean(retention)):.6f} |"
        )
    upgd_rows = grouped_final.get("upgd", [])
    tuned_rows = grouped_final["cap_tuned"]
    if upgd_rows:
        upgd_p = _mean([float(row["plasticity"]) for row in upgd_rows])
        upgd_r = _mean([float(row["retention"]) for row in upgd_rows])
        upgd_b = _mean([float(row["seedwise_min"]) for row in upgd_rows])
        lines += [
            "",
            "## Gap to UPGD",
            "",
            "| Method | P gap | R gap | B gap |",
            "|---|---:|---:|---:|",
        ]
        for method, rows in sorted(grouped_final.items()):
            method_p = _mean([float(row["plasticity"]) for row in rows])
            method_r = _mean([float(row["retention"]) for row in rows])
            method_b = _mean([float(row["seedwise_min"]) for row in rows])
            lines.append(
                f"| {method} | {method_p - upgd_p:+.6f} | "
                f"{method_r - upgd_r:+.6f} | {method_b - upgd_b:+.6f} |"
            )
    lines += [
        "",
        "## Tuned CAP seed values",
        "",
        "| Seed | P | R | B=min(P,R) |",
        "|---:|---:|---:|---:|",
    ]
    for row in sorted(tuned_rows, key=lambda item: int(item["seed"])):
        lines.append(
            f"| {int(row['seed'])} | {float(row['plasticity']):.6f} | "
            f"{float(row['retention']):.6f} | "
            f"{float(row['seedwise_min']):.6f} |"
        )
    lines += [
        "",
        "## Retention by revisit gap",
        "",
        "| Method | Gap bucket | Mean retention |",
        "|---|---|---:|",
    ]
    grouped_gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in gap_rows:
        grouped_gaps[(str(row["method"]), str(row["gap_bucket"]))].append(
            float(row["mean_retention"])
        )
    for (method, bucket), values in sorted(grouped_gaps.items()):
        lines.append(f"| {method} | {bucket} | {_mean(values):.6f} |")

    lines += [
        "",
        "## Stream-A temporal plasticity",
        "",
        "| Method | Window | Mean batch accuracy |",
        "|---|---|---:|",
    ]
    grouped_windows: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in temporal_rows:
        if str(row["summary_type"]) == "stream_window":
            grouped_windows[(str(row["method"]), str(row["window_or_step"]))].append(
                float(row["mean_accuracy"])
            )
    for (method, window), values in sorted(grouped_windows.items()):
        lines.append(f"| {method} | {window} | {_mean(values):.6f} |")

    lines += [
        "",
        "## CAP diagnostics",
        "",
        "| Method | Bind rate | Median importance | Median radius | Frozen fraction | Parameter norm | Current-task accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(method_roots):
        rows = [row for row in diagnostic_rows if row["method"] == method]
        values = {}
        for field in (
            "bind_rate",
            "median_imp",
            "median_radius",
            "frozen_fraction",
            "theta_norm",
            "current_task_accuracy",
        ):
            observed = [
                float(row[field]) for row in rows if row.get(field, "") != ""
            ]
            values[field] = _mean(observed)
        lines.append(
            f"| {method} | {values['bind_rate']:.6f} | "
            f"{values['median_imp']:.6f} | {values['median_radius']:.6f} | "
            f"{values['frozen_fraction']:.6f} | {values['theta_norm']:.6f} | "
            f"{values['current_task_accuracy']:.6f} |"
        )
    if paired_rows:
        delta_p = _mean([float(row["delta_plasticity"]) for row in paired_rows])
        delta_r = _mean([float(row["delta_retention"]) for row in paired_rows])
        delta_min = _mean([float(row["delta_seedwise_min"]) for row in paired_rows])
        effect_tolerance = 1e-6
        if max(abs(delta_p), abs(delta_r), abs(delta_min)) < effect_tolerance:
            decision = "Outcome C: retain the center configuration for simplicity."
            adopt_selected = False
        elif (
            delta_min > effect_tolerance
            and delta_p >= -effect_tolerance
            and delta_r >= -effect_tolerance
        ):
            decision = "Outcome A: adopt the tuned configuration as final CAP."
            adopt_selected = True
        else:
            decision = "Outcome B: retain the center configuration; retuning did not improve balance cleanly."
            adopt_selected = False
        lines += [
            "",
            "## Paired retuning effect",
            "",
            f"Mean delta P: {delta_p:+.6f}",
            "",
            f"Mean delta R: {delta_r:+.6f}",
            "",
            f"Mean delta seedwise min: {delta_min:+.6f}",
            "",
            decision,
        ]
    else:
        decision = "Selected configuration is the center; no duplicate control was required."
        adopt_selected = True
        lines += ["", decision]
    report = "\n".join(lines) + "\n"
    (analysis_root / "report.md").write_text(report, encoding="utf-8")
    final_decision = {
        "experiment_version": EXPERIMENT_VERSION,
        "selected_config_id": selected["selected_config_id"],
        "adopted_config_id": (
            selected["selected_config_id"]
            if adopt_selected
            else _config_id(CENTER)
        ),
        "final_cap_config": (
            selected["cap_config"]
            if adopt_selected
            else _full_cap_config(CENTER)
        ),
        "decision": decision,
        "algorithm_development_stop_rule": True,
    }
    final_decision["decision_hash"] = _hash_payload(final_decision)
    _immutable_json(analysis_root / "final_decision.json", final_decision)
    print(f"Wrote final CAP analysis under {analysis_root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.retune:
        run_retune(args)
    elif args.train:
        run_train(args)
    else:
        run_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
