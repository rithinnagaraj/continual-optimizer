"""Analyze completed CAP-v2.5 conflict-momentum-restart experiments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import experiment_e4b as stream_a_engine
from experiment_e2 import read_csv, write_csv
from experiment_stream_b import end_retention_by_seed
from post_phase1_report import stream_a_plasticity
from cap_v2_postp1_common import (
    DEFAULT_OUTPUT_ROOT,
    FULL_SEEDS,
    MODES,
    POST_P1_POOL_SIZE,
    add_seed_arguments,
    arm_name,
    block_stream_b_schedule,
    resolve_seeds,
)


SUMMARY_FIELDS = (
    "variant",
    "metric",
    "seed_count",
    "status",
    "mean",
    "sample_sd",
    "ci95_low",
    "ci95_high",
    "seed_values_json",
)
DIFFERENCE_FIELDS = (
    "comparison",
    "metric",
    "seed_count",
    "status",
    "mean_difference",
    "sample_sd",
    "ci95_low",
    "ci95_high",
    "seed_differences_json",
)
T95 = {1: 12.706204736, 2: 4.30265273, 3: 3.182446305, 4: 2.776445105}
GAP_BUCKETS = (
    ("1_5", 1, 5),
    ("6_10", 6, 10),
    ("11_20", 11, 20),
    ("21_40", 21, 40),
    ("41_80", 41, 80),
)
WITHIN_TASK_STEPS = (100, 200, 300, 500, 1_000, 1_500, 2_000, 5_000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_seed_arguments(parser)
    args = parser.parse_args(argv)
    try:
        args.resolved_seeds = resolve_seeds(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _summary(values: Mapping[int, float], seeds: Sequence[int]) -> dict[str, Any]:
    ordered = [float(values[seed]) for seed in FULL_SEEDS if seed in seeds and seed in values]
    if not ordered:
        raise RuntimeError("cannot summarize an empty seed set")
    deviation = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    low: float | str = ""
    high: float | str = ""
    if len(ordered) == len(FULL_SEEDS):
        margin = T95[len(ordered) - 1] * deviation / math.sqrt(len(ordered))
        low = statistics.mean(ordered) - margin
        high = statistics.mean(ordered) + margin
    return {
        "seed_count": len(ordered),
        "status": "FINAL" if len(ordered) == len(FULL_SEEDS) else "PRELIMINARY",
        "mean": statistics.mean(ordered),
        "sample_sd": deviation,
        "ci95_low": low,
        "ci95_high": high,
        "values": {
            seed: float(values[seed])
            for seed in FULL_SEEDS
            if seed in seeds and seed in values
        },
    }


def _mean_by_seed(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field, "")
        if value != "":
            grouped[int(float(row["seed"]))].append(float(value))
    return {seed: statistics.mean(values) for seed, values in grouped.items()}


def _ratio_by_seed(
    rows: Sequence[Mapping[str, Any]], numerator: str, denominator: str
) -> dict[int, float]:
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        seed = int(float(row["seed"]))
        totals[seed][0] += float(row[numerator])
        totals[seed][1] += float(row[denominator])
    return {
        seed: numerator_total / max(denominator_total, 1.0)
        for seed, (numerator_total, denominator_total) in totals.items()
    }


def _retention_by_revisit_gap(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, float]]:
    schedule = block_stream_b_schedule(POST_P1_POOL_SIZE)
    gaps: dict[tuple[int, int], int] = {}
    last_visit: dict[int, int] = {}
    for task, current in enumerate(schedule, start=1):
        for pool_index, previous in last_visit.items():
            if pool_index != current:
                gaps[(task, pool_index)] = task - previous
        last_visit[int(current)] = task

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("is_current", "")).lower() in ("true", "1"):
            continue
        gap = gaps.get((int(float(row["task"])), int(float(row["pool_index"]))))
        if gap is None:
            continue
        for label, lower, upper in GAP_BUCKETS:
            if lower <= gap <= upper:
                grouped[(label, int(float(row["seed"])))].append(float(row["accuracy"]))
                break
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (label, seed), values in grouped.items():
        result[label][seed] = statistics.mean(values)
    return dict(result)


def _within_task_accuracy(
    rows: Sequence[Mapping[str, Any]], *, late_only: bool
) -> dict[int, dict[int, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        step = int(float(row["step"]))
        task = ((step - 1) // stream_a_engine.STEPS_PER_TASK) + 1
        if late_only and task <= stream_a_engine.TASK_COUNT // 2:
            continue
        local_step = ((step - 1) % stream_a_engine.STEPS_PER_TASK) + 1
        if local_step in WITHIN_TASK_STEPS:
            grouped[(local_step, int(float(row["seed"])))].append(
                float(row["batch_accuracy"])
            )
    result: dict[int, dict[int, float]] = defaultdict(dict)
    for (local_step, seed), values in grouped.items():
        result[local_step][seed] = statistics.mean(values)
    return dict(result)


def _variant_metrics(results_dir: Path, variant: str) -> dict[str, dict[int, float]]:
    arm = arm_name(variant)
    root = results_dir / f"v25_{variant}"
    root_a = root / "stream_a"
    root_b = root / "stream_b"
    metrics: dict[str, dict[int, float]] = {}

    scalar_path = root_a / "stream_scalars.csv"
    if scalar_path.exists():
        scalar_rows = [row for row in read_csv(scalar_path) if str(row["arm"]) == arm]
        plasticity = stream_a_plasticity(scalar_rows)
        if arm in plasticity:
            metrics["stream_a_plasticity"] = plasticity[arm]
        for local_step, values in _within_task_accuracy(
            scalar_rows, late_only=False
        ).items():
            metrics[f"stream_a_accuracy_step_{local_step}"] = values
        for local_step, values in _within_task_accuracy(
            scalar_rows, late_only=True
        ).items():
            metrics[f"late_stream_a_accuracy_step_{local_step}"] = values

    pool_path = root_b / "streamb_pool.csv"
    if pool_path.exists():
        pool_rows = [row for row in read_csv(pool_path) if str(row["arm"]) == arm]
        retention = end_retention_by_seed(pool_rows, pool_size=POST_P1_POOL_SIZE)
        if arm in retention:
            metrics["stream_b_end_retention"] = retention[arm]
        for label, values in _retention_by_revisit_gap(pool_rows).items():
            metrics[f"stream_b_retention_gap_{label}"] = values

    task_path = root_b / "streamb_tasks.csv"
    if task_path.exists():
        task_rows = [row for row in read_csv(task_path) if str(row["arm"]) == arm]
        metrics["stream_b_current_task_mean"] = _mean_by_seed(
            task_rows, "task_end_accuracy"
        )

    diagnostic_rows: list[dict[str, Any]] = []
    for stream_root in (root_a, root_b):
        path = stream_root / "cap_v2_diagnostics.csv"
        if path.exists():
            diagnostic_rows.extend(
                row for row in read_csv(path) if str(row["variant"]) == variant
            )
    if diagnostic_rows:
        for metric, field in (
            ("bind_rate", "bind_rate"),
            ("rebind_given_current_bind", "rebind_given_current_bind"),
            ("rebind_given_previous_bind", "rebind_given_previous_bind"),
            (
                "momentum_conflict_given_previous_bind",
                "momentum_conflict_given_previous_bind",
            ),
            ("median_importance", "median_imp"),
            ("median_radius", "median_radius"),
            ("parameter_norm", "theta_norm"),
        ):
            metrics[metric] = _mean_by_seed(diagnostic_rows, field)
        metrics["momentum_restart_rate"] = _ratio_by_seed(
            diagnostic_rows, "momentum_restart_count", "previous_bind_count"
        )

    if "stream_a_plasticity" in metrics and "stream_b_end_retention" in metrics:
        metrics["minimum_plasticity_retention"] = {
            seed: min(
                metrics["stream_a_plasticity"][seed],
                metrics["stream_b_end_retention"][seed],
            )
            for seed in set(metrics["stream_a_plasticity"])
            & set(metrics["stream_b_end_retention"])
        }
    return {metric: values for metric, values in metrics.items() if values}


def analyze(results_dir: Path, seeds: Sequence[int]) -> tuple[list[dict], list[dict], str]:
    metrics = {
        variant: values
        for variant in MODES
        if (values := _variant_metrics(results_dir, variant))
    }
    if not metrics:
        raise RuntimeError(f"no CAP-v2.5 results found under {results_dir}")

    summary_rows: list[dict[str, Any]] = []
    for variant, variant_metrics in metrics.items():
        for metric, values in variant_metrics.items():
            selected = {seed: values[seed] for seed in seeds if seed in values}
            if not selected:
                continue
            summary = _summary(selected, seeds)
            summary_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    **{key: value for key, value in summary.items() if key != "values"},
                    "seed_values_json": json.dumps(summary["values"], sort_keys=True),
                }
            )

    difference_rows: list[dict[str, Any]] = []
    restart = metrics.get("conflict_momentum_restart", {})
    baseline = metrics.get("baseline", {})
    for metric in sorted(set(restart) & set(baseline)):
        differences = {
            seed: restart[metric][seed] - baseline[metric][seed]
            for seed in FULL_SEEDS
            if seed in seeds and seed in restart[metric] and seed in baseline[metric]
        }
        if not differences:
            continue
        summary = _summary(differences, seeds)
        difference_rows.append(
            {
                "comparison": "conflict_momentum_restart-baseline",
                "metric": metric,
                "seed_count": summary["seed_count"],
                "status": summary["status"],
                "mean_difference": summary["mean"],
                "sample_sd": summary["sample_sd"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "seed_differences_json": json.dumps(summary["values"], sort_keys=True),
            }
        )

    complete = all(
        variant in metrics
        and all(
            metric in metrics[variant]
            and all(seed in metrics[variant][metric] for seed in FULL_SEEDS)
            for metric in ("stream_a_plasticity", "stream_b_end_retention")
        )
        for variant in MODES
    )
    status = (
        "FINAL"
        if set(seeds) == set(FULL_SEEDS) and len(seeds) == len(FULL_SEEDS) and complete
        else "PRELIMINARY"
    )
    lines = [
        "# CAP-v2.5 Conflict Momentum Restart",
        "",
        f"Status: **{status}**.",
        "",
        "## Metrics",
        "",
        "| Variant | Metric | n | Mean | Sample SD | 95% t CI | Seed values |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        ci = (
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
            if row["ci95_low"] != ""
            else "PRELIMINARY"
        )
        lines.append(
            f"| {row['variant']} | {row['metric']} | {row['seed_count']} | "
            f"{float(row['mean']):.6f} | {float(row['sample_sd']):.6f} | "
            f"{ci} | `{row['seed_values_json']}` |"
        )
    lines += [
        "",
        "## Restart minus baseline",
        "",
        "| Metric | n | Mean difference | Sample SD | 95% t CI |",
        "|---|---:|---:|---:|---|",
    ]
    for row in difference_rows:
        ci = (
            f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
            if row["ci95_low"] != ""
            else "PRELIMINARY"
        )
        lines.append(
            f"| {row['metric']} | {row['seed_count']} | "
            f"{float(row['mean_difference']):+.6f} | "
            f"{float(row['sample_sd']):.6f} | {ci} |"
        )
    return summary_rows, difference_rows, "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_rows, difference_rows, report = analyze(args.results_dir, args.resolved_seeds)
    analysis_dir = args.results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(
        analysis_dir / "paired_differences.csv", difference_rows, DIFFERENCE_FIELDS
    )
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote CAP-v2.5 analysis to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
