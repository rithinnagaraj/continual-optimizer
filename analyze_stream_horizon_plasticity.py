"""Analyze horizon-normalized Stream-A plasticity from existing 1M runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
EXPERIMENT_VERSION = "cap-stream-horizon-plasticity-v1"
FINAL_SEEDS = (42, 123, 2026, 451, 789)
HORIZONS = (100_000, 250_000, 500_000, 750_000, 1_000_000)
SCALAR_LOG_EVERY = 100
STEPS_PER_TASK = 5_000
EXPECTED_FINAL_MEANS = {"cap": 0.902670, "upgd": 0.912714}
EXPECTED_MEAN_TOLERANCE = 5e-6
REFERENCE_TOLERANCE = 1e-12
T_CRITICAL_DF4_95 = 2.7764451051977987
FIXED_WINDOWS = (
    ("1-100k", 0, 100_000),
    ("101-250k", 100_000, 250_000),
    ("251-500k", 250_000, 500_000),
    ("501-750k", 500_000, 750_000),
    ("751k-1M", 750_000, 1_000_000),
)

DEFAULT_CAP_DIR = ROOT / "results" / "cap_release_final" / "train" / "tuned" / "stream_a"
DEFAULT_BASELINE_DIR = ROOT / "results" / "post_phase1_v2" / "stream_a"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "cap_release_final" / "analysis" / "stream_horizon_plasticity"
)

PER_SEED_FIELDS = (
    "method",
    "seed",
    "horizon",
    "window_start",
    "window_end",
    "observation_count",
    "plasticity",
)
SUMMARY_FIELDS = (
    "method",
    "horizon",
    "mean_plasticity",
    "std_plasticity",
    "min_seed_value",
    "max_seed_value",
    "seed_values_json",
)
DELTA_FIELDS = (
    "seed",
    "horizon",
    "cap_plasticity",
    "upgd_plasticity",
    "delta_cap_minus_upgd",
    "mean_paired_delta",
    "std_paired_delta",
    "ci95_low",
    "ci95_high",
    "seed_deltas_json",
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    directory: Path
    arm: str


@dataclass
class LoadedTrajectory:
    spec: MethodSpec
    scalar_path: Path
    values: dict[int, dict[int, float]]
    duplicate_rows_removed: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-dir", type=Path, default=DEFAULT_CAP_DIR)
    parser.add_argument("--upgd-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-metrics", type=Path)
    parser.add_argument("--cap-arm", default="cap_release_final_tuned")
    parser.add_argument("--upgd-arm", default="upgd_retention")
    parser.add_argument("--include-adamw", action="store_true")
    parser.add_argument("--include-ewc", action="store_true")
    parser.add_argument("--include-replay", action="store_true")
    return parser.parse_args(argv)


def _scalar_path(directory: Path) -> Path:
    if directory.is_file():
        return directory
    candidates = (
        directory / "stream_scalars.csv",
        directory / "stream_a" / "stream_scalars.csv",
    )
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"expected exactly one Stream-A scalar file under {directory}; checked {rendered}"
        )
    return existing[0]


def _integer_field(row: Mapping[str, str], field: str, context: str) -> int:
    value = float(row[field])
    if not math.isfinite(value) or not value.is_integer():
        raise RuntimeError(f"invalid integer {field} in {context}: {row[field]!r}")
    return int(value)


def load_trajectory(spec: MethodSpec) -> LoadedTrajectory:
    path = _scalar_path(spec.directory)
    values: dict[int, dict[int, float]] = defaultdict(dict)
    available_arms: set[str] = set()
    duplicate_rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"arm", "seed", "step", "batch_accuracy"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise RuntimeError(f"{path} lacks required fields {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            arm = str(row["arm"])
            available_arms.add(arm)
            if arm != spec.arm:
                continue
            context = f"{path}:{row_number}"
            seed = _integer_field(row, "seed", context)
            step = _integer_field(row, "step", context)
            accuracy = float(row["batch_accuracy"])
            if step <= 0 or not math.isfinite(accuracy):
                raise RuntimeError(f"invalid scalar observation in {context}")
            prior = values[seed].get(step)
            if prior is not None:
                if prior != accuracy:
                    raise RuntimeError(
                        f"conflicting duplicate for {spec.arm}/seed{seed}/step{step}"
                    )
                duplicate_rows += 1
                continue
            values[seed][step] = accuracy
    if not values:
        raise RuntimeError(
            f"arm {spec.arm!r} is absent from {path}; available arms: "
            + ", ".join(sorted(available_arms))
        )
    return LoadedTrajectory(spec, path, dict(values), duplicate_rows)


def validate_trajectory(trajectory: LoadedTrajectory) -> None:
    observed_seeds = set(trajectory.values)
    if observed_seeds != set(FINAL_SEEDS):
        raise RuntimeError(
            f"{trajectory.spec.key} seed mismatch: expected {FINAL_SEEDS}, "
            f"observed {sorted(observed_seeds)}"
        )
    expected_steps = set(range(SCALAR_LOG_EVERY, HORIZONS[-1] + 1, SCALAR_LOG_EVERY))
    for seed in FINAL_SEEDS:
        observed_steps = set(trajectory.values[seed])
        if observed_steps != expected_steps:
            missing = sorted(expected_steps - observed_steps)
            extra = sorted(observed_steps - expected_steps)
            raise RuntimeError(
                f"incomplete/noncanonical trajectory for {trajectory.spec.key}/seed{seed}; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )


def horizon_plasticity(
    values: Mapping[int, Mapping[int, float]], method: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        if seed not in values:
            raise RuntimeError(f"missing {method} trajectory for seed{seed}")
        for horizon in HORIZONS:
            start = horizon // 2
            expected_steps = list(
                range(start + SCALAR_LOG_EVERY, horizon + 1, SCALAR_LOG_EVERY)
            )
            selected = [
                accuracy
                for step, accuracy in values[seed].items()
                if step > start and step <= horizon
            ]
            actual_steps = sorted(
                step for step in values[seed] if step > start and step <= horizon
            )
            if actual_steps != expected_steps:
                raise RuntimeError(
                    f"incorrect horizon window coverage for {method}/seed{seed}/H={horizon}"
                )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "horizon": horizon,
                    "window_start": start,
                    "window_end": horizon,
                    "observation_count": len(selected),
                    "plasticity": statistics.mean(selected),
                }
            )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["method"]), int(row["horizon"]))
        seed = int(row["seed"])
        if seed in grouped[key]:
            raise RuntimeError(f"duplicate horizon result for {key}/seed{seed}")
        grouped[key][seed] = float(row["plasticity"])
    result: list[dict[str, Any]] = []
    for (method, horizon), seed_values in sorted(grouped.items()):
        if set(seed_values) != set(FINAL_SEEDS):
            raise RuntimeError(f"incomplete summary seed coverage for {method}/H={horizon}")
        ordered = [seed_values[seed] for seed in FINAL_SEEDS]
        result.append(
            {
                "method": method,
                "horizon": horizon,
                "mean_plasticity": statistics.mean(ordered),
                "std_plasticity": statistics.stdev(ordered),
                "min_seed_value": min(ordered),
                "max_seed_value": max(ordered),
                "seed_values_json": json.dumps(seed_values, sort_keys=True),
            }
        )
    return result


def paired_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["method"]), int(row["seed"]), int(row["horizon"])): float(
            row["plasticity"]
        )
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        deltas: dict[int, float] = {}
        pairs: dict[int, tuple[float, float]] = {}
        for seed in FINAL_SEEDS:
            cap_key = ("cap", seed, horizon)
            upgd_key = ("upgd", seed, horizon)
            if cap_key not in indexed or upgd_key not in indexed:
                raise RuntimeError(f"CAP/UPGD pairing is incomplete for seed{seed}/H={horizon}")
            cap_value = indexed[cap_key]
            upgd_value = indexed[upgd_key]
            pairs[seed] = (cap_value, upgd_value)
            deltas[seed] = cap_value - upgd_value
        ordered = [deltas[seed] for seed in FINAL_SEEDS]
        mean_delta = statistics.mean(ordered)
        sd_delta = statistics.stdev(ordered)
        margin = T_CRITICAL_DF4_95 * sd_delta / math.sqrt(len(ordered))
        seed_json = json.dumps(deltas, sort_keys=True)
        for seed in FINAL_SEEDS:
            cap_value, upgd_value = pairs[seed]
            result.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "cap_plasticity": cap_value,
                    "upgd_plasticity": upgd_value,
                    "delta_cap_minus_upgd": deltas[seed],
                    "mean_paired_delta": mean_delta,
                    "std_paired_delta": sd_delta,
                    "ci95_low": mean_delta - margin,
                    "ci95_high": mean_delta + margin,
                    "seed_deltas_json": seed_json,
                }
            )
    return result


def _find_final_metrics(explicit: Path | None, cap_scalar_path: Path) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"final metrics file does not exist: {explicit}")
        return explicit
    candidates = [
        parent / "analysis" / "final_metrics.csv"
        for parent in (cap_scalar_path.parent, *cap_scalar_path.parents)
    ]
    candidates.append(
        ROOT / "results" / "cap_release_final" / "analysis" / "final_metrics.csv"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not locate final_metrics.csv; pass it explicitly with --final-metrics"
    )


def _reference_plasticity(path: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {"cap": {}, "upgd": {}}
    method_map = {"cap_tuned": "cap", "upgd": "upgd"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "seed", "plasticity"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise RuntimeError(f"{path} lacks required final metric fields")
        for row in reader:
            key = method_map.get(str(row["method"]))
            if key is None:
                continue
            seed = _integer_field(row, "seed", str(path))
            value = float(row["plasticity"])
            if seed in result[key]:
                raise RuntimeError(f"duplicate final metric for {key}/seed{seed}")
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite final metric for {key}/seed{seed}")
            result[key][seed] = value
    for method, values in result.items():
        if set(values) != set(FINAL_SEEDS):
            raise RuntimeError(f"final_metrics.csv lacks five-seed {method} coverage")
    return result


def validate_one_million_equivalence(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[int, float]],
    *,
    expected_means: Mapping[str, float] = EXPECTED_FINAL_MEANS,
    reference_tolerance: float = REFERENCE_TOLERANCE,
    expected_mean_tolerance: float = EXPECTED_MEAN_TOLERANCE,
) -> dict[str, float]:
    indexed = {
        (str(row["method"]), int(row["seed"])): float(row["plasticity"])
        for row in rows
        if int(row["horizon"]) == HORIZONS[-1]
    }
    means: dict[str, float] = {}
    for method in ("cap", "upgd"):
        values = []
        for seed in FINAL_SEEDS:
            observed = indexed.get((method, seed))
            expected = reference.get(method, {}).get(seed)
            if observed is None or expected is None:
                raise RuntimeError(f"missing 1M equivalence value for {method}/seed{seed}")
            if abs(observed - expected) > reference_tolerance:
                raise RuntimeError(
                    f"1M metric mismatch for {method}/seed{seed}: "
                    f"horizon={observed:.12f}, final={expected:.12f}"
                )
            values.append(observed)
        means[method] = statistics.mean(values)
        if abs(means[method] - float(expected_means[method])) > expected_mean_tolerance:
            raise RuntimeError(
                f"unexpected 1M {method} mean {means[method]:.12f}; "
                f"expected approximately {expected_means[method]:.6f}"
            )
    return means


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_equivalence_metadata() -> dict[str, Any]:
    engine_path = ROOT / "experiment_e4b.py"
    runtime_path = ROOT / "post_phase1_streams.py"
    engine = engine_path.read_text(encoding="utf-8")
    runtime = runtime_path.read_text(encoding="utf-8")
    checks = {
        "fixed_task_count_200": bool(re.search(r"^TASK_COUNT\s*=\s*200\s*$", engine, re.MULTILINE)),
        "fixed_task_length_5000": bool(
            re.search(r"^STEPS_PER_TASK\s*=\s*5_000\s*$", engine, re.MULTILINE)
        ),
        "constant_learning_rate_no_scheduler": not bool(
            re.search(r"lr_scheduler|learning_rate_scheduler", engine)
        ),
        "task_is_current_step_function": "task = ((step - 1) // STEPS_PER_TASK) + 1" in engine,
        "loader_seed_independent_of_horizon": "seed=seed" in engine,
        "fixed_permutation_seed_by_task": (
            "stream_a.PERMUTATION_SEED_BASE + task" in runtime
        ),
        "fixed_scalar_cadence": "if step % SCALAR_LOG_EVERY == 0" in engine,
        "fixed_task_boundary_evaluation": "if within_task_step == STEPS_PER_TASK" in engine,
        "resume_fast_forwards_same_loader": "fast_forward_batches(batches, start_step)" in engine,
        "terminal_horizon_only_bounds_future_loop": (
            "for step in range(start_step + 1, TOTAL_STEPS + 1)" in engine
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "prefix-equivalence check failed; dedicated shorter runs are required: "
            + ", ".join(failed)
        )
    return {
        "verified": True,
        "reused_1m_prefixes": True,
        "dedicated_shorter_runs_required": False,
        "conclusion": (
            "The training loader, task permutation, optimizer learning rate, logging, "
            "evaluation, and checkpoint behavior of every prefix are independent of the "
            "declared terminal horizon; future steps cannot affect an earlier prefix."
        ),
        "checks": checks,
        "source_files": {
            str(engine_path): _sha256(engine_path),
            str(runtime_path): _sha256(runtime_path),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _plot_primary(summary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create horizon figures") from error
    labels = {"cap": "Final tuned CAP", "upgd": "UPGD", "adamw": "AdamW", "ewc": "EWC", "replay": "Replay"}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["method"])].append(row)
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    for method, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["horizon"]))
        axis.errorbar(
            [int(row["horizon"]) for row in ordered],
            [float(row["mean_plasticity"]) for row in ordered],
            yerr=[float(row["std_plasticity"]) for row in ordered],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=labels.get(method, method),
        )
    axis.set_xticks(HORIZONS, ["100k", "250k", "500k", "750k", "1M"])
    axis.set_xlabel("Stream horizon H")
    axis.set_ylabel("Late-half plasticity $P_H$")
    axis.set_title("Stream-horizon plasticity sensitivity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_delta(delta_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to create horizon figures") from error
    by_horizon = {
        int(row["horizon"]): row
        for row in delta_rows
        if int(row["seed"]) == FINAL_SEEDS[0]
    }
    means = [float(by_horizon[horizon]["mean_paired_delta"]) for horizon in HORIZONS]
    lower = [float(by_horizon[horizon]["ci95_low"]) for horizon in HORIZONS]
    upper = [float(by_horizon[horizon]["ci95_high"]) for horizon in HORIZONS]
    errors = ([mean - low for mean, low in zip(means, lower)], [high - mean for mean, high in zip(means, upper)])
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.errorbar(HORIZONS, means, yerr=errors, marker="o", capsize=3, linewidth=1.8)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xticks(HORIZONS, ["100k", "250k", "500k", "750k", "1M"])
    axis.set_xlabel("Stream horizon H")
    axis.set_ylabel("CAP $P_H$ - UPGD $P_H$")
    axis.set_title("Paired stream-horizon plasticity difference (95% CI)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _fixed_window_means(trajectories: Sequence[LoadedTrajectory]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        if trajectory.spec.key not in ("cap", "upgd"):
            continue
        for label, start, end in FIXED_WINDOWS:
            per_seed = []
            for seed in FINAL_SEEDS:
                selected = [
                    value
                    for step, value in trajectory.values[seed].items()
                    if step > start and step <= end
                ]
                if not selected:
                    raise RuntimeError(
                        f"empty fixed window for {trajectory.spec.key}/seed{seed}/{label}"
                    )
                per_seed.append(statistics.mean(selected))
            rows.append(
                {
                    "method": trajectory.spec.key,
                    "window": label,
                    "mean": statistics.mean(per_seed),
                    "sd": statistics.stdev(per_seed),
                }
            )
    return rows


def _interpretation(delta_rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    means = {
        int(row["horizon"]): float(row["mean_paired_delta"])
        for row in delta_rows
        if int(row["seed"]) == FINAL_SEEDS[0]
    }
    cap_leads = [horizon for horizon in HORIZONS if means[horizon] > 0]
    upgd_leads = [horizon for horizon in HORIZONS if means[horizon] < 0]
    ties = [horizon for horizon in HORIZONS if means[horizon] == 0]
    reversals = [
        f"{left:,}-{right:,} steps"
        for left, right in zip(HORIZONS, HORIZONS[1:])
        if means[left] * means[right] < 0
    ]
    statement = (
        f"CAP has the higher paired mean at {cap_leads or 'no tested horizons'}; "
        f"UPGD has the higher paired mean at {upgd_leads or 'no tested horizons'}"
        + (f"; exact tested ties occur at {ties}" if ties else "")
        + "."
    )
    return statement, reversals


def _report(
    trajectories: Sequence[LoadedTrajectory],
    summary_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
    final_metrics_path: Path,
) -> str:
    statement, reversals = _interpretation(delta_rows)
    lines = [
        "# Stream-horizon plasticity sensitivity",
        "",
        "This is an analysis-only supporting characterization. The five-seed 1M Stream-A/Stream-B benchmark remains the primary benchmark.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{list(FINAL_SEEDS)}`",
        f"- Horizons: `{list(HORIZONS)}`",
        "- Metric: mean `batch_accuracy` for observations satisfying `step > H/2` and `step <= H`.",
        "- Existing deterministic 1M prefixes were reused; dedicated shorter runs were not required.",
        f"- Final-metric equivalence reference: `{final_metrics_path}`",
        "- Input trajectories:",
    ]
    for trajectory in trajectories:
        lines.append(
            f"  - {trajectory.spec.display_name}: `{trajectory.scalar_path}` "
            f"(arm `{trajectory.spec.arm}`, duplicate rows removed: {trajectory.duplicate_rows_removed})"
        )
    lines += [
        "",
        "## Horizon summary",
        "",
        "| Method | Horizon | Mean P_H | Sample SD | Min seed | Max seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display = {trajectory.spec.key: trajectory.spec.display_name for trajectory in trajectories}
    for row in summary_rows:
        lines.append(
            f"| {display[str(row['method'])]} | {int(row['horizon']):,} | "
            f"{float(row['mean_plasticity']):.6f} | {float(row['std_plasticity']):.6f} | "
            f"{float(row['min_seed_value']):.6f} | {float(row['max_seed_value']):.6f} |"
        )
    lines += [
        "",
        "## Paired CAP minus UPGD",
        "",
        "| Horizon | Mean delta | Sample SD | 95% paired-mean CI | Seed deltas |",
        "|---:|---:|---:|---:|---|",
    ]
    for horizon in HORIZONS:
        row = next(
            row
            for row in delta_rows
            if int(row["horizon"]) == horizon and int(row["seed"]) == FINAL_SEEDS[0]
        )
        lines.append(
            f"| {horizon:,} | {float(row['mean_paired_delta']):+.6f} | "
            f"{float(row['std_paired_delta']):.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | "
            f"`{row['seed_deltas_json']}` |"
        )
    lines += ["", "## Fixed-window context", "", "| Method | Window | Mean | Sample SD |", "|---|---|---:|---:|"]
    for row in fixed_rows:
        lines.append(
            f"| {display[str(row['method'])]} | {row['window']} | "
            f"{float(row['mean']):.6f} | {float(row['sd']):.6f} |"
        )
    lines += ["", "## Interpretation", "", statement]
    if reversals:
        lines += [
            "",
            "The paired mean changes sign between the following adjacent tested horizons: "
            + ", ".join(reversals)
            + ". No more precise crossover time is inferred.",
        ]
    else:
        lines += [
            "",
            "The paired mean does not change sign between adjacent tested horizons, so these data do not establish a tested crossover interval.",
        ]
    lines += [
        "",
        "This result characterizes temporal behavior and does not replace or redefine the 1M benchmark.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    prefix_metadata = prefix_equivalence_metadata()
    specs = [
        MethodSpec("cap", "Final tuned CAP", args.cap_dir, args.cap_arm),
        MethodSpec("upgd", "UPGD", args.upgd_dir, args.upgd_arm),
    ]
    contextual = (
        (args.include_adamw, "adamw", "AdamW", "adamw_retention"),
        (args.include_ewc, "ewc", "EWC", "ewc_oracle_retention"),
        (args.include_replay, "replay", "Replay", "adam_replay_retention"),
    )
    for enabled, key, name, arm in contextual:
        if enabled:
            specs.append(MethodSpec(key, name, args.upgd_dir, arm))

    trajectories = [load_trajectory(spec) for spec in specs]
    for trajectory in trajectories:
        validate_trajectory(trajectory)
    if set(trajectories[0].values) != set(trajectories[1].values):
        raise RuntimeError("CAP and UPGD do not have identical seed sets")

    per_seed_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        per_seed_rows.extend(
            horizon_plasticity(trajectory.values, trajectory.spec.key)
        )
    summary_rows = summarize(per_seed_rows)
    delta_rows = paired_deltas(per_seed_rows)
    final_metrics_path = _find_final_metrics(args.final_metrics, trajectories[0].scalar_path)
    reference = _reference_plasticity(final_metrics_path)
    one_million_means = validate_one_million_equivalence(per_seed_rows, reference)
    fixed_rows = _fixed_window_means(trajectories)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output_dir / "horizon_plasticity_per_seed.csv",
        per_seed_rows,
        PER_SEED_FIELDS,
    )
    _write_csv(
        args.output_dir / "horizon_plasticity_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    _write_csv(
        args.output_dir / "cap_vs_upgd_horizon_deltas.csv",
        delta_rows,
        DELTA_FIELDS,
    )
    _plot_primary(summary_rows, args.output_dir / "stream_horizon_plasticity.png")
    _plot_delta(delta_rows, args.output_dir / "stream_horizon_delta.png")

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(FINAL_SEEDS),
        "horizons": list(HORIZONS),
        "metric": "mean(batch_accuracy) for step > H/2 and step <= H",
        "scalar_log_every": SCALAR_LOG_EVERY,
        "task_length": STEPS_PER_TASK,
        "one_million_equivalence": {
            "reference_file": str(final_metrics_path.resolve()),
            "reference_tolerance": REFERENCE_TOLERANCE,
            "expected_mean_tolerance": EXPECTED_MEAN_TOLERANCE,
            "observed_means": one_million_means,
            "expected_approximate_means": EXPECTED_FINAL_MEANS,
            "passed": True,
        },
        "prefix_equivalence": prefix_metadata,
        "inputs": [
            {
                "method": trajectory.spec.key,
                "display_name": trajectory.spec.display_name,
                "directory": str(trajectory.spec.directory),
                "scalar_path": str(trajectory.scalar_path.resolve()),
                "arm": trajectory.spec.arm,
                "sha256": _sha256(trajectory.scalar_path),
                "duplicate_rows_removed": trajectory.duplicate_rows_removed,
            }
            for trajectory in trajectories
        ],
    }
    _write_json(args.output_dir / "horizon_analysis_metadata.json", metadata)
    (args.output_dir / "horizon_analysis.md").write_text(
        _report(
            trajectories,
            summary_rows,
            delta_rows,
            fixed_rows,
            final_metrics_path,
        ),
        encoding="utf-8",
    )
    print(f"Wrote stream-horizon analysis to {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
