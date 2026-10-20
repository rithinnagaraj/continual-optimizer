"""Analyze completed frozen-ResNet-50-head mini-ImageNet scale runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from miniimagenet_scale_common import (
    DOMAINS,
    EXPERIMENT_VERSION,
    FINAL_SEEDS,
    METHODS,
    atomic_write_csv,
    atomic_write_json,
    read_csv,
    scale_metrics,
)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _load_runs(results_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for method in METHODS:
        for seed in FINAL_SEEDS:
            run_dir = results_dir / "train" / method / f"seed_{seed}"
            config_path = run_dir / "run_config.json"
            segment_path = run_dir / "stream_segments.csv"
            if not config_path.exists() or not segment_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("experiment_version") != EXPERIMENT_VERSION:
                raise ValueError(f"wrong experiment version in {config_path}")
            segments = read_csv(segment_path)
            if len(segments) != 45:
                raise ValueError(f"final run is incomplete ({len(segments)}/45): {run_dir}")
            runs[method][seed] = {
                "dir": run_dir,
                "config": config,
                "segments": segments,
                "online": read_csv(run_dir / "online_steps.csv"),
                "fast": read_csv(run_dir / "domain_eval_fast.csv"),
                "full": read_csv(run_dir / "domain_eval_full.csv"),
                "diagnostics": read_csv(run_dir / "optimizer_diagnostics.csv"),
            }
    return runs


def _final_domain_accuracy(run: Mapping[str, Any]) -> float:
    rows = run["full"]
    final_segment = max(int(row["checkpoint_segment"]) for row in rows)
    values = [
        float(row["accuracy"])
        for row in rows
        if int(row["checkpoint_segment"]) == final_segment
    ]
    if len(values) != len(DOMAINS):
        raise ValueError("final full evaluation does not cover all nine domains")
    return fmean(values)


def _efficiency(run: Mapping[str, Any]) -> tuple[float, float, float, float]:
    online = run["online"]
    final_update = max(int(row["global_update"]) for row in online)
    interior = [
        row for row in online if 500 < int(row["global_update"]) <= final_update - 500
    ] or online
    update_seconds = fmean(float(row["update_seconds"]) for row in interior)
    peak_memory = max(float(row["peak_gpu_memory_bytes"]) for row in online)
    diagnostics = run["diagnostics"]
    optimizer_memory = max(float(row["optimizer_state_bytes"]) for row in diagnostics)
    checkpoint_size = float((Path(run["dir"]) / "checkpoint_final.pt").stat().st_size)
    return update_seconds, peak_memory, optimizer_memory, checkpoint_size


def build_summary(runs: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_runs = runs.get(method, {})
        if not method_runs:
            continue
        seed_metrics = [scale_metrics(run["segments"]) for run in method_runs.values()]
        p_mean, p_sd = _mean_sd([row["P_scale"] for row in seed_metrics])
        r_mean, r_sd = _mean_sd([row["R_scale"] for row in seed_metrics])
        b_mean, b_sd = _mean_sd([row["B_scale"] for row in seed_metrics])
        full_retention_mean, full_retention_sd = _mean_sd(
            [row["R_full_stream"] for row in seed_metrics]
        )
        final_mean, final_sd = _mean_sd(
            [_final_domain_accuracy(run) for run in method_runs.values()]
        )
        efficiencies = [_efficiency(run) for run in method_runs.values()]
        update_mean, update_sd = _mean_sd([value[0] for value in efficiencies])
        rows.append(
            {
                "method": method,
                "seeds": len(method_runs),
                "P_scale_mean": p_mean,
                "P_scale_sd": p_sd,
                "R_scale_mean": r_mean,
                "R_scale_sd": r_sd,
                "B_scale_mean": b_mean,
                "B_scale_sd": b_sd,
                "R_full_stream_mean": full_retention_mean,
                "R_full_stream_sd": full_retention_sd,
                "final_all_domain_accuracy_mean": final_mean,
                "final_all_domain_accuracy_sd": final_sd,
                "update_seconds_mean": update_mean,
                "update_seconds_sd": update_sd,
                "peak_gpu_memory_bytes_mean": fmean(value[1] for value in efficiencies),
                "optimizer_state_bytes_mean": fmean(value[2] for value in efficiencies),
                "checkpoint_size_bytes_mean": fmean(value[3] for value in efficiencies),
            }
        )
    return rows


def _gap_bins(gaps: Sequence[int]) -> list[tuple[int, int | None]]:
    bins: list[tuple[int, int | None]] = [(1, 3), (4, 6), (7, 9), (10, 12), (13, None)]
    supported = [
        item
        for item in bins
        if any(gap >= item[0] and (item[1] is None or gap <= item[1]) for gap in gaps)
    ]
    # Merge bins with fewer than three schedule events using only support,
    # never method performance.  Counts are calculated from one copy of each
    # benchmark event supplied by the caller.
    while len(supported) > 1:
        counts = [
            sum(gap >= low and (high is None or gap <= high) for gap in gaps)
            for low, high in supported
        ]
        if min(counts) >= 3:
            break
        index = counts.index(min(counts))
        if index == 0:
            low, _ = supported[0]
            _, high = supported[1]
            supported[:2] = [(low, high)]
        else:
            low, _ = supported[index - 1]
            _, high = supported[index]
            supported[index - 1 : index + 1] = [(low, high)]
    return supported


def _bin_label(item: tuple[int, int | None]) -> str:
    return f"{item[0]}+" if item[1] is None else f"{item[0]}-{item[1]}"


def build_gap_rows(runs: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    support_rows: list[Mapping[str, Any]] = []
    for method in METHODS:
        if runs.get(method):
            support_rows = next(iter(runs[method].values()))["segments"]
            break
    gaps = [int(row["revisit_gap"]) for row in support_rows if row["revisit_gap"] != ""]
    bins = _gap_bins(gaps)
    result: list[dict[str, Any]] = []
    for method, method_runs in runs.items():
        for low, high in bins:
            values = [
                float(row["pre_visit_accuracy"])
                for run in method_runs.values()
                for row in run["segments"]
                if row["revisit_gap"] != ""
                and int(row["revisit_gap"]) >= low
                and (high is None or int(row["revisit_gap"]) <= high)
            ]
            if values:
                mean, sd = _mean_sd(values)
                result.append(
                    {
                        "method": method,
                        "gap_bin": _bin_label((low, high)),
                        "gap_low": low,
                        "gap_high": "" if high is None else high,
                        "events": len(values),
                        "retention_mean": mean,
                        "retention_sd": sd,
                    }
                )
    return result


def build_cycle_rows(runs: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method, method_runs in runs.items():
        for cycle in range(1, 6):
            per_seed = [
                fmean(
                    float(row["mean_online_accuracy"])
                    for row in run["segments"]
                    if int(row["cycle"]) == cycle
                )
                for run in method_runs.values()
            ]
            mean, sd = _mean_sd(per_seed)
            result.append(
                {
                    "method": method,
                    "cycle": cycle,
                    "plasticity_mean": mean,
                    "plasticity_sd": sd,
                }
            )
    return result


def _plots(
    analysis_dir: Path,
    summary: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required for scale figures") from error

    colors = {"adamw": "#777777", "cap": "#0072B2", "upgd": "#D55E00", "ewc": "#009E73"}
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for row in summary:
        axis.errorbar(
            float(row["R_scale_mean"]),
            float(row["P_scale_mean"]),
            xerr=float(row["R_scale_sd"]),
            yerr=float(row["P_scale_sd"]),
            marker="o",
            capsize=3,
            label=str(row["method"]),
            color=colors[str(row["method"])],
        )
    axis.set(xlabel="R_scale (pre-visit retention)", ylabel="P_scale (online plasticity)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(analysis_dir / "figure1_stability_plasticity.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in METHODS:
        rows = [row for row in cycle_rows if row["method"] == method]
        if rows:
            axis.errorbar(
                [int(row["cycle"]) for row in rows],
                [float(row["plasticity_mean"]) for row in rows],
                yerr=[float(row["plasticity_sd"]) for row in rows],
                marker="o",
                capsize=3,
                label=method,
                color=colors[method],
            )
    axis.set(xlabel="Cycle", ylabel="Mean pre-update batch accuracy", xticks=range(1, 6))
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(analysis_dir / "figure2_plasticity_by_cycle.png", dpi=180)
    plt.close(figure)

    labels = list(dict.fromkeys(str(row["gap_bin"]) for row in gap_rows))
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in METHODS:
        rows = [row for row in gap_rows if row["method"] == method]
        if rows:
            mapping = {str(row["gap_bin"]): row for row in rows}
            axis.plot(
                labels,
                [float(mapping[label]["retention_mean"]) for label in labels],
                marker="o",
                label=method,
                color=colors[method],
            )
    axis.set(xlabel="Intervening-segment revisit gap", ylabel="Pre-visit accuracy")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(analysis_dir / "figure3_retention_by_gap.png", dpi=180)
    plt.close(figure)

    for method, method_runs in runs.items():
        checkpoints = sorted(
            {
                int(row["checkpoint_segment"])
                for run in method_runs.values()
                for row in run["full"]
            }
        )
        matrix = np.full((len(checkpoints), len(DOMAINS)), np.nan)
        for i, checkpoint in enumerate(checkpoints):
            for domain_id in range(len(DOMAINS)):
                values = [
                    float(row["accuracy"])
                    for run in method_runs.values()
                    for row in run["full"]
                    if int(row["checkpoint_segment"]) == checkpoint
                    and int(row["evaluated_domain_id"]) == domain_id
                ]
                if values:
                    matrix[i, domain_id] = fmean(values)
        figure, axis = plt.subplots(figsize=(10.2, 4.3))
        image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
        axis.set_yticks(range(len(checkpoints)), checkpoints)
        axis.set(xlabel="Evaluation domain", ylabel="Checkpoint segment", title=method)
        figure.colorbar(image, ax=axis, label="Accuracy")
        figure.tight_layout()
        figure.savefig(analysis_dir / f"figure4_domain_matrix_{method}.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.bar(
        [str(row["method"]) for row in summary],
        [float(row["update_seconds_mean"]) * 1000 for row in summary],
        color=[colors[str(row["method"])] for row in summary],
    )
    axis.set(ylabel="Mean interior update time (ms)")
    figure.tight_layout()
    figure.savefig(analysis_dir / "figure5_efficiency.png", dpi=180)
    plt.close(figure)


def _report(summary: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# mini-ImageNet frozen ResNet-50 head recurrent-domain scale analysis",
        "",
        f"Experiment version: `{EXPERIMENT_VERSION}`.",
        "",
        "EWC is labeled oracle because it receives segment boundaries.",
        "",
        "| Method | P_scale mean ± SD | R_scale mean ± SD | B_scale mean ± SD | Final all-domain accuracy | Update ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {p:.4f} ± {ps:.4f} | {r:.4f} ± {rs:.4f} | "
            "{b:.4f} ± {bs:.4f} | {final:.4f} ± {finals:.4f} | {update:.3f} |".format(
                method=row["method"],
                p=float(row["P_scale_mean"]),
                ps=float(row["P_scale_sd"]),
                r=float(row["R_scale_mean"]),
                rs=float(row["R_scale_sd"]),
                b=float(row["B_scale_mean"]),
                bs=float(row["B_scale_sd"]),
                final=float(row["final_all_domain_accuracy_mean"]),
                finals=float(row["final_all_domain_accuracy_sd"]),
                update=float(row["update_seconds_mean"]) * 1000,
            )
        )
    lines.extend(
        [
            "",
            "The primary interpretation is CAP versus UPGD on the stability–plasticity plane. "
            "Use the cycle and gap figures to determine whether any aggregate difference is driven "
            "by stream age or long revisit gaps; the code does not encode a favorable verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(results_dir: Path) -> Path:
    runs = _load_runs(results_dir)
    missing = [method for method in METHODS if not runs.get(method)]
    if missing:
        raise FileNotFoundError(f"no complete final runs found for: {', '.join(missing)}")
    summary = build_summary(runs)
    cycle_rows = build_cycle_rows(runs)
    gap_rows = build_gap_rows(runs)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(analysis_dir / "main_table.csv", summary, tuple(summary[0]))
    atomic_write_csv(
        analysis_dir / "plasticity_by_cycle.csv", cycle_rows, tuple(cycle_rows[0])
    )
    atomic_write_csv(
        analysis_dir / "retention_by_gap.csv", gap_rows, tuple(gap_rows[0])
    )
    atomic_write_json(
        analysis_dir / "analysis_summary.json",
        {
            "experiment_version": EXPERIMENT_VERSION,
            "methods": list(METHODS),
            "summary": summary,
        },
    )
    _plots(analysis_dir, summary, cycle_rows, gap_rows, runs)
    (analysis_dir / "analysis_report.md").write_text(_report(summary), encoding="utf-8")
    return analysis_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/miniimagenet_frozen_resnet50_head"),
    )
    parser.add_argument("--analyze", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = analyze(args.results_dir)
    print(f"Wrote scale analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
