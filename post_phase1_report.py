"""Produce the fully self-contained Post-Phase-1 v2 report."""

from __future__ import annotations

import html
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from experiment_e2 import read_csv
from experiment_stream_b import end_retention_by_seed
from post_phase1_common import (
    EXPECTED_GRID_COUNTS,
    LIVENESS_RECORD,
    METHODS,
    POST_DIR,
    SEEDS,
    V2_ISOLATION,
    load_json,
    load_pool_decision,
    load_winner_artifact,
    post_guard,
)
from post_phase1_streams import POST_STREAM_A, POST_STREAM_B, arm_name


REPORT = POST_DIR / "post_report.md"
FRONTIER = POST_DIR / "post_validation_frontiers.svg"
FRONTIER_CAVEAT = (
    "Validation-scale quantities use a pool half the selected Stream-B size "
    "for two shuffled blocks. They are not comparable to final-stream values. "
    "The frontier is descriptive only and does not select configurations."
)


def _label(method: str) -> str:
    if method == "adam_replay":
        return "adam_replay (out-of-class)"
    if method == "ewc_oracle":
        return "ewc_oracle (boundary-privileged)"
    return method


def _sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summary(values: Mapping[int, float]) -> str:
    ordered = [float(values[seed]) for seed in SEEDS]
    return f"{statistics.mean(ordered):.6f} ± {_sd(ordered):.6f}"


def paired_difference(left: Mapping[int, float], right: Mapping[int, float]) -> str:
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise RuntimeError("paired comparison lacks the full seed set")
    return summary({seed: float(left[seed]) - float(right[seed]) for seed in SEEDS})


def stream_a_plasticity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        step = int(float(row["step"]))
        if 500_001 <= step <= 1_000_000:
            grouped[(str(row["arm"]), int(float(row["seed"])))].append(
                float(row["batch_accuracy"])
            )
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (arm, seed), values in grouped.items():
        if not values:
            raise RuntimeError(f"missing Stream-A plasticity window: {arm}/{seed}")
        result[arm][seed] = statistics.mean(values)
    return dict(result)


def _valid_cells(items: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    return [
        (index, item)
        for index, item in enumerate(items)
        if item.get("validation_scale") == "pp1_a1_v2"
        and item.get("provenance") in {"registered_grid", "edge_extension"}
        and item.get("validation_retention_metric") is not None
        and item.get("validation_plasticity") is not None
        and not bool(item.get("diverged"))
    ]


def pareto(items: Sequence[Mapping[str, Any]]) -> set[int]:
    valid = _valid_cells(items)
    return {
        index
        for index, item in valid
        if not any(
            float(other["validation_plasticity"]) >= float(item["validation_plasticity"])
            and float(other["validation_retention_metric"])
            >= float(item["validation_retention_metric"])
            and (
                float(other["validation_plasticity"])
                > float(item["validation_plasticity"])
                or float(other["validation_retention_metric"])
                > float(item["validation_retention_metric"])
            )
            for other_index, other in valid
            if other_index != index
        )
    }


def build_frontier_svg(artifact: Mapping[str, Any]) -> str:
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.t{font-size:16px;font-weight:700}.n{font-size:11px}</style>',
    ]
    for panel, method in enumerate(METHODS):
        x0, y0 = 30 + (panel % 3) * 390, 35 + (panel // 3) * 330
        tried = artifact[method]["tried_configs"]
        valid = _valid_cells(tried)
        if not valid:
            raise RuntimeError(f"no v2 frontier cells for {method}")
        xs = [float(item["validation_plasticity"]) for _, item in valid]
        ys = [float(item["validation_retention_metric"]) for _, item in valid]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
        chunks += [
            f'<text x="{x0}" y="{y0}" class="t">{html.escape(_label(method))}</text>',
            f'<rect x="{x0}" y="{y0+15}" width="340" height="260" fill="#f8fafc" stroke="#cbd5e1"/>',
        ]
        frontier = pareto(tried)
        selected = artifact[method]["retention_config"]
        for original_index, item in valid:
            px = x0 + 15 + (float(item["validation_plasticity"]) - xmin) / dx * 310
            py = y0 + 260 - (float(item["validation_retention_metric"]) - ymin) / dy * 225
            chosen = item["config"] == selected
            color = "#dc2626" if chosen else (
                "#0f766e" if original_index in frontier else "#94a3b8"
            )
            chunks.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{5 if chosen else 3}" fill="{color}"/>'
            )
        chunks.append(
            f'<text x="{x0+170}" y="{y0+295}" text-anchor="middle" class="n">plasticity →; retention ↑</text>'
        )
    chunks += [
        f'<text x="30" y="700" class="n">{html.escape(FRONTIER_CAVEAT)}</text>',
        "</svg>",
    ]
    return "\n".join(chunks)


def render_report() -> str:
    artifact = load_winner_artifact()
    decision = load_pool_decision()
    pool_size = int(decision["chosen_pool_size"])
    liveness = load_json(LIVENESS_RECORD)
    if liveness.get("isolation") != V2_ISOLATION:
        raise RuntimeError("knob-liveness record is not from this v2 experiment")
    for stream_root in (POST_STREAM_A, POST_STREAM_B):
        config = load_json(stream_root / "post_run_config.json")
        if config.get("isolation") != V2_ISOLATION:
            raise RuntimeError(f"non-v2 stream output found: {stream_root}")

    plasticity = stream_a_plasticity(read_csv(POST_STREAM_A / "stream_scalars.csv"))
    retention = end_retention_by_seed(
        read_csv(POST_STREAM_B / "streamb_pool.csv"), pool_size=pool_size
    )
    expected = {arm_name(method) for method in METHODS}
    if set(plasticity) != expected or set(retention) != expected:
        raise RuntimeError("v2 stream coverage is incomplete")
    for arm in expected:
        if set(plasticity[arm]) != set(SEEDS) or set(retention[arm]) != set(SEEDS):
            raise RuntimeError(f"incomplete five-seed metrics for {arm}")

    adam = statistics.mean(retention[arm_name("adamw")].values())
    replay = statistics.mean(retention[arm_name("adam_replay")].values())
    ewc = statistics.mean(retention[arm_name("ewc_oracle")].values())
    replay_absent = replay <= adam + 0.01
    ewc_suspect = ewc <= adam + 0.01
    replay_verdict = "REPLAY-CONTROL-ABSENT" if replay_absent else "PASS"
    ewc_verdict = "EWC-INTEGRATION-SUSPECT" if ewc_suspect else "PASS"
    cap_r = retention[arm_name("cap_release")]
    cap_p = plasticity[arm_name("cap_release")]
    degenerate = [
        method for method in METHODS
        if statistics.mean(plasticity[arm_name(method)].values()) < 0.50
    ]

    lines = [
        "# Self-contained Post-Phase-1 Retention Analysis",
        "",
        "This experiment uses shared implementation code but reads no older "
        "Phase-1 or Post-Phase-1 result, manifest, verdict, configuration, metric, "
        "threshold, or cached stream row. Every reported number is produced under "
        "the current Post-Phase-1 protocol.",
        "",
        "## Pool-headroom decision",
        "",
        "| Pool size | Mean end retention | SD | Qualifies (≤0.85) |",
        "|---:|---:|---:|---|",
    ]
    for row in decision["measurements"]:
        lines.append(
            f"| {row['pool_size']} | {float(row['mean_end_retention']):.6f} | "
            f"{float(row['sd_end_retention']):.6f} | {row['qualifies']} |"
        )
    lines += [
        "",
        f"Selected Stream-B pool size: **{pool_size}**.",
        "",
        "## Grid definitions and fixed knobs",
        "",
        "All methods sweep lr={1e-4,3e-4,1e-3} × weight_decay={0,1e-2}.",
        "",
        "| Method | Cells | Additional dimensions / fixed knobs |",
        "|---|---:|---|",
        "| adamw | 6 | none |",
        "| adam_replay (out-of-class) | 12 | replay_count={32,96}; buffer_capacity=5000 fixed |",
        "| upgd | 12 | sigma={1e-4,1e-2}; beta_utility=0.9999 fixed |",
        "| ewc_oracle (boundary-privileged) | 12 | lambda={1e3,1e5}; gamma=0.95 fixed |",
        "| cap_release | 48 | k={1e-5,7e-5}; T_a={400,2000}; T_m={10000,200000}; T_s=2000 fixed |",
        "| cap_ratchet | 24 | k={1e-5,7e-5}; T_a={400,2000}; T_s=2000 fixed |",
        "",
        f"Total registered cells: **{sum(EXPECTED_GRID_COUNTS.values())}**.",
        "",
        "## Knob-liveness preflight",
        "",
        "| Method | Knob | Values | Diagnostic state | Steps | L2 distance | Result |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in liveness["results"]:
        diagnostic_state = (
            f"primed ({row.get('state_priming_amendment', 'POST-P1-A2')})"
            if row.get("state_primed")
            else "cold start"
        )
        lines.append(
            f"| {row['method']} | {row['knob']} | {row['low']} → {row['high']} | "
            f"{diagnostic_state} | {row['steps']} | "
            f"{float(row['l2_distance']):.6g} | {row['status']} |"
        )

    lines += [
        "",
        "## Edge extension and winners",
        "",
        "Selection maximized min(plasticity, retention), then their mean, then "
        "earlier grid order. Pools contain only current v2 grid and edge cells.",
        "",
        "| Method | Config | Retention | Plasticity | Min score | Extended | Still at edge |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for method in METHODS:
        entry = artifact[method]
        flag = " **RETENTION-DEGENERATE**" if method in degenerate else ""
        lines.append(
            f"| {_label(method)}{flag} | `{entry['retention_config']}` | "
            f"{float(entry['retention_metric']):.6f} | "
            f"{float(entry['validation_plasticity']):.6f} | "
            f"{float(entry['selection_min_score']):.6f} | "
            f"{entry['edge_extension_applied']} | {entry['still_at_edge']} "
            f"{entry.get('still_at_edge_details', [])} |"
        )

    lines += [
        "",
        "## Validity controls",
        "",
        f"- Replay: {replay:.6f} versus AdamW+0.01={adam+0.01:.6f}; **{replay_verdict}**.",
        f"- EWC: {ewc:.6f} versus AdamW+0.01={adam+0.01:.6f}; **{ewc_verdict}**.",
        "",
        "## Headline comparison",
        "",
        "| Method | Retention mean ± SD | Plasticity mean ± SD | Retention Δ vs CAP release | Plasticity Δ vs CAP release |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        arm = arm_name(method)
        suppress = method == "ewc_oracle" and ewc_suspect
        blocked = "SUPPRESSED — EWC-INTEGRATION-SUSPECT"
        flag = " **RETENTION-DEGENERATE**" if method in degenerate else ""
        lines.append(
            f"| {_label(method)}{flag} | {blocked if suppress else summary(retention[arm])} | "
            f"{blocked if suppress else summary(plasticity[arm])} | "
            f"{blocked if suppress else paired_difference(retention[arm], cap_r)} | "
            f"{blocked if suppress else paired_difference(plasticity[arm], cap_p)} |"
        )
    if replay_absent:
        lines += [
            "",
            "**REPLAY-CONTROL-ABSENT fired. The table is printed but retention "
            "comparisons must not be interpreted until the control is debugged.**",
        ]

    lines += [
        "",
        "## Validation-scale trade-off frontiers",
        "",
        "![Validation-scale frontiers](post_validation_frontiers.svg)",
        "",
        f"**Mandatory caveat:** {FRONTIER_CAVEAT}",
        "",
        "## Per-seed values and paired differences",
        "",
    ]
    for method in METHODS:
        arm = arm_name(method)
        flag = " **RETENTION-DEGENERATE**" if method in degenerate else ""
        lines += [
            f"### {_label(method)}{flag}",
            "",
            "| Seed | Retention | Plasticity | Retention Δ vs CAP release | Plasticity Δ vs CAP release |",
            "|---:|---:|---:|---:|---:|",
        ]
        for seed in SEEDS:
            suppress = method == "ewc_oracle" and ewc_suspect
            lines.append(
                f"| {seed} | {retention[arm][seed]:.6f} | {plasticity[arm][seed]:.6f} | "
                f"{'SUPPRESSED' if suppress else f'{retention[arm][seed]-cap_r[seed]:+.6f}'} | "
                f"{'SUPPRESSED' if suppress else f'{plasticity[arm][seed]-cap_p[seed]:+.6f}'} |"
            )
        lines.append("")

    ewc_diverged = any(
        item.get("status") == "EWC-DIVERGED"
        for item in artifact["ewc_oracle"]["tried_configs"]
    )
    edges = {
        method: artifact[method]["still_at_edge_details"]
        for method in METHODS if artifact[method]["still_at_edge"]
    }
    lines += [
        "## Named verdict ledger",
        "",
        f"- REPLAY-CONTROL-ABSENT: **{replay_verdict}**.",
        f"- EWC-INTEGRATION-SUSPECT: **{ewc_verdict}**.",
        f"- EWC-DIVERGED: **{'FIRED' if ewc_diverged else 'NOT-FIRED'}**.",
        f"- EWC-RETENTION-DIVERGED: **{'FIRED' if (POST_DIR / 'post_ewc_divergence.json').exists() else 'NOT-FIRED'}**.",
        f"- RETENTION-DEGENERATE: **{degenerate if degenerate else 'NOT-FIRED'}**.",
        f"- still_at_edge: **{edges if edges else 'NOT-FIRED'}**.",
        "",
        "This analysis is descriptive and creates no external pass/fail threshold.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    with post_guard():
        artifact = load_winner_artifact()
        FRONTIER.write_text(build_frontier_svg(artifact), encoding="utf-8")
        REPORT.write_text(render_report(), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
