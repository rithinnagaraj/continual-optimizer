"""Neutral paired-seed analysis of the recurrent DN4IL frozen-head experiment."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

from dn4il_dataset import DEFAULT_OUTPUT_DIR, DOMAINS, EXPERIMENT_VERSION
from frozen_head_metrics import dn4il_metrics
from miniimagenet_scale_common import METHODS, FINAL_SEEDS, atomic_write_csv, read_csv
from analyze_miniimagenet_resnet import _efficiency

LABELS = {"adamw": "AdamW", "cap": "CAP", "upgd": "UPGD", "ewc": "EWC-oracle"}


def mean_sd(values):
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def write_rows(directory, name, rows):
    if rows:
        atomic_write_csv(directory / name, rows, tuple(rows[0]))


def load_runs(results_dir, *, method_roots=None, expected_versions=None):
    runs = defaultdict(dict)
    paired = {}
    for method in METHODS:
        for seed in FINAL_SEEDS:
            directory = Path((method_roots or {}).get(method, results_dir)) / "train" / method / f"seed_{seed}"
            if not (directory / "checkpoint_final.pt").is_file():
                raise FileNotFoundError(f"missing completed paired reporting run: {directory}")
            config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            if config.get("experiment_version") != (expected_versions or {}).get(method, EXPERIMENT_VERSION):
                raise ValueError(f"wrong experiment version: {directory}")
            segments = read_csv(directory / "stream_segments.csv")
            evaluation = read_csv(directory / "domain_eval.csv")
            online = read_csv(directory / "online_steps.csv")
            if [int(row["segment"]) for row in segments] != list(range(1, 31)) or len(online) != 6000:
                raise ValueError(f"incomplete final 30-segment / 6000-update stream: {directory}")
            if len(evaluation) != 180 or {(int(row["checkpoint_segment"]), row["evaluated_domain"]) for row in evaluation} != {(s, d) for s in range(1, 31) for d in DOMAINS}:
                raise ValueError(f"incomplete full-test matrix: {directory}")
            fingerprint = (config["manifest_hash"], config["schedule_hash"], config["full_evaluation_bank_hash"],
                           config["head_initialization_hash"], [row["sample_order_hash"] for row in segments])
            if seed in paired and paired[seed] != fingerprint:
                raise ValueError(f"unpaired methods for seed {seed}")
            paired[seed] = fingerprint
            runs[method][seed] = {"dir": directory, "config": config, "segments": segments,
                                 "online": online, "fast": evaluation, "full": evaluation,
                                 "diagnostics": read_csv(directory / "optimizer_diagnostics.csv")}
    return runs


def analyze_runs(runs, output, *, labels=None, provenance=None, disclosure=None):
    labels = labels or LABELS
    provenance = provenance or {}
    output.mkdir(parents=True, exist_ok=True)
    per_seed, forgetting, matrix, efficiency, cycles_p, cycles_r, gaps = [], [], [], [], [], [], []
    for method, method_runs in runs.items():
        for seed, run in method_runs.items():
            base = {"method": labels[method], "seed": seed, **provenance.get(method, {})}
            metrics, domain_metrics = dn4il_metrics(run["segments"], run["fast"])
            per_seed.append({**base, **metrics})
            forgetting.extend({**base, **row} for row in domain_metrics)
            update, peak, state, checkpoint = _efficiency(run)
            efficiency.append({**base, "update_ms": update * 1000, "peak_gpu_memory_bytes": peak,
                               "optimizer_state_bytes": state, "checkpoint_size_bytes": checkpoint})
            for segment in range(1, 31):
                values = {r["evaluated_domain"]: float(r["accuracy"]) for r in run["fast"] if int(r["checkpoint_segment"]) == segment}
                matrix.append({**base, "segment": segment, **{domain: values[domain] for domain in DOMAINS}})
            for cycle in range(1, 6):
                rows = [r for r in run["segments"] if int(r["cycle"]) == cycle]
                cycles_p.append({**base, "cycle": cycle, "plasticity": fmean(float(r["mean_online_accuracy"]) for r in rows)})
                revisits = [r for r in rows if r["revisit_gap"] not in (None, "")]
                if revisits:
                    cycles_r.append({**base, "cycle": cycle, "retention": fmean(float(r["pre_visit_accuracy"]) for r in revisits)})
            for row in run["segments"]:
                if row["revisit_gap"] not in (None, ""):
                    gaps.append({**base, "domain": row["domain_name"], "previous_visit_segment": row["previous_visit_segment"],
                                 "current_segment": row["segment"], "revisit_gap": int(row["revisit_gap"]),
                                 "pre_visit_accuracy": float(row["pre_visit_accuracy"])})
    summary = []
    gap_summary = []
    for method in runs:
        for gap in sorted({row["revisit_gap"] for row in gaps}):
            events = [row for row in gaps if row["method"] == labels[method] and row["revisit_gap"] == gap]
            if events:
                by_seed = defaultdict(list)
                for row in events:
                    by_seed[row["seed"]].append(row["pre_visit_accuracy"])
                mean, sd = mean_sd([fmean(values) for values in by_seed.values()])
                gap_summary.append({"method": labels[method], "revisit_gap": gap, "events": len(events),
                                    "seeds": len(by_seed), "retention_mean": mean, "retention_sd": sd})
    for method in runs:
        label = labels[method]
        row = {"method": label, "seeds": len(runs[method]), **provenance.get(method, {})}
        for key in ("P_scale", "R_scale", "B_scale", "final_all_domain_accuracy", "mean_forgetting", "backward_transfer"):
            row[f"{key}_mean"], row[f"{key}_sd"] = mean_sd([r[key] for r in per_seed if r["method"] == label])
        row["update_ms_mean"], row["update_ms_sd"] = mean_sd([r["update_ms"] for r in efficiency if r["method"] == label])
        summary.append(row)
    for name, rows in (("final_metrics.csv", summary), ("per_seed_metrics.csv", per_seed),
                       ("cycle_plasticity.csv", cycles_p), ("cycle_retention.csv", cycles_r),
                       ("retention_by_gap.csv", gaps), ("forgetting_by_domain.csv", forgetting),
                       ("domain_accuracy_matrix.csv", matrix), ("efficiency.csv", efficiency)):
        write_rows(output, name, rows)
    write_rows(output, "retention_by_gap_summary.csv", gap_summary)
    lines = ["# Recurrent DN4IL frozen ResNet-50 head", "",
             "This uses official DN4IL splits with a five-cycle recurrent Domain-IL schedule; it is not the original paper protocol.", "",
             "P_scale: mean pre-update batch accuracy in segments 16–30. R_scale: full-test pre-visit accuracy for revisits in segments 16–30. B_scale is min(P,R) per seed. Tables report mean ± sample SD over seeds.", "",
             "Forgetting: best previous post-first-exposure checkpoint accuracy minus final accuracy, averaged over six domains (signed). Recurrent BWT: mean over domains of final accuracy minus accuracy just after that domain's first training segment. Revisits can contribute to this BWT. Gap is the number of intervening segments; integer gaps were fixed before performance was observed.", "",
             "EWC-oracle receives every segment boundary. Update time covers cached-head forward/backward and optimizer update, excluding the first and last 500 updates when available; extraction/cache time is recorded separately in manifest/feature_cache_verification.json.", "",
             "| Method | P_scale | R_scale | B_scale | Final all-domain accuracy | Mean forgetting | Update ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if disclosure:
        lines[2:2] = [disclosure, ""]
    for row in summary:
        values = [f"{row[key + '_mean']:.4f} ± {row[key + '_sd']:.4f}" for key in ("P_scale", "R_scale", "B_scale", "final_all_domain_accuracy", "mean_forgetting")]
        lines.append(f"| {row['method']} | " + " | ".join(values) + f" | {row['update_ms_mean']:.3f} |")
    if "cap" in runs:
        cap = {r["seed"]: r for r in per_seed if r["method"] == "CAP"}
        lines.extend(["", "Paired CAP minus comparator differences (positive P/R/B means higher accuracy):", ""])
        for other in ("adamw", "upgd", "ewc"):
            if other in runs:
                other_rows = {r["seed"]: r for r in per_seed if r["method"] == labels[other]}
                seeds = sorted(cap.keys() & other_rows.keys())
                differences = [f"{key} {fmean(cap[s][key] - other_rows[s][key] for s in seeds):+.4f}" for key in ("P_scale", "R_scale", "B_scale")]
                lines.append(f"- CAP versus {labels[other]}: " + ", ".join(differences))
        p_values = [fmean(r["plasticity"] for r in cycles_p if r["method"] == "CAP" and r["cycle"] == c) for c in range(1, 6)]
        lines.extend(["", "CAP cycle plasticity: " + ", ".join(f"{v:.4f}" for v in p_values) + ".",
                      "Monotonic decline across all five cycles: " + str(all(b < a for a, b in zip(p_values, p_values[1:]))) + ".",
                      "Integer-gap retention is plotted; event and seed counts are recorded in retention_by_gap_summary.csv. Sparse gaps should be interpreted cautiously."])
        supported = [row["revisit_gap"] for row in gap_summary if row["method"] == "CAP" and row["events"] >= 3]
        if supported:
            long_gap = max(supported)
            lines.extend(["", f"At the largest integer gap with at least three schedule events ({long_gap} intervening segments), mean per-seed retention is:", ""])
            for row in gap_summary:
                if row["revisit_gap"] == long_gap:
                    lines.append(f"- {row['method']}: {row['retention_mean']:.4f} ± {row['retention_sd']:.4f} ({row['events']} events, {row['seeds']} seeds).")
    (output / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plots(output, summary, cycles_p, cycles_r, gaps, matrix, forgetting)
    return output


def plots(output, summary, cycles_p, cycles_r, gaps, matrix, forgetting):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for row in summary:
        ax.errorbar(row["R_scale_mean"], row["P_scale_mean"], xerr=row["R_scale_sd"], yerr=row["P_scale_sd"], marker="o", capsize=3, label=row["method"])
    ax.set(xlabel="Full-test pre-visit retention (R_scale)", ylabel="Online plasticity (P_scale)")
    ax.legend()
    save(fig, "stability_plasticity_plane.png")
    for rows, xkey, ykey, name in ((cycles_p, "cycle", "plasticity", "plasticity_by_cycle.png"),
                                   (cycles_r, "cycle", "retention", "retention_by_cycle.png"),
                                   (gaps, "revisit_gap", "pre_visit_accuracy", "retention_by_gap.png")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in dict.fromkeys(r["method"] for r in rows):
            subset = [r for r in rows if r["method"] == method]
            xs = sorted({r[xkey] for r in subset})
            # Average events within seed first, then give seeds equal weight.
            means, sds = [], []
            for x in xs:
                by_seed = defaultdict(list)
                for r in subset:
                    if r[xkey] == x:
                        by_seed[r["seed"]].append(r[ykey])
                mean, sd = mean_sd([fmean(v) for v in by_seed.values()])
                means.append(mean); sds.append(sd)
            ax.errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=method)
        ax.set(xlabel=xkey.replace("_", " "), ylabel=ykey.replace("_", " "))
        ax.legend()
        save(fig, name)
    methods = list(dict.fromkeys(row["method"] for row in matrix))
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 8), squeeze=False)
    for ax, method in zip(axes[0], methods):
        array = [[fmean(r[d] for r in matrix if r["method"] == method and r["segment"] == s) for d in DOMAINS] for s in range(1, 31)]
        im = ax.imshow(np.asarray(array), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(6), DOMAINS, rotation=50, ha="right")
        ax.set_yticks(range(0, 30, 3), range(1, 31, 3))
        ax.set(title=method, ylabel="Post-segment checkpoint")
        fig.colorbar(im, ax=ax, label="Full-test accuracy")
    save(fig, "domain_accuracy_matrix.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        ax.plot(DOMAINS, [fmean(r["forgetting"] for r in forgetting if r["method"] == method and r["domain"] == d) for d in DOMAINS], marker="o", label=method)
    ax.axhline(0, color="gray", linewidth=.5)
    ax.set(ylabel="Final forgetting (signed)")
    ax.legend()
    save(fig, "forgetting_by_domain.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dn4il", "miniimagenet"), default="dn4il")
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dataset == "miniimagenet":
        from analyze_miniimagenet_resnet import analyze
        analyze(args.results_dir or Path("results/miniimagenet_frozen_resnet50_head"))
    else:
        root = args.results_dir or DEFAULT_OUTPUT_DIR
        analyze_runs(load_runs(root), root / "analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
