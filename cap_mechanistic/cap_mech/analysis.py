from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from .records import read_table, write_csv
from .statistics import cluster_bootstrap, paired_deltas, spearman
from .plotting import seed_lines, branch_plot, frontier
from .checkpointing import require_gate, prepare_run, atomic_json, digest, file_hash, source_version
from .runner import NAMES, manifest, provenance

def summarize_causal(root, experiment):
    tasks, retention, traces, curves = [read_table(root, t) for t in ("tasks", "retention", "mechanism_traces", "learning_curves")]
    rows = []
    keys = ["seed", "checkpoint_task", "branch"]
    for block, group in tasks.groupby(keys, sort=False):
        seed, age, branch = block
        select = lambda df: df[(df.seed == seed) & (df.checkpoint_task == age) & (df.branch == branch)]
        ret = select(retention)
        ret = ret[ret.task > age]
        final = ret[(ret.task == ret.task.max()) & (ret.within_task_update == ret.within_task_update.max())]
        measured = ret[~ret.is_current] if experiment in ("m4", "m6") else final
        mech = select(traces)
        row = {k: group.iloc[0][k] for k in ("source_checkpoint_sha256", "baseline_config_sha256", "experiment_config_sha256", "source_version")}
        rows.append({**row, **dict(zip(keys, block)), "early_auc": group.early_auc.mean(),
                     "plasticity": group.plasticity.mean(), "task_end_accuracy": group.task_end_accuracy.mean(),
                     "retention": measured.accuracy.mean(), "forgetting": measured.forgetting.mean(),
                     "old_task_accuracy_before": measured.before_accuracy.mean(),
                     "final_forgetting": final.forgetting.mean(), "worst_old_task_forgetting": final.forgetting.max(),
                     "bind_rate": group.bind_rate_mean.mean(), "median_importance": mech.median_importance.mean(),
                     "median_radius": mech.median_radius.mean(), "bound_importance_decay": mech.bound_importance_decay.mean()})
    metrics = pd.DataFrame(rows)
    baseline = {"m2": "k_1", "m3": "baseline", "m4": "true_importance", "m6": "release_1000000.0"}[experiment]
    outcomes = ["early_auc", "plasticity", "retention", "forgetting", "bind_rate"]
    paired = paired_deltas(metrics, baseline, outcomes)
    write_csv(root/"raw"/f"{experiment}_branch_metrics.csv", paired)
    write_csv(root/"raw"/f"{experiment}_learning_curves.csv", curves)
    write_csv(root/"raw"/f"{experiment}_mechanism_traces.csv", traces)
    write_csv(root/"raw"/f"{experiment}_old_task_retention.csv", retention)
    inference = paired.copy()
    if experiment == "m4":
        write_csv(root/"raw"/"m4_shuffle_replicates.csv", paired[paired.branch.str.startswith("shuffle_")])
        inference["branch"] = inference.branch.map(lambda b: "layer_shuffled" if b.startswith("shuffle_") else b)
        numeric = inference.select_dtypes(include="number").columns.difference(["seed", "checkpoint_task"])
        grouped = inference.groupby(keys, as_index=False)
        averaged = grouped[list(numeric)].mean()
        provenance_cols = ["source_checkpoint_sha256", "baseline_config_sha256", "experiment_config_sha256", "source_version"]
        inference = averaged.merge(grouped[provenance_cols].first(), on=keys)
    write_csv(root/"summary"/f"{experiment}_block_effects.csv", inference)
    summaries = []
    for branch, group in inference.groupby("branch", sort=False):
        for outcome in outcomes:
            summaries.append({"branch": branch, "metric": f"delta_{outcome}", **cluster_bootstrap(group, f"delta_{outcome}")})
    names = {"m2": "m2_k_response_summary.csv", "m3": "m3_state_reset_summary.csv", "m4": "m4_summary.csv", "m6": "m6_release_summary.csv"}
    write_csv(root/"summary"/names[experiment], summaries)
    if experiment in ("m2", "m6"):
        correlations = []
        for (seed, age), group in metrics.groupby(["seed", "checkpoint_task"]):
            if experiment == "m2":
                g = group[group.branch.str.startswith("k_")]
                x = g.branch.str[2:].astype(float)
            else:
                g = group
                x = g.branch.map(lambda b: 0. if b == "release_infinity" else 1/float(b.removeprefix("release_")))
            for outcome in ("early_auc", "forgetting", "bind_rate", "median_importance", "median_radius"):
                correlations.append({"seed": seed, "checkpoint_task": age, "metric": outcome,
                                     "spearman": spearman(x, g[outcome])})
        write_csv(root/"summary"/f"{experiment}_rank_responses.csv", correlations)
    if experiment in ("m4", "m6"):
        gaps = retention[(retention.task > retention.checkpoint_task) & ~retention.is_current].copy()
        gaps["gap_bucket"] = pd.cut(gaps.revisit_gap, [0, 1, 5, 10, 20, np.inf], labels=["1", "2-5", "6-10", "11-20", "21+"], include_lowest=True)
        write_csv(root/"raw"/f"{experiment}_gap_retention.csv", gaps)
    if experiment == "m2":
        frontier(metrics, root/"summary"/"m2_k_stability_plasticity.png")
        finite = metrics[metrics.branch.str.startswith("k_")].copy()
        finite["k_multiplier"] = finite.branch.str[2:].astype(float)
        seed_lines(finite, "k_multiplier", "bind_rate", root/"summary"/"m2_k_bind_response.png", "checkpoint_task")
    elif experiment == "m3":
        tax = paired[paired.branch == "box_off"]
        seed_lines(tax, "checkpoint_task", "delta_early_auc", root/"summary"/"m3_constraint_tax_by_age.png")
        seed_lines(curves, "within_task_update", "accuracy", root/"summary"/"m3_branch_learning_curves.png", "branch")
        branch_plot(inference[inference.checkpoint_task == inference.checkpoint_task.max()], ["early_auc", "forgetting"], root/"summary"/"m3_retention_cost.png")
    else:
        branch_plot(inference, ["retention", "plasticity"], root/"summary"/("m4_true_vs_shuffle.png" if experiment == "m4" else "m6_release_response.png"))

CLAIMS = [
    ("m1", "CAP importance is functionally meaningful", "High-I damage is no greater than random/bottom"),
    ("m2", "k causally controls stability and plasticity", "Geometry changes without systematic behavioral P/R changes"),
    ("m3", "Accumulated CAP state contributes to late plasticity loss", "Removing boxes/state fails to recover late plasticity"),
    ("m4", "Coordinate assignment of importance matters", "Shuffled importance retains function like true importance"),
    ("m5", "CAP state predicts plasticity beyond stream age", "Combined held-out prediction does not improve over baseline"),
    ("m6", "Release acts as a second control dial (secondary)", "No consistent mechanistic and behavioral response")]

def claim_effects(root, experiment):
    directory = root/NAMES[experiment]/"summary"
    if experiment == "m1":
        frame = pd.read_csv(directory/"m1_summary.csv")
        frame = frame[frame.comparator == "importance"].copy()
        frame["random_minus_bottom"] = frame.random_damage_mean-frame.bottom_imp_damage
        frame = frame.groupby(["seed", "checkpoint_task"], as_index=False).mean(numeric_only=True)
        return [("top minus random damage", frame, "top_minus_random_mean", 1),
                ("random minus bottom damage", frame, "random_minus_bottom", 1)]
    if experiment in ("m2", "m6"):
        frame = pd.read_csv(directory/f"{experiment}_rank_responses.csv")
        expected = [("early_auc", 1), ("forgetting", 1), ("bind_rate", -1)] if experiment == "m2" else [("early_auc", 1), ("forgetting", 1), ("median_importance", -1), ("median_radius", 1)]
        return [(f"rank response {m}", frame[frame.metric == m], "spearman", direction) for m, direction in expected]
    if experiment == "m3":
        frame = pd.read_csv(directory/"m3_block_effects.csv")
        tax = frame[frame.branch == "box_off"]
        ages = [{"seed": seed, "age_response": spearman(g.checkpoint_task, g.delta_early_auc)} for seed, g in tax.groupby("seed")]
        return [("late constraint tax", tax[tax.checkpoint_task == tax.checkpoint_task.max()], "delta_early_auc", 1),
                ("tax growth with age", pd.DataFrame(ages), "age_response", 1)]
    if experiment == "m4":
        frame = pd.read_csv(directory/"m4_block_effects.csv")
        return [("shuffle retention minus true", frame[frame.branch == "layer_shuffled"], "delta_retention", -1)]
    frame = pd.read_csv(directory/"m5_model_metrics.csv")
    frame = frame[(frame.outcome == "early_auc") & (frame.seed != "overall")]
    pivot = frame.pivot(index="seed", columns="model", values="MAE").reset_index()
    pivot["improvement"] = pivot.baseline-pivot.combined
    return [("baseline MAE minus combined MAE", pivot, "improvement", 1)]

def analyze(args):
    verified = require_gate(args.output_dir, args.allow_unverified_reference)
    root = Path(args.output_dir)
    inputs = {str(p.relative_to(root)): file_hash(p) for name in NAMES.values()
              for p in sorted((root/name/"summary").glob("*.csv"))}
    config = manifest(args, "analysis", inputs)
    output = root/"analysis"
    prepare_run(output, config, args.resume)
    effects, claims = [], []
    for experiment, claim, falsifier in CLAIMS:
        complete = root/NAMES[experiment]/"complete.json"
        if not complete.exists():
            claims.append({"claim": claim, "experiment": experiment, "status": "NOT SUPPORTED",
                           "reason": "No completed evidence available", "falsifying_outcome": falsifier})
            continue
        result_config = json.loads((complete.parent/"run_config.json").read_text())
        completion = json.loads(complete.read_text())
        if result_config["source_version"] != source_version() or completion["config_hash"] != digest(result_config):
            raise RuntimeError("Completed experiment source/config is stale")
        tests = []
        for label, frame, metric, direction in claim_effects(root, experiment):
            interval = cluster_bootstrap(frame, metric)
            positive = direction*interval["ci_low"] > 0 if direction == 1 else direction*interval["ci_high"] > 0
            negative = direction*interval["ci_high"] <= 0 if direction == 1 else direction*interval["ci_low"] <= 0
            tests.append((positive, negative, interval["n_seeds"]))
            for seed, g in frame.groupby("seed"):
                effects.append({"experiment": experiment, "contrast": label, "seed": seed,
                                "effect": g[metric].mean(), "expected_direction": direction, **interval})
        sufficient = verified and completion.get("scientifically_verified") and all(t[2] == 5 for t in tests)
        status = ("SUPPORTED" if sufficient and all(t[0] for t in tests) else
                  "NOT SUPPORTED" if sufficient and all(t[1] for t in tests) else "MIXED")
        claims.append({"claim": claim, "experiment": experiment, "status": status,
                       "reason": "All required directional cluster CIs exclude zero" if status == "SUPPORTED" else
                       "Unverified/incomplete evidence, uncertain intervals, or inconsistent required effects" if status == "MIXED" else
                       "All required contrasts fail the prespecified direction",
                       "falsifying_outcome": falsifier})
    prov = provenance(digest(inputs), config)
    for row in effects+claims:
        row.update(prov)
    write_csv(output/"mechanistic_summary.csv", effects)
    write_csv(output/"claim_evidence_table.csv", claims)
    (output/"figures").mkdir(exist_ok=True)
    for name in NAMES.values():
        for image in (root/name/"summary").glob("*.png"):
            shutil.copy2(image, output/"figures"/image.name)
    lines = ["# CAP mechanistic report", "", "Scope: controlled Permuted-MNIST continual learning.",
             "", "Reference verified: "+str(verified),
             "", "Synthetic smoke outputs are implementation checks and provide no scientific evidence.",
             "", "Inference resamples whole training seeds (10,000 draws; 95% percentile intervals).",
             "SUPPORTED requires all specified directional intervals to exclude zero, five seeds, and verified outputs.",
             "NOT SUPPORTED means all intervals fail direction, or no completed evidence exists (distinguished below).",
             "Other outcomes are MIXED. Prediction is not causation. No p-values are reported.", ""]
    for c in claims:
        lines.extend([f"- **{c['experiment'].upper()}: {c['status']}** — {c['claim']}. {c['reason']}."])
    lines.extend(["", "All seed effects and intervals: `mechanistic_summary.csv`.",
                  "", "M1 requires high > random > low damage; M2 requires positive k/plasticity and k/forgetting and negative k/binding rank responses;",
                  "M3 requires positive late constraint tax and increasing tax with age; M4 requires shuffle retention loss;",
                  "M5 requires lower combined held-out MAE; M6 requires faster-release plasticity/forgetting gains, lower importance and larger radii.",
                  "", "These are transparent report rules, not proof of universal interpretability or transfer beyond this benchmark."])
    (output/"mechanistic_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    atomic_json(output/"complete.json", {"config_hash": digest(config)})
