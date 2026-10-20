"""Frozen, post-hoc CAP/UPGD retuning protocol (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev

VERSION = "dn4il-frozen-resnet50-head-mnist-sweep-v2"
BASE_VERSION = "dn4il-frozen-resnet50-head-recurrent-v1"
UPGD_VARIANT = "global-upgd-consistent-bias-normalization-v1"
ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT / "results" / "dn4il_frozen_resnet50_head"
OUTPUT_ROOT = ROOT / "results" / "dn4il_frozen_resnet50_head_retune"
DEVELOPMENT_SEEDS = (31415, 27182)
FINAL_SEEDS = (42, 123, 2026, 451, 789)
METHODS = ("cap", "upgd")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def grid(method):
    if method == "cap":
        return [dict(lr=lr, k=k, T_a=ta, T_s=2000.0, T_m=tm, weight_decay=0.0)
                for lr, k, ta, tm in itertools.product(
                    (3e-4, 1e-3, 3e-3), (3e-5, 7e-5, 1.4e-4), (2000.0, 10000.0), (10000.0, 50000.0))]
    if method == "upgd":
        return [dict(lr=lr, sigma=sigma, beta_utility=beta, weight_decay=wd)
                for lr, sigma, beta, wd in itertools.product(
                    (.01 / 3, .01, .03), (.01, .1), (.9999, .99999), (1e-4, 1e-3, 1e-2))]
    raise ValueError(f"only CAP and UPGD are retuned: {method}")


def center(method):
    if method == "cap":
        return dict(lr=1e-3, k=7e-5, T_a=2000.0, T_s=2000.0, T_m=10000.0, weight_decay=0.0)
    if method == "upgd":
        return dict(lr=.01, sigma=.01, beta_utility=.9999, weight_decay=1e-4)
    raise ValueError(method)


def candidate_summary(config, metrics):
    if len(metrics) != len(DEVELOPMENT_SEEDS):
        raise ValueError("selection requires both development seeds")
    clean = []
    for row in metrics:
        p, r = float(row["P_scale"]), float(row["R_scale"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, r)):
            raise ValueError("nonfinite or invalid development accuracy")
        clean.append({"P_scale": p, "R_scale": r, "B_scale": min(p, r)})
    return {"config": config, "metrics_by_seed": clean,
            "mean_metrics": {k: fmean(r[k] for r in clean) for k in clean[0]},
            "B_sd": pstdev(r["B_scale"] for r in clean)}


def select(candidates, method):
    reference = center(method)
    def rank(row):
        m = row["mean_metrics"]
        distance = sum(abs(float(v) - reference[k]) / max(abs(reference[k]), 1e-12)
                       for k, v in row["config"].items())
        return (m["B_scale"], min(m["P_scale"], m["R_scale"]),
                (m["P_scale"] + m["R_scale"]) / 2, -row["B_sd"], -distance)
    if not candidates:
        raise ValueError("no valid candidates")
    return sorted(candidates, key=rank, reverse=True)


def validate_output_path(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("retune output and baseline results must be separate, non-overlapping directories")


def source_fingerprint(source):
    """Pin original scientific logs/configs; never regenerate original analysis."""
    source = Path(source)
    paths = ["manifest/dn4il_manifest.json", "qualification/qualification.json"]
    paths += [f"schedules/seed_{seed}.json" for seed in (*DEVELOPMENT_SEEDS, *FINAL_SEEDS)]
    checkpoints = {}
    for method in ("adamw", "ewc"):
        paths.append(f"selected_configs/{method}.json")
        for seed in FINAL_SEEDS:
            folder = f"train/{method}/seed_{seed}"
            config = read_json(source / folder / "run_config.json")
            if config.get("experiment_version") != BASE_VERSION or config.get("method") != method:
                raise ValueError(f"wrong baseline identity: {folder}")
            if (config.get("batch_size"), config.get("updates_per_segment"), config.get("max_segments")) != (128, 200, 30):
                raise ValueError(f"wrong baseline stream: {folder}")
            checkpoint = f"{folder}/checkpoint_final.pt"
            checkpoints[checkpoint] = (source / checkpoint).stat().st_size
            paths += [f"{folder}/{name}" for name in (
                "run_config.json", "stream_segments.csv", "online_steps.csv", "domain_eval.csv", "optimizer_diagnostics.csv")]
    return {"files": {p: file_digest(source / p) for p in paths}, "checkpoint_sizes": checkpoints}


def freeze_protocol(source=BASE_ROOT, output=OUTPUT_ROOT):
    validate_output_path(source, output)
    source, output = Path(source), Path(output)
    manifest = read_json(source / "manifest/dn4il_manifest.json")
    protocol = {
        "experiment_version": VERSION, "baseline_experiment_version": BASE_VERSION,
        "disclosure": "Post-hoc CAP/UPGD retuning after observing v1 reporting results. AdamW and EWC are reused unchanged; tuning budgets differ from those baselines.",
        "manifest_hash": manifest["manifest_hash"], "sources": source_fingerprint(source),
        "development_seeds": list(DEVELOPMENT_SEEDS), "reporting_seeds": list(FINAL_SEEDS),
        "grids": {m: grid(m) for m in METHODS}, "upgd_variant": UPGD_VARIANT,
        "selection": "Every candidate on both development seeds; maximize mean_seed(min(P_scale,R_scale)); then min(mean P,mean R), mean(P+R)/2, lower B SD, central config, grid order.",
        "stream": {"segments": 30, "cycles": 5, "updates_per_segment": 200, "batch_size": 128},
        "upgd_liveness": {"seed": DEVELOPMENT_SEEDS[0], "segments": 6,
                          "min_mean_online_accuracy": .05, "min_mean_learning_gain": .01},
    }
    protocol["protocol_hash"] = digest(protocol)
    path = output / "protocol.json"
    if path.exists():
        if read_json(path) != protocol:
            raise RuntimeError("frozen retune protocol or original baselines changed; use a fresh output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("nonempty output has no frozen protocol; use a new directory")
        write_json(path, protocol)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results-dir", type=Path, default=BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    frozen = freeze_protocol(args.baseline_results_dir, args.output_dir)
    print(f"Frozen {VERSION}: 36 candidates x 2 development seeds per method. Protocol: {args.output_dir / 'protocol.json'}")
