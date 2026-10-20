"""Development variants and analysis for Appendix Tables 20--23."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "original": {"release_update": "legacy"},
    "gated_anchor": {"release_update": "legacy", "paper_anchor_gated": True},
    "conflict_release": {"release_update": "legacy", "paper_conflict_release": True},
    "final": {"release_update": "contest_blocked"},
    "selective_box": {"release_update": "contest_blocked", "paper_selective_box": True},
    "adam_v": {"release_update": "contest_blocked", "importance_evidence": "adam_v"},
    "row_slack": {"release_update": "contest_blocked", "box_geometry": "q4_slack"},
    "momentum_restart": {"release_update": "contest_blocked", "momentum_response": "conflict_restart"},
}
WINDOWS = (("0-100k", 0, 100000), ("100k-250k", 100000, 250000),
           ("250k-500k", 250000, 500000), ("500k-750k", 500000, 750000),
           ("750k-1M", 750000, 1000000))
GAPS = (("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-40", 21, 40), ("41-80", 41, 80))


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        raise ValueError(f"No measurements for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def gap_rows(rows, variant, seed):
    last = {}
    values = {name: [] for name, _, _ in GAPS}
    tasks = {}
    for row in rows:
        if int(float(row["seed"])) == seed:
            tasks.setdefault(int(float(row["task"])), []).append(row)
    for task, block in sorted(tasks.items()):
        current = [int(float(r["pool_index"])) for r in block if str(r["is_current"]).lower() in {"true", "1", "1.0"}]
        if len(current) != 1:
            raise ValueError("Every pool evaluation must have exactly one current domain")
        for row in block:
            domain = int(float(row["pool_index"]))
            if domain == current[0] or domain not in last:
                continue
            gap = task - last[domain]
            for name, low, high in GAPS:
                if low <= gap <= high:
                    values[name].append(float(row["accuracy"]))
        last[current[0]] = task
    return [{"variant": variant, "seed": seed, "gap_bucket": name,
             "retention": mean(v), "observations": len(v)} for name, v in values.items() if v]


def require_complete_pool(rows, seed):
    subset = [r for r in rows if int(float(r["seed"])) == seed]
    expected = {(task, domain) for task in range(1, 201) for domain in range(40)}
    observed = {(int(float(r["task"])), int(float(r["pool_index"]))) for r in subset}
    if len(subset) != len(expected) or observed != expected:
        raise ValueError("Paper pool analysis requires all 200 x 40 evaluations per seed")


def analyze(root, variants, seeds, baseline_pool=None):
    summary, temporal, gaps, within = [], [], [], []
    for variant in variants:
        a = read(root / variant / "stream_a/stream_scalars.csv")
        b = read(root / variant / "stream_b/streamb_pool.csv")
        for seed in seeds:
            require_complete_pool(b, seed)
            scalars = [r for r in a if int(float(r["seed"])) == seed]
            expected = set(range(100, 1_000_001, 100))
            if len(scalars) != len(expected) or {int(float(r["step"])) for r in scalars} != expected:
                raise ValueError("Development analysis requires the complete 1M-update scalar trajectory")
            final = [r for r in b if int(float(r["seed"])) == seed and int(float(r["task"])) == 200
                     and str(r["is_current"]).lower() not in {"true", "1", "1.0"}]
            if len(final) != 39:
                raise ValueError("Final retention requires 39 non-current domains")
            p = mean(float(r["batch_accuracy"]) for r in scalars if int(float(r["step"])) > 500000)
            retention = mean(float(r["accuracy"]) for r in final)
            summary.append({"variant": variant, "seed": seed, "plasticity": p,
                            "retention": retention, "balanced": min(p, retention)})
            gaps.extend(gap_rows(b, variant, seed))
            for name, low, high in WINDOWS:
                block = [r for r in scalars if low < int(float(r["step"])) <= high]
                temporal.append({"variant": variant, "seed": seed, "window": name,
                                 **{key: mean(float(r[key]) for r in block) for key in
                                    ("batch_accuracy", "bind_rate", "median_imp", "median_radius")}})
            for local in (100, 300, 1000, 2000, 3000, 5000):
                block = [r for r in scalars if int(float(r["step"])) > 500000
                         and (int(float(r["step"])) - 1) % 5000 + 1 == local]
                within.append({"variant": variant, "seed": seed, "within_task_update": local,
                               "accuracy": mean(float(r["batch_accuracy"]) for r in block)})
    if baseline_pool is not None:
        baseline = [r for r in read(baseline_pool) if r["arm"] == "paper_upgd"]
        for seed in seeds:
            require_complete_pool(baseline, seed)
            gaps.extend(gap_rows(baseline, "upgd_reference", seed))
    out = root / "analysis"
    for name, rows in (("table20_variants.csv", summary), ("table21_temporal.csv", temporal),
                       ("table22_gap_retention.csv", gaps), ("table23_within_task.csv", within)):
        write(out / name, rows)


def train(args):
    import torch
    from cap_adam_experimental import CAPAdam
    import experiment_cap_v2_postp1 as engine
    from cap_v2_postp1_common import build_run_config, lock_run_config
    from experiment_e3 import cap_parameter_groups
    from post_phase1_streams import StreamRuntime

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = StreamRuntime(args.data_dir, torch.device(args.device), args.num_workers, 40)
    source_files = (Path(__file__), ROOT / "cap_adam_experimental.py", ROOT / "experiment_cap_v2_postp1.py")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    for variant in args.variants:
        settings = VARIANTS[variant]
        recorded = dict(settings)
        if recorded.get("paper_conflict_release"):
            recorded["release_update"] = "conflict_gated"
        context = {"experiment_version": "cap-paper-development-v1", "paper_variant": variant,
                   "optimizer_settings": recorded, "source_hashes": hashes}
        def factory(model):
            optimizer = CAPAdam(cap_parameter_groups(model, 0), lr=1/300, betas=(.9, .999),
                eps=1e-8, weight_decay=0, k=7e-5, T_a=2000, T_s=2000, T_m=1e6,
                retention="release", decay_mode="zero", **settings)
            return optimizer, {"batch_mixer": None, "grad_hook": None, "on_boundary": None, "is_cap": True}
        for stream in ("a", "b"):
            directory = args.output_dir / variant / f"stream_{stream}"
            directory.mkdir(parents=True, exist_ok=True)
            paths = engine.stream_paths(directory, stream)
            config = build_run_config(stream=stream, variant="baseline", device=runtime.device,
                runner_path=Path(__file__), experiment_context=context)
            lock_run_config(paths["config"], config, requested_seeds=args.seeds,
                            resume=args.resume, device=runtime.device)
            for seed in args.seeds:
                engine.run_cell(stream=stream, variant="baseline", seed=seed, runtime=runtime,
                    paths=paths, resume=args.resume, arm_override="cap_paper_" + variant,
                    experiment_context=context, factory_override=factory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_development")
    parser.add_argument("--baseline-pool", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Variants and seeds must be unique")
    if any(s not in {42, 123, 451, 789, 2026} for s in args.seeds) or args.num_workers < 0:
        parser.error("Use reporting seeds and nonnegative workers")
    if args.dry_run:
        print(json.dumps({"variants": {v: VARIANTS[v] for v in args.variants}, "seeds": args.seeds,
                          "updates": 2_000_000 * len(args.variants) * len(args.seeds)}, indent=2))
        return 0
    if not args.analyze_only:
        train(args)
    analyze(args.output_dir, args.variants, args.seeds, args.baseline_pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
