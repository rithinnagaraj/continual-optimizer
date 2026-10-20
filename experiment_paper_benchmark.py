"""Run the paper's fixed S1 benchmark settings, without reselecting hyperparameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from experiment_paper_development import read, write, gap_rows, require_complete_pool

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 451, 789, 2026)
CONFIGS = {
    "cap": {"lr": 1/300, "weight_decay": 0.0, "k": 7e-5, "T_a": 2000,
            "T_s": 2000, "T_m": 1000000, "release_update": "contest_blocked"},
    "adamw": {"lr": 1e-4, "weight_decay": 0.0},
    "upgd": {"lr": 1e-3, "weight_decay": 0.0, "sigma": 1e-4, "beta_utility": .9999},
    "ewc_oracle": {"lr": 1e-3, "weight_decay": 0.0, "lambda": 1000, "gamma": .95},
    "adam_replay": {"lr": 1e-4/3, "weight_decay": .01, "buffer_capacity": 5000, "replay_count": 96},
}


def analyze(root, methods, seeds):
    metrics, gaps, horizons = [], [], []
    for method in methods:
        scalars = read(root/method/"stream_a/stream_scalars.csv")
        pool = read(root/method/"stream_b/streamb_pool.csv")
        for seed in seeds:
            require_complete_pool(pool, seed)
            a = [r for r in scalars if int(float(r["seed"])) == seed]
            if len(a) != 10000 or {int(float(r["step"])) for r in a} != set(range(100, 1000001, 100)):
                raise ValueError("Benchmark analysis requires every 100-step observation through 1M updates")
            b = [r for r in pool if int(float(r["seed"])) == seed and int(float(r["task"])) == 200
                 and str(r["is_current"]).lower() not in {"true", "1", "1.0"}]
            if len(b) != 39:
                raise ValueError("Benchmark final retention requires all 39 non-current domains")
            p = mean(float(r["batch_accuracy"]) for r in a if int(float(r["step"])) > 500000)
            retention = mean(float(r["accuracy"]) for r in b)
            metrics.append({"method": method, "seed": seed, "plasticity": p,
                            "retention": retention, "balanced": min(p, retention)})
            gaps.extend(gap_rows(pool, method, seed))
            for horizon in (100000, 250000, 500000, 750000, 1000000):
                subset = [float(r["batch_accuracy"]) for r in a if horizon/2 < int(float(r["step"])) <= horizon]
                horizons.append({"method": method, "seed": seed, "horizon": horizon,
                                 "plasticity": mean(subset), "observations": len(subset)})
    out = root/"analysis"
    write(out/"per_seed_metrics.csv", metrics)
    write(out/"retention_by_gap.csv", gaps)
    write(out/"horizon_plasticity.csv", horizons)
    summary = []
    for method in methods:
        subset = [r for r in metrics if r["method"] == method]
        summary.append({"method": method, "seeds": len(subset),
                        **{name: mean(r[name] for r in subset) for name in ("plasticity", "retention", "balanced")},
                        **{name+"_std": stdev(r[name] for r in subset) if len(subset)>1 else 0.0
                           for name in ("plasticity", "retention", "balanced")}})
    write(out/"table03_benchmark.csv", summary)


def train(args):
    import torch
    from cap_adam_experimental import CAPAdam
    from experiment_e3 import cap_parameter_groups
    from phase1_arms import build_arm
    from post_phase1_streams import StreamRuntime
    import experiment_cap_v2_postp1 as engine
    from cap_v2_postp1_common import build_run_config, lock_run_config
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = StreamRuntime(args.data_dir, torch.device(args.device), args.num_workers, 40)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__), ROOT/"phase1_arms.py", ROOT/"cap_adam_experimental.py")}
    for method in args.methods:
        context = {"experiment_version": "cap-paper-fixed-benchmark-v1", "method": method,
                   "optimizer_settings": CONFIGS[method], "source_hashes": hashes}
        for stream in ("a", "b"):
            directory = args.output_dir/method/f"stream_{stream}"
            directory.mkdir(parents=True, exist_ok=True)
            paths = engine.stream_paths(directory, stream)
            config = build_run_config(stream=stream, variant="baseline", device=runtime.device,
                runner_path=Path(__file__), experiment_context=context)
            config["paper_method"] = method
            config["paper_fixed_config"] = CONFIGS[method]
            lock_run_config(paths["config"], config, requested_seeds=args.seeds,
                            resume=args.resume, device=runtime.device)
            for seed in args.seeds:
                def factory(model):
                    if method == "cap":
                        optimizer = CAPAdam(cap_parameter_groups(model, 0), **CONFIGS[method], decay_mode="zero")
                        return optimizer, {"batch_mixer": None, "grad_hook": None, "on_boundary": None, "is_cap": True}
                    return build_arm(method, model, {"methods": {method: {"config": CONFIGS[method]}}}, seed, runtime.device)
                engine.run_cell(stream=stream, variant="baseline", seed=seed, runtime=runtime,
                    paths=paths, resume=args.resume, arm_override="paper_"+method,
                    experiment_context=context, factory_override=factory, record_diagnostics=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/paper_benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-settings", action="store_true", help="Export fixed paper configurations without training")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must be unique")
    if any(s not in SEEDS for s in args.seeds) or args.num_workers < 0:
        parser.error("Use reporting seeds and nonnegative workers")
    if args.dry_run:
        print(json.dumps({"fixed_settings": {m: CONFIGS[m] for m in args.methods}, "seeds": args.seeds,
                          "updates": 2000000*len(args.methods)*len(args.seeds)}, indent=2))
        return 0
    if args.export_settings:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/"fixed_settings.json").write_text(json.dumps(CONFIGS, indent=2)+"\n", encoding="utf-8")
        return 0
    if not args.analyze_only:
        train(args)
    analyze(args.output_dir, args.methods, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
