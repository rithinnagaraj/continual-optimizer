"""Separate post-hoc MNIST-style CAP/UPGD sweep with frozen DN4IL baselines."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean

from dn4il_retune_protocol import (
    BASE_ROOT, OUTPUT_ROOT, VERSION, UPGD_VARIANT, DEVELOPMENT_SEEDS, FINAL_SEEDS,
    METHODS, grid, center, candidate_summary, select, freeze_protocol, read_json,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "tune", "train", "analyze"):
        mode.add_argument(f"--{name}", action="store_true")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--method", choices=METHODS)
    method.add_argument("--all-methods", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+", choices=FINAL_SEEDS)
    seeds.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--baseline-results-dir", type=Path, default=BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data/DomainNet")
    parser.add_argument("--dn4il-splits-root", type=Path, default=Path(__file__).resolve().parent / "data/DN4IL-dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every-updates", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.checkpoint_every_updates < 1:
        parser.error("invalid worker/checkpoint settings")
    if (args.seeds or args.all_seeds) and not args.train:
        parser.error("only final training accepts reporting seeds; smoke/tuning use fixed development seeds")
    if args.seeds and len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate reporting seeds")
    return args


def build_retune_method(method, head, config):
    from miniimagenet_scale_common import build_method, MethodBundle
    from dn4il_upgd_corrected import ConsistentGlobalUPGD
    if method == "cap":
        return build_method(method, head, config)
    if method == "upgd":
        return MethodBundle(ConsistentGlobalUPGD(head.named_parameters(), **config), None,
                            {"variant": UPGD_VARIANT})
    raise ValueError("AdamW and EWC must be reused, not trained in this study")


def prepare_runtime(args, protocol):
    import torch
    import experiment_miniimagenet_resnet as engine
    from experiment_frozen_resnet_head import verify_cache, require_qualification
    from dn4il_dataset import validate_dn4il, DN4ILAdapter
    from miniimagenet_scale_common import build_frozen_resnet50_head
    root, train, test, metadata = validate_dn4il(args.data_root, args.dn4il_splits_root)
    if metadata["manifest_hash"] != protocol["manifest_hash"]:
        raise RuntimeError("dataset identity differs from the original experiment; old baseline numbers would not be paired")
    require_qualification(args.baseline_results_dir, metadata["manifest_hash"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    class ReadOnlyCache(engine.FeatureCacheStore):
        def _build(self, *args, **kwargs):
            raise FileNotFoundError("an original feature cache is missing; retuning never rebuilds or modifies original caches")

    adapter = DN4ILAdapter(metadata)
    cache = ReadOnlyCache(args.baseline_results_dir / "feature_cache", rebuild=False,
        data_root=root, manifest_hash=metadata["manifest_hash"], train_records=train,
        eval_records=test, workers=args.num_workers, batch_size=128, adapter=adapter)
    for seed in (*DEVELOPMENT_SEEDS, *FINAL_SEEDS):
        new = engine._write_schedule(args.output_dir, seed, train, adapter=adapter)
        old = read_json(args.baseline_results_dir / "schedules" / f"seed_{seed}.json")
        if new != old:
            raise RuntimeError(f"retune schedule differs from baseline for seed {seed}")
    model = build_frozen_resnet50_head(DEVELOPMENT_SEEDS[0]).to(device)
    checks = verify_cache(cache, model, device)
    write_json(args.output_dir / "manifest/feature_cache_verification.json", {
        "experiment_version": VERSION, "protocol_hash": protocol["protocol_hash"],
        "reused_cache_root": str(cache.cache_dir.resolve()), "online_cache_checks": checks,
    })
    del model
    # Bind checkpoints to the exact corrected optimizer and CAP implementation.
    from miniimagenet_scale_common import file_hash
    code_hashes = {name: file_hash(Path(__file__).resolve().parent / name) for name in (
        "dn4il_upgd_corrected.py", "cap_adam_experimental.py", "miniimagenet_scale_common.py",
        "experiment_miniimagenet_resnet.py", "experiment_dn4il_retune.py")}
    return dict(output_dir=args.output_dir, data_root=root, train_records=train,
        eval_records=test, manifest_metadata=metadata, device=device, workers=args.num_workers,
        batch_size=128, updates_per_segment=200, max_segments=30, resume=args.resume,
        checkpoint_every_updates=args.checkpoint_every_updates, feature_cache=cache,
        method_factory=build_retune_method, study_protocol={"experiment_version": VERSION,
            "protocol_hash": protocol["protocol_hash"], "upgd_variant": UPGD_VARIANT,
            "code_hashes": code_hashes})


def run_candidate(args, common, method, seed, config, phase, segments=30):
    import experiment_miniimagenet_resnet as engine
    # Candidate hashes use the shared runner's canonical serialization.
    directory = args.output_dir / phase / method
    if phase == "tune":
        directory /= engine._config_id(config)
    return engine.run_single(method=method, seed=seed, config=config,
                             run_dir=directory / f"seed_{seed}",
                             **{**common, "max_segments": segments})


def require_liveness(args, protocol, study_protocol):
    path = args.output_dir / "smoke/upgd_liveness.json"
    if not path.exists():
        raise RuntimeError("run --smoke --all-methods first; UPGD must demonstrate learning before tuning")
    result = read_json(path)
    if (result.get("protocol_hash") != protocol["protocol_hash"]
        or result.get("study_protocol") != study_protocol or not result.get("passed")):
        raise RuntimeError("UPGD learning gate failed or targets another protocol/code; inspect smoke diagnostics")


def tune(args, common, protocol, method):
    import experiment_miniimagenet_resnet as engine
    if method == "upgd":
        require_liveness(args, protocol, common["study_protocol"])
    candidates, failures = [], []
    for index, config in enumerate(grid(method)):
        metrics = []
        for seed in DEVELOPMENT_SEEDS:
            print(f"{method}: candidate {index+1}/36, development seed {seed}: {config}", flush=True)
            try:
                result = run_candidate(args, common, method, seed, config, "tune")
                metrics.append(result["metrics"])
            except FloatingPointError as error:
                failures.append({"config": config, "seed": seed, "reason": str(error)})
                break
        if len(metrics) == 2:
            try:
                candidates.append(candidate_summary(config, metrics))
            except ValueError as error:
                failures.append({"config": config, "reason": str(error)})
        write_json(args.output_dir / "tune" / method / "progress.json", {
            "protocol_hash": protocol["protocol_hash"], "completed_candidates": candidates, "failed_candidates": failures})
    ranked = select(candidates, method)
    # Refuse to publish a best-of-failed-learning UPGD result as a usable baseline.
    if method == "upgd" and ranked[0]["mean_metrics"]["B_scale"] < .05:
        raise RuntimeError("every selected UPGD result remains below 5% B_scale; inspect learning/update diagnostics")
    selected = {"experiment_version": VERSION, "protocol_hash": protocol["protocol_hash"],
        "method": method, "manifest_hash": protocol["manifest_hash"], "development_seeds": list(DEVELOPMENT_SEEDS),
        "selection": protocol["selection"], "candidate_count": len(grid(method)),
        "config": ranked[0]["config"], "mean_metrics": ranked[0]["mean_metrics"],
        "ranked_candidates": ranked, "failed_candidates": failures,
        "study_protocol": common["study_protocol"],
        "selected_config_hash": engine.payload_hash(ranked[0]["config"])}
    write_json(args.output_dir / "selected_configs" / f"{method}.json", selected)
    print(f"Selected {method}: {selected['config']}; mean B={selected['mean_metrics']['B_scale']:.6f}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    protocol = freeze_protocol(args.baseline_results_dir, args.output_dir)
    if args.prepare:
        print(f"Protocol frozen at {args.output_dir / 'protocol.json'}; 36 candidates x 2 seeds for each method.")
        return 0
    if args.analyze:
        from analyze_dn4il_retune import analyze
        analyze(args.baseline_results_dir, args.output_dir)
        return 0
    common = prepare_runtime(args, protocol)
    methods = (args.method,) if args.method else METHODS
    for method in methods:
        if args.smoke:
            result = run_candidate(args, common, method, DEVELOPMENT_SEEDS[0], center(method), "smoke", segments=6)
            rows = result["rows"]["segments"]
            p = fmean(float(r["mean_online_accuracy"]) for r in rows)
            gain = fmean(float(r["learning_gain"]) for r in rows)
            if method == "upgd":
                passed = math.isfinite(p) and math.isfinite(gain) and p >= .05 and gain >= .01
                write_json(args.output_dir / "smoke/upgd_liveness.json", {
                    "protocol_hash": protocol["protocol_hash"], "passed": passed,
                    "mean_online_accuracy": p, "mean_learning_gain": gain, "config": center(method),
                    "study_protocol": common["study_protocol"]})
                if not passed:
                    raise RuntimeError("UPGD failed the predeclared learning gate; inspect smoke logs before tuning")
            print(f"{method} smoke: online accuracy={p:.4f}, learning gain={gain:.4f}")
        elif args.tune:
            tune(args, common, protocol, method)
        elif args.train:
            import experiment_miniimagenet_resnet as engine
            selected = read_json(args.output_dir / "selected_configs" / f"{method}.json")
            if (selected.get("protocol_hash") != protocol["protocol_hash"]
                or selected.get("method") != method or selected.get("experiment_version") != VERSION
                or selected.get("study_protocol") != common["study_protocol"]
                or selected.get("selected_config_hash") != engine.payload_hash(selected["config"])
                or selected["config"] not in grid(method)):
                raise RuntimeError("selected configuration does not match this frozen study/code")
            for seed in args.seeds or FINAL_SEEDS:
                run_candidate(args, common, method, seed, selected["config"], "train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
