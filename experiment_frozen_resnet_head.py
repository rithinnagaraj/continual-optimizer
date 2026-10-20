"""Shared frozen ResNet-50 + MLP runner, with official DN4IL natural domains."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean

import torch

import experiment_miniimagenet_resnet as engine
from dn4il_dataset import (DEFAULT_DATA_ROOT, DEFAULT_SPLITS_ROOT, DEFAULT_OUTPUT_DIR,
                           DN4ILAdapter, EXPERIMENT_VERSION, validate_dn4il)
from frozen_head_metrics import dn4il_metrics
from miniimagenet_scale_common import (BATCH_SIZE, UPDATES_PER_SEGMENT, DEVELOPMENT_SEEDS,
    FINAL_SEEDS, METHODS, DEFAULT_HEAD_HIDDEN_DIM, BACKBONE_WEIGHTS, MODEL_NAME,
    atomic_write_json, build_frozen_resnet50_head, default_config, model_parameter_counts,
    payload_hash)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dn4il", "miniimagenet"), default="dn4il")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "build-feature-cache", "qualify", "calibrate", "smoke", "tune", "train"):
        mode.add_argument(f"--{name}", action="store_true")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--method", choices=METHODS)
    method.add_argument("--all-methods", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+")
    seeds.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dn4il-splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, choices=(DEFAULT_HEAD_HIDDEN_DIM,), default=DEFAULT_HEAD_HIDDEN_DIM)
    parser.add_argument("--checkpoint-every-updates", type=int, default=UPDATES_PER_SEGMENT)
    parser.add_argument("--cap-k-center", type=float)
    parser.add_argument("--smoke-segments", type=int, default=2)
    parser.add_argument("--smoke-updates-per-segment", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or min(args.checkpoint_every_updates, args.smoke_updates_per_segment, args.smoke_batch_size) < 1:
        parser.error("workers must be nonnegative and update/batch/checkpoint counts positive")
    if not 1 <= args.smoke_segments <= 30:
        parser.error("--smoke-segments must be in 1..30")
    if args.cap_k_center is not None and (not math.isfinite(args.cap_k_center) or args.cap_k_center <= 0):
        parser.error("--cap-k-center must be finite and positive")
    if args.tune and (not args.method or args.all_methods):
        parser.error("--tune requires one --method")
    if args.qualify and (args.all_methods or args.method not in (None, "adamw")):
        parser.error("qualification runs AdamW only")
    if args.calibrate and (args.all_methods or args.method not in (None, "cap")):
        parser.error("calibration runs CAP only")
    if (args.qualify or args.calibrate or args.tune) and (args.seeds or args.all_seeds):
        parser.error("qualification, calibration, and tuning use fixed development seeds")
    if args.smoke and args.seeds and any(seed in FINAL_SEEDS for seed in args.seeds):
        parser.error("smoke uses development seeds, disjoint from final reporting")
    return args


def prepare(args):
    output = args.output_dir.resolve()
    cache = (args.feature_cache_dir or output / "feature_cache").resolve()
    # Reject old output trees even if only checkpoints, manifests, or caches exist.
    for root in (output, cache):
        if any("miniimagenet" in part.lower() for part in root.parts):
            raise ValueError("DN4IL needs a separate output/cache directory")
    if output.exists():
        for path in output.rglob("run_config.json"):
            if json.loads(path.read_text(encoding="utf-8")).get("experiment_version") != EXPERIMENT_VERSION:
                raise ValueError(f"output contains another experiment: {path}")
    root, train, test, metadata = validate_dn4il(args.data_root, args.dn4il_splits_root, output / "manifest")
    adapter = DN4ILAdapter(metadata)
    for seed in (*DEVELOPMENT_SEEDS, *FINAL_SEEDS):
        engine._write_schedule(output, seed, train, adapter=adapter)
    return root, train, test, metadata, adapter


def verify_cache(store, model, device):
    """Recompute original batches containing first/middle/last samples.

    CUDA convolution kernels can use different arithmetic for different batch
    shapes. Reproduce the cache loader's batches, including its short final
    batch, so this tests cache equivalence without changing the computation.
    """
    checks = []
    for domain_id, domain in enumerate(store.adapter.domains):
        for split in ("train", "eval"):
            records = store._records(split, domain_id)
            sample_indices = sorted({0, len(records) // 2, len(records) - 1})
            batch_starts = sorted({i // store.batch_size * store.batch_size for i in sample_indices})
            batches = []
            maximum_error = 0.0
            for start in batch_starts:
                stop = min(start + store.batch_size, len(records))
                batch_records = records[start:stop]
                cached = store.dataset(batch_records, split=split, domain_id=domain_id, model=model, device=device)
                online = store.adapter.images(batch_records)
                images, labels = next(iter(engine._loader(online, batch_size=store.batch_size, workers=0, seed=0)))
                with torch.inference_mode():
                    features = model.extract_features(images.to(device)).cpu()
                try:
                    torch.testing.assert_close(cached.tensors[0], features, rtol=1e-4, atol=1e-5)
                except AssertionError as error:
                    raise RuntimeError(
                        f"feature-cache verification failed for {domain}/{split}, original batch "
                        f"[{start}:{stop}]. Do not train with this cache. Check that the cache was "
                        "built with the same weights, device/backend settings, and batch size; "
                        "rebuild it in the current environment if necessary.\n" + str(error)
                    ) from error
                if not torch.equal(cached.tensors[1], labels):
                    raise RuntimeError(f"online/cache label mismatch for {domain}/{split}")
                maximum_error = max(maximum_error, float((cached.tensors[0] - features).abs().max()))
                batches.append({"start_index": start, "stop_index": stop, "batch_size": stop - start})
            checks.append({"domain": domain, "split": split,
                           "source_ids": [records[i]["source_id"] for i in sample_indices],
                           "verification_batches": batches,
                           "checked_features": sum(batch["batch_size"] for batch in batches),
                           "max_absolute_error": maximum_error})
    return checks


def require_qualification(output, manifest_hash):
    path = output / "qualification" / "qualification.json"
    if not path.exists():
        raise RuntimeError("run --qualify before tuning or final training")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("manifest_hash") != manifest_hash or report.get("experiment_version") != EXPERIMENT_VERSION:
        raise RuntimeError("qualification targets another dataset/experiment")
    if not report.get("qualified"):
        raise RuntimeError("AdamW qualification failed: frozen head-only Domain-IL did not show sufficient learning/interference; inspect qualification analysis and stop")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep the existing mini-ImageNet CLI and output identity available unchanged.
    if "--dataset" in argv and argv[argv.index("--dataset") + 1] == "miniimagenet":
        index = argv.index("--dataset")
        return engine.main(argv[:index] + argv[index + 2:])
    if "--dataset=miniimagenet" in argv:
        return engine.main([arg for arg in argv if arg != "--dataset=miniimagenet"])
    args = parse_args(argv)
    root, train, test, metadata, adapter = prepare(args)
    print(json.dumps({"split_files_detected": len(metadata["split_file_hashes"]),
                      "domain_counts": {domain: {split: values["count"] for split, values in counts.items()}
                                        for domain, counts in metadata["domain_counts"].items()}, "num_classes": 100,
                      "train_test_leakage": False}, indent=2))
    if args.prepare:
        return 0
    if args.tune or args.train:
        require_qualification(args.output_dir, metadata["manifest_hash"])
    if args.tune and args.method == "cap":
        calibration_path = args.output_dir / "calibration" / "cap_k_center.json"
        if not calibration_path.exists():
            raise RuntimeError("run --calibrate before CAP tuning")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("manifest_hash") != metadata["manifest_hash"] or not calibration.get("healthy"):
            raise RuntimeError("CAP calibration is incompatible or degenerate; inspect diagnostics before setting a new center")
        if args.cap_k_center is not None and args.cap_k_center != calibration["k_center"]:
            raise ValueError("--cap-k-center must match the accuracy-blind calibrated center")
        args.cap_k_center = calibration["k_center"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    store = engine.FeatureCacheStore(args.feature_cache_dir or args.output_dir / "feature_cache",
        rebuild=args.rebuild_feature_cache, data_root=root, manifest_hash=metadata["manifest_hash"],
        train_records=train, eval_records=test, workers=args.num_workers, batch_size=BATCH_SIZE, adapter=adapter)
    start = time.perf_counter()
    model = build_frozen_resnet50_head(DEVELOPMENT_SEEDS[0], hidden_dim=args.head_hidden_dim).to(device)
    for domain_id in range(6):
        for split in ("train", "eval"):
            store._load(split, domain_id, model, device)
    checks = verify_cache(store, model, device)
    atomic_write_json(args.output_dir / "manifest" / "feature_cache_verification.json", {
        "experiment_version": EXPERIMENT_VERSION, "backbone_weights": BACKBONE_WEIGHTS,
        "feature_dim": model.feature_dim, **model_parameter_counts(model),
        "cache_construction_seconds": sum(p.get("cache_construction_seconds", 0.0) for p in store._payloads.values()),
        "cache_load_build_verify_seconds": time.perf_counter() - start,
        "online_cache_checks": checks, "rtol": 1e-4, "atol": 1e-5,
    })
    del model
    if args.build_feature_cache:
        return 0
    common = dict(output_dir=args.output_dir, data_root=root, train_records=train, eval_records=test,
                  manifest_metadata=metadata, device=device, workers=args.num_workers,
                  batch_size=BATCH_SIZE, updates_per_segment=UPDATES_PER_SEGMENT, max_segments=30,
                  resume=args.resume, checkpoint_every_updates=args.checkpoint_every_updates,
                  feature_cache=store, head_hidden_dim=args.head_hidden_dim)
    if args.calibrate:
        # Same two-visit accuracy-blind rule and transferred timescales as the existing pilot.
        args.smoke_batch_size, args.smoke_updates_per_segment = BATCH_SIZE, UPDATES_PER_SEGMENT
        path = args.output_dir / "calibration" / "cap_k_calibration_pilot.json"
        engine._run_cap_calibration_pilot(args, seed=DEVELOPMENT_SEEDS[0], data_root=root,
            train_records=train, device=device, output_path=path, feature_cache=store)
        report = json.loads(path.read_text(encoding="utf-8"))
        diagnostic = report["diagnostics"][-1]
        healthy = (all(math.isfinite(float(diagnostic[key])) for key in ("bind_rate", "median_importance", "median_radius", "frozen_fraction"))
                   and 0 < diagnostic["bind_rate"] < .99 and diagnostic["median_radius"] > 0
                   and diagnostic["median_importance"] > 0 and diagnostic["frozen_fraction"] < .99)
        atomic_write_json(args.output_dir / "calibration" / "cap_k_center.json", {
            "experiment_version": EXPERIMENT_VERSION, "manifest_hash": metadata["manifest_hash"],
            "k_center": report["pilot_k"], "healthy": healthy,
            "selection_basis": "accuracy-blind two-domain diagnostics; retain transferred 7e-5 when healthy",
            "health_rule": "finite positive importance/radius, 0 < bind_rate < .99, frozen_fraction < .99",
        })
        print("Calibration healthy; center saved." if healthy else "Degenerate calibration; inspect diagnostics and establish one new center without accuracy.")
        return 0 if healthy else 2
    if args.qualify:
        run_dir = args.output_dir / "qualification" / "adamw" / f"seed_{DEVELOPMENT_SEEDS[0]}"
        result = engine.run_single(method="adamw", seed=DEVELOPMENT_SEEDS[0], config=default_config("adamw"), run_dir=run_dir, **common)
        metrics, _ = dn4il_metrics(result["rows"]["segments"], result["rows"]["fast"])
        gain = fmean(float(row["learning_gain"]) for row in result["rows"]["segments"])
        report = {"experiment_version": EXPERIMENT_VERSION, "manifest_hash": metadata["manifest_hash"],
                  "metrics": metrics, "mean_learning_gain": gain,
                  "qualified": metrics["mean_forgetting"] >= .01 and gain >= .01,
                  "qualification_rule": "predeclared diagnostic: mean final forgetting and mean within-visit gain both >= 0.01 (one percentage point)"}
        atomic_write_json(args.output_dir / "qualification" / "qualification.json", report)
        from analyze_frozen_resnet_head import analyze_runs
        analyze_runs({"adamw": {DEVELOPMENT_SEEDS[0]: {"dir": run_dir, "config": {}, **result["rows"]}}}, args.output_dir / "qualification" / "analysis")
        print(json.dumps(report, indent=2))
        return 0 if report["qualified"] else 2
    if args.tune:
        engine.run_tuning(args, method=args.method, data_root=root, train_records=train,
            eval_records=test, manifest_metadata=metadata, device=device, feature_cache=store)
        return 0
    methods, seeds = engine._resolve_methods_and_seeds(args)
    if args.method is None and not args.all_methods:
        methods = METHODS
    fingerprints = {}
    for method in methods:
        config = default_config(method, cap_k=args.cap_k_center or 7e-5)
        if args.train:
            path = args.output_dir / "selected_configs" / f"{method}.json"
            selected = json.loads(path.read_text(encoding="utf-8"))
            config = selected["config"]
            if (selected.get("experiment_version") != EXPERIMENT_VERSION or selected.get("model") != MODEL_NAME
                or selected.get("manifest_hash") != metadata["manifest_hash"] or selected.get("method") != method
                or selected.get("head_hidden_dim") != args.head_hidden_dim or selected.get("backbone_weights") != BACKBONE_WEIGHTS
                or selected.get("selected_config_hash") != payload_hash(config)):
                raise ValueError(f"incompatible frozen selected config: {path}")
        for seed in seeds:
            kwargs = dict(common)
            if args.smoke:
                kwargs.update(batch_size=args.smoke_batch_size, updates_per_segment=args.smoke_updates_per_segment,
                              max_segments=args.smoke_segments)
            run_dir = args.output_dir / ("smoke" if args.smoke else "train") / method / f"seed_{seed}"
            result = engine.run_single(method=method, seed=seed, config=config, run_dir=run_dir, **kwargs)
            hashes = [row["sample_order_hash"] for row in result["rows"]["segments"]]
            if seed in fingerprints and fingerprints[seed] != hashes:
                raise RuntimeError("method data pairing failed")
            fingerprints[seed] = hashes
    atomic_write_json(args.output_dir / ("smoke" if args.smoke else "train") / "paired_sample_order_hashes.json", fingerprints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
