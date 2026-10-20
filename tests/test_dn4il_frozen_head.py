from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import experiment_miniimagenet_resnet as engine
from dn4il_dataset import (DOMAINS, EXPERIMENT_VERSION, DN4ILDataset, DN4ILAdapter,
                          validate_dn4il, sample_segment)
from experiment_frozen_resnet_head import parse_args, verify_cache, require_qualification
from frozen_head_metrics import continual_metrics
from miniimagenet_scale_common import (FrozenResNet50Head, default_config, METHODS,
    scale_metrics, sample_order_hash, build_frozen_resnet50_head, model_parameter_counts)


@pytest.fixture
def official_fixture(tmp_path):
    root, splits = tmp_path / "DomainNet", tmp_path / "DN4IL-dataset"
    splits.mkdir()
    for domain in DOMAINS:
        for split in ("train", "test"):
            lines = []
            for label in range(100):
                relative = f"{domain}/class_{label}/{split}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 9), (label, 100, 200)).save(path)
                lines.append(f"{relative} {label}\n")
            (splits / f"{domain}_{split}.txt").write_text("".join(lines))
    return root, splits


def test_official_parsing_validation_metadata(official_fixture, tmp_path):
    root, splits = official_fixture
    _, train, test, metadata = validate_dn4il(root, splits, tmp_path / "manifest")
    assert len(train) == len(test) == 600
    assert len(metadata["split_file_hashes"]) == 12
    assert metadata["num_classes"] == 100
    assert metadata["train_test_leakage"] is False
    sample = train[0]
    assert sample["domain"] == "real" and sample["split"] == "train"
    assert sample["label"] == 0 and sample["source_index"] == 0
    assert Path(sample["absolute_path"]).is_absolute()
    assert sample["relative_path"] == "real/class_0/train.png"
    data = DN4ILDataset(root, splits / "real_test.txt", "real", "test")
    assert data[0][0].shape == (3, 224, 224)
    assert (tmp_path / "manifest" / "dn4il_manifest.json").exists()


@pytest.mark.parametrize("case", ["label", "noninteger", "image", "split", "domain", "leakage", "mapping", "class_space", "escape", "wrong_domain", "duplicate"])
def test_bad_official_annotations_fail(official_fixture, case):
    root, splits = official_fixture
    path = splits / "sketch_test.txt"
    text = path.read_text()
    if case == "label":
        path.write_text(text.replace("test.png 0\n", "test.png 100\n"))
    elif case == "noninteger":
        path.write_text(text.replace("test.png 0\n", "test.png 0.0\n"))
    elif case == "image":
        (root / "sketch/class_0/test.png").unlink()
    elif case == "split":
        path.unlink()
    elif case == "domain":
        (root / "sketch").rename(root / "absent")
    elif case == "leakage":
        path.write_text(text.replace("class_0/test.png", "class_0/train.png"))
    elif case == "mapping":
        # Preserve all 100 integer labels but exchange two semantic assignments.
        path.write_text(text.replace("test.png 0\n", "test.png X\n").replace("test.png 1\n", "test.png 0\n").replace("test.png X\n", "test.png 1\n"))
    elif case == "class_space":
        path.write_text("\n".join(text.splitlines()[1:]))
    elif case == "escape":
        path.write_text(text.replace("sketch/class_0/test.png", "sketch/../class_0/test.png"))
    elif case == "wrong_domain":
        path.write_text(text.replace("sketch/class_0/test.png", "real/class_0/test.png"))
    elif case == "duplicate":
        path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_dn4il(root, splits)


def test_recurrent_schedule_exact_sampling_and_tamper_detection(tmp_path):
    adapter = DN4ILAdapter({})
    records = [{"source_id": f"{d}/{i}", "domain": d, "class_index": i % 100} for d in DOMAINS for i in range(137)]
    schedule = engine._write_schedule(tmp_path, 31415, records, adapter=adapter)
    rows = schedule["segments"]
    assert len(rows) == 30
    assert Counter(r["domain"] for r in rows) == {d: 5 for d in DOMAINS}
    for cycle in range(5):
        assert {r["domain"] for r in rows[cycle * 6:(cycle + 1) * 6]} == set(DOMAINS)
        if cycle:
            assert rows[cycle * 6 - 1]["domain"] != rows[cycle * 6]["domain"]
    previous = {}
    for row in rows:
        domain, segment = row["domain"], row["segment"]
        assert row["previous_visit_segment"] == previous.get(domain)
        assert row["revisit_gap"] == (segment - previous[domain] - 1 if domain in previous else None)
        previous[domain] = segment
    domain = rows[0]["domain"]
    samples = sample_segment(records, 31415, 0, domain)
    assert len(samples) == 200 * 128
    assert all(len(samples[i:i + 128]) == 128 for i in range(0, len(samples), 128))
    assert len({r["source_id"] for r in samples[:137]}) == 137
    assert len({r["source_id"] for r in samples[137:274]}) == 137
    assert [r["source_id"] for r in samples[:137]] != [r["source_id"] for r in samples[137:274]]
    for method in METHODS:
        assert sample_order_hash(sample_segment(records, 31415, 0, domain)) == rows[0]["sample_order_hash"]
        assert engine._write_schedule(tmp_path, 31415, records, adapter=adapter) == schedule
    path = tmp_path / "schedules" / "seed_31415.json"
    modified = json.loads(path.read_text()); modified["segments"][0]["sample_order_hash"] = "bad"
    path.write_text(json.dumps(modified))
    with pytest.raises(RuntimeError, match="incompatible"):
        engine._write_schedule(tmp_path, 31415, records, adapter=adapter)


def tiny_model(seed, hidden_dim=300):
    torch.manual_seed(seed)
    return FrozenResNet50Head(nn.Sequential(nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten()),
                             feature_dim=3, hidden_dim=hidden_dim, num_classes=100, weights_name="test-only")


def tiny_store(tmp_path):
    train, test = [], []
    hashes = {}
    for d in DOMAINS:
        for split in ("train", "test"):
            hashes[f"{d}_{split}.txt"] = f"test-hash-{d}-{split}"
            for i in range(3):
                path = tmp_path / f"{d}_{split}_{i}.png"
                Image.new("RGB", (9, 10), (i * 50, 60, 80)).save(path)
                (train if split == "train" else test).append({"source_id": path.name, "source_path": path.name,
                    "absolute_path": str(path), "domain": d, "class_index": i})
    adapter = DN4ILAdapter({"split_file_hashes": hashes, "dataset_version": "test-only"})
    cache = engine.FeatureCacheStore(tmp_path / "cache", rebuild=False, data_root=tmp_path, manifest_hash="fixture",
        train_records=train, eval_records=test, workers=0, batch_size=2, adapter=adapter)
    return cache, train, test


def test_cache_equivalence_determinism_and_metadata_rejection(tmp_path):
    store, train, test = tiny_store(tmp_path)
    model = tiny_model(10)
    checks = verify_cache(store, model, torch.device("cpu"))
    assert len(checks) == 12
    domain = 0
    before = store._load("train", domain, model, torch.device("cpu"))
    store._payloads.clear(); store.rebuild = True; store._rebuilt.clear()
    after = store._load("train", domain, model, torch.device("cpu"))
    torch.testing.assert_close(before["features"], after["features"], rtol=0, atol=0)
    assert after["features"].shape == (3, model.feature_dim)
    assert after["metadata"]["preprocessing_hash"]
    assert after["metadata"]["split_file_hash"] == "test-hash-real-train"
    store.rebuild = False
    for key in ("preprocessing_hash", "backbone_weights", "split_file_hash", "feature_dim", "dataset_version"):
        bad = {**after, "metadata": {**after["metadata"], key: "changed"}}
        engine._atomic_torch_save(store._path("train", domain), bad)
        store._payloads.clear()
        with pytest.raises(RuntimeError, match="incompatible"):
            store._load("train", domain, model, torch.device("cpu"))
    for key, value in (("labels", torch.tensor([99, 1, 2])), ("source_ids", ["bad", "bad", "bad"]),
                       ("features", torch.zeros(3, 4))):
        engine._atomic_torch_save(store._path("train", domain), {**after, key: value})
        store._payloads.clear()
        with pytest.raises(RuntimeError):
            store._load("train", domain, model, torch.device("cpu"))


def test_metric_definitions():
    segments = [{"segment": s, "mean_online_accuracy": .2 if s <= 15 else .8,
                 "pre_visit_accuracy": .6, "revisit_gap": None if s == 16 else 3} for s in range(1, 31)]
    segments[15]["pre_visit_accuracy"] = 0
    m = scale_metrics(segments, total_segments=30)
    assert m["P_scale"] == pytest.approx(.8)
    assert m["R_scale"] == pytest.approx(.6)
    assert m["B_scale"] == pytest.approx(.6)
    visits = [{"segment": 1, "domain_name": "real"}, {"segment": 2, "domain_name": "sketch"},
              {"segment": 3, "domain_name": "real"}]
    matrix = [{"checkpoint_segment": s, "evaluated_domain": d, "accuracy": a}
              for s, values in enumerate(((.7, .9), (.8, .6), (.6, .5)), 1)
              for d, a in zip(("real", "sketch"), values)]
    m, _ = continual_metrics(visits, matrix)
    assert m["final_all_domain_accuracy"] == pytest.approx(.55)
    assert m["mean_forgetting"] == pytest.approx(.15)  # unseen sketch .9 excluded
    assert m["backward_transfer"] == pytest.approx(-.1)


def test_cache_verification_replays_original_batches_and_still_rejects_bad_features(tmp_path):
    store, _, _ = tiny_store(tmp_path)

    class BatchShapeSensitiveModel(nn.Module):
        feature_dim = 3

        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def extract_features(self, images):
            # Emulate batch-shape-dependent numerical differences without a GPU.
            self.batch_sizes.append(len(images))
            return images.mean(dim=(2, 3)) + len(images) * .001

    model = BatchShapeSensitiveModel()
    device = torch.device("cpu")
    for domain_id in range(len(DOMAINS)):
        for split in ("train", "eval"):
            store._load(split, domain_id, model, device)
    model.batch_sizes.clear()
    checks = verify_cache(store, model, device)
    assert model.batch_sizes == [2, 1] * 12  # Includes the original short tail.
    assert all(check["checked_features"] == 3 for check in checks)
    assert all(check["max_absolute_error"] == 0 for check in checks)
    # Extraction creates inference tensors; clone before injecting corruption.
    payload = store._payloads[("train", 0)]
    payload["features"] = payload["features"].clone()
    payload["features"][0, 0] += .1
    with pytest.raises(RuntimeError, match="verification failed for real/train"):
        verify_cache(store, model, device)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("interrupt_segment", [0, 1])
def test_interrupted_resume_frozen_scope_and_boundary_semantics(tmp_path, monkeypatch, method, interrupt_segment):
    torch.set_num_threads(1)
    monkeypatch.setattr(engine, "build_frozen_resnet50_head", tiny_model)
    store, train, test = tiny_store(tmp_path)
    original_build = engine.build_method
    boundaries = []

    def build(*args, **kwargs):
        bundle = original_build(*args, **kwargs)
        boundary = bundle.on_boundary
        def record_boundary():
            boundaries.append(len(bundle.optimizer.state))
            boundary()
        bundle.on_boundary = record_boundary
        return bundle
    monkeypatch.setattr(engine, "build_method", build)
    common = dict(method=method, seed=31415, config=default_config(method), output_dir=tmp_path,
                  data_root=tmp_path, train_records=train, eval_records=test, manifest_metadata={"manifest_hash": "fixture"},
                  device=torch.device("cpu"), workers=0, batch_size=2, updates_per_segment=3,
                  max_segments=2, resume=False, checkpoint_every_updates=1, feature_cache=store)
    result = engine.run_single(run_dir=tmp_path / "full", **common)
    assert len(boundaries) == 2
    assert len(result["rows"]["online"]) == 6
    assert len(result["rows"]["fast"]) == 12
    assert all(row["bank_size"] == 3 for row in result["rows"]["fast"])
    save = engine._save_checkpoint
    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if kwargs["next_segment"] == interrupt_segment and kwargs["next_update"] == 1:
            raise InterruptedError("injected interruption after durable checkpoint")
    monkeypatch.setattr(engine, "_save_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        engine.run_single(run_dir=tmp_path / "resume", **common)
    assert len(boundaries) == 2 + interrupt_segment  # No boundary on a partial segment.
    monkeypatch.setattr(engine, "_save_checkpoint", save)
    resumed = engine.run_single(run_dir=tmp_path / "resume", **{**common, "resume": True})
    assert len(boundaries) == 4  # Exactly two more, including the resumed segment.
    uninterrupted = engine._torch_load(tmp_path / "full" / "checkpoint_final.pt")
    interrupted = engine._torch_load(tmp_path / "resume" / "checkpoint_final.pt")
    assert_nested_equal(uninterrupted["model"], interrupted["model"])
    assert_nested_equal(uninterrupted["optimizer"], interrupted["optimizer"])
    for key in ("segments", "fast", "full"):
        assert result["rows"][key] == resumed["rows"][key]
    expected_backbone = tiny_model(31415).backbone.state_dict()
    for name, value in expected_backbone.items():
        assert torch.equal(interrupted["model"][f"backbone.{name}"], value)
    assert all(row["backbone_trainable_params"] == 0 and row["backbone_mode"] == "eval" for row in result["rows"]["diagnostics"])
    assert all(len(group["params"]) <= 4 for group in interrupted["optimizer"]["param_groups"])
    with pytest.raises(RuntimeError, match="configuration changed"):
        engine.run_single(run_dir=tmp_path / "resume", **{**common, "resume": True, "max_segments": 1})


def test_real_resnet50_architecture_counts_without_downloading():
    model = build_frozen_resnet50_head(31415, pretrained=False)
    assert model.feature_dim == 2048
    counts = model_parameter_counts(model)
    assert counts["head_trainable_params"] == 644800
    assert counts["backbone_total_params"] == 23508032
    assert counts["backbone_trainable_params"] == 0


def test_cli_paths_methods_and_gates(tmp_path):
    args = parse_args(["--smoke", "--all-methods"])
    assert args.data_root.name == "DomainNet" and args.data_root.parent.name == "data"
    assert args.dn4il_splits_root.name == "DN4IL-dataset"
    assert METHODS == ("adamw", "cap", "upgd", "ewc")
    with pytest.raises(SystemExit):
        parse_args(["--smoke", "--method", "replay"])
    with pytest.raises(SystemExit):
        parse_args(["--tune", "--method", "cap", "--seeds", "42"])
    with pytest.raises(RuntimeError, match="qualify"):
        require_qualification(tmp_path, "fixture")


def test_tuning_uses_only_two_development_seeds_and_preserves_cap_timescales(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from miniimagenet_scale_common import DEVELOPMENT_SEEDS, tuning_grid
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"metrics": {"P_scale": .6, "R_scale": .5, "B_scale": .5}}
    monkeypatch.setattr(engine, "run_single", run)
    args = SimpleNamespace(cap_k_center=7e-5, output_dir=tmp_path, num_workers=0, resume=True,
                           checkpoint_every_updates=200, head_hidden_dim=300)
    cache = SimpleNamespace(adapter=DN4ILAdapter({}))
    for method in METHODS:
        calls.clear()
        engine.run_tuning(args, method=method, data_root=tmp_path, train_records=[], eval_records=[],
            manifest_metadata={"manifest_hash": "fixture"}, device=torch.device("cpu"), feature_cache=cache)
        size = len(tuning_grid(method))
        assert [r["seed"] for r in calls] == [DEVELOPMENT_SEEDS[0]] * size + [DEVELOPMENT_SEEDS[1]] * 3
        assert all((r["max_segments"], r["updates_per_segment"], r["batch_size"]) == (30, 200, 128) for r in calls)
        if method == "cap":
            assert {r["config"]["k"] for r in calls} == {3.5e-5, 7e-5, 1.4e-4}
            assert all(tuple(r["config"][k] for k in ("T_a", "T_s", "T_m")) == (2000, 2000, 10000) for r in calls)
        selected = json.loads((tmp_path / "selected_configs" / f"{method}.json").read_text())
        assert selected["experiment_version"] == EXPERIMENT_VERSION
        assert selected["config"] == default_config(method)


def test_calibration_is_accuracy_blind_and_follows_seeded_first_two_domains(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(engine, "build_frozen_resnet50_head", tiny_model)
    def forbidden(*args, **kwargs):
        pytest.fail("calibration must never evaluate test accuracy")
    monkeypatch.setattr(engine, "evaluate", forbidden)
    store, train, test = tiny_store(tmp_path)
    args = SimpleNamespace(cap_k_center=None, head_hidden_dim=300, smoke_batch_size=2,
                           smoke_updates_per_segment=3, num_workers=0)
    path = tmp_path / "pilot.json"
    engine._run_cap_calibration_pilot(args, seed=31415, data_root=tmp_path, train_records=train,
                                     device=torch.device("cpu"), output_path=path, feature_cache=store)
    report = json.loads(path.read_text())
    assert report["pilot_domains"] == [r["domain_name"] for r in store.adapter.schedule(31415)["segments"][:2]]
    assert report["pilot_is_accuracy_blind"] is True
    assert all("accuracy" not in k for row in report["diagnostics"] for k in row)


def test_analysis_artifacts_seed_first_scores_and_ewc_label(tmp_path):
    from analyze_frozen_resnet_head import analyze_runs
    from miniimagenet_scale_common import read_csv
    runs = {}
    for method in METHODS:
        runs[method] = {}
        for seed, p, r in ((31415, .8, .4), (27182, .3, .9)):
            schedule = DN4ILAdapter({}).schedule(seed)["segments"]
            segments = [{"segment": row["segment_index"] + 1, "cycle": row["cycle_index"] + 1,
                         "domain_name": row["domain_name"], "previous_visit_segment": row["previous_visit_segment"],
                         "revisit_gap": row["revisit_gap"], "pre_visit_accuracy": r,
                         "mean_online_accuracy": p} for row in schedule]
            evaluation = [{"checkpoint_segment": s, "evaluated_domain": d, "accuracy": .4 + .001 * s}
                          for s in range(1, 31) for d in DOMAINS]
            directory = tmp_path / method / str(seed)
            directory.mkdir(parents=True)
            (directory / "checkpoint_final.pt").write_bytes(b"fixture checkpoint size")
            runs[method][seed] = {"dir": directory, "segments": segments, "fast": evaluation,
                "online": [{"global_update": 1, "update_seconds": .001, "peak_gpu_memory_bytes": 0}],
                "diagnostics": [{"optimizer_state_bytes": 1024}]}
    output = analyze_runs(runs, tmp_path / "analysis")
    summary = read_csv(output / "final_metrics.csv")
    assert {r["method"] for r in summary} == {"CAP", "AdamW", "UPGD", "EWC-oracle"}
    assert all(float(r["B_scale_mean"]) == pytest.approx(.35) for r in summary)
    assert all(float(r["B_scale_sd"]) == pytest.approx(2 ** .5 * .05) for r in summary)
    for name in ("per_seed_metrics.csv", "cycle_plasticity.csv", "cycle_retention.csv", "retention_by_gap.csv",
                 "forgetting_by_domain.csv", "domain_accuracy_matrix.csv", "efficiency.csv",
                 "stability_plasticity_plane.png", "plasticity_by_cycle.png", "retention_by_cycle.png",
                 "retention_by_gap.png", "domain_accuracy_matrix.png", "forgetting_by_domain.png"):
        assert (output / name).stat().st_size > 0
    assert "EWC-oracle" in (output / "analysis_report.md").read_text(encoding="utf-8")
