"""PyTorch runtime checks for the separate corrected-UPGD experiment."""
import copy
import json

import pytest
import torch
from torch import nn

from dn4il_upgd_corrected import ConsistentGlobalUPGD
from dn4il_retune_protocol import VERSION, BASE_VERSION, FINAL_SEEDS, center
from experiment_dn4il_retune import build_retune_method


@pytest.mark.parametrize("beta", [0.0, .9999, .99999])
def test_upgd_bias_normalization_matches_closed_form_from_first_step(beta):
    parameter = nn.Parameter(torch.tensor([1., 2., 3.]))
    grad = torch.tensor([-.5, .1, 0.])
    original = parameter.detach().clone()
    optimizer = ConsistentGlobalUPGD([("weight", parameter)], lr=.01, sigma=0, weight_decay=0, beta_utility=beta)
    parameter.grad = grad.clone()
    optimizer.step()
    utility = -original * grad
    gate = torch.sigmoid(-utility / utility.max())
    torch.testing.assert_close(parameter, original - .01 * grad * gate)
    assert gate[0] == pytest.approx(.26894142)
    assert optimizer.diagnostics()["upgd_mean_gate"] == pytest.approx(float(gate.mean()))
    assert optimizer.diagnostics()["upgd_protected_fraction"] == pytest.approx(1/3)
    assert optimizer.diagnostics()["upgd_perturbation_norm"] == 0


@pytest.mark.parametrize("initial,gradient", [(0., 1.), (1., 1.), (1., 0.)])
def test_upgd_zero_and_nonpositive_utility_are_finite(initial, gradient):
    parameter = nn.Parameter(torch.tensor([initial]))
    optimizer = ConsistentGlobalUPGD([("weight", parameter)], lr=.01, sigma=0, weight_decay=0)
    parameter.grad = torch.tensor([gradient])
    optimizer.step()
    gate = .5 if initial * gradient == 0 else float(torch.sigmoid(torch.tensor(1.)))
    assert parameter.item() == pytest.approx(initial - .01 * gradient * gate)


def test_upgd_noise_and_utility_state_resume_exactly():
    p = nn.Parameter(torch.tensor([1., 2., 3.]))
    opt = ConsistentGlobalUPGD([("weight", p)], **center("upgd"))
    torch.manual_seed(42)
    def step(parameter, optimizer):
        parameter.grad = parameter.detach().sin()
        optimizer.step()
    step(p, opt)
    value, state, rng = p.detach().clone(), copy.deepcopy(opt.state_dict()), torch.get_rng_state()
    for _ in range(3):
        step(p, opt)
    q = nn.Parameter(value)
    other = ConsistentGlobalUPGD([("weight", q)], **center("upgd"))
    other.load_state_dict(state)
    torch.set_rng_state(rng)
    for _ in range(3):
        step(q, other)
    torch.testing.assert_close(p, q, atol=0, rtol=0)
    torch.testing.assert_close(opt.state[p]["avg_utility"], other.state[q]["avg_utility"], atol=0, rtol=0)
    assert opt.diagnostics() == other.diagnostics()


def test_corrected_factory_is_separate_from_original():
    from miniimagenet_scale_common import build_method
    model = nn.Linear(3, 2)
    assert isinstance(build_retune_method("upgd", model, center("upgd")).optimizer, ConsistentGlobalUPGD)
    assert not isinstance(build_method("upgd", model, center("upgd")).optimizer, ConsistentGlobalUPGD)
    with pytest.raises(ValueError):
        build_retune_method("adamw", model, {"lr": .001})


def test_new_shared_runner_identity_and_interrupted_resume(tmp_path, monkeypatch):
    import experiment_miniimagenet_resnet as engine
    from tests.test_dn4il_frozen_head import tiny_model, tiny_store, assert_nested_equal
    torch.set_num_threads(1)
    monkeypatch.setattr(engine, "build_frozen_resnet50_head", tiny_model)
    cache, train, test = tiny_store(tmp_path)
    kwargs = dict(method="upgd", seed=31415, config=center("upgd"), output_dir=tmp_path,
        data_root=tmp_path, train_records=train, eval_records=test, manifest_metadata={"manifest_hash": "fixture"},
        device=torch.device("cpu"), workers=0, batch_size=2, updates_per_segment=3, max_segments=2,
        resume=False, checkpoint_every_updates=1, feature_cache=cache, method_factory=build_retune_method,
        study_protocol={"experiment_version": VERSION, "protocol_hash": "fixture"})
    full = engine.run_single(run_dir=tmp_path / "full", **kwargs)
    save = engine._save_checkpoint
    def interrupt(*args, **kw):
        save(*args, **kw)
        if kw["next_segment"] == 1 and kw["next_update"] == 1:
            raise InterruptedError()
    monkeypatch.setattr(engine, "_save_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        engine.run_single(run_dir=tmp_path / "resume", **kwargs)
    monkeypatch.setattr(engine, "_save_checkpoint", save)
    resumed = engine.run_single(run_dir=tmp_path / "resume", **{**kwargs, "resume": True})
    a = engine._torch_load(tmp_path / "full/checkpoint_final.pt")
    b = engine._torch_load(tmp_path / "resume/checkpoint_final.pt")
    assert a["identity"]["experiment_version"] == VERSION
    assert_nested_equal(a["model"], b["model"])
    assert_nested_equal(a["optimizer"], b["optimizer"])
    assert full["rows"]["segments"] == resumed["rows"]["segments"]
    assert "upgd_mean_gate" in full["rows"]["diagnostics"][0]


def test_mixed_analysis_reads_only_new_cap_upgd_and_old_adamw_ewc(tmp_path, monkeypatch):
    from analyze_frozen_resnet_head import load_runs, analyze_runs
    from miniimagenet_scale_common import atomic_write_json, atomic_write_csv
    from dn4il_dataset import DOMAINS
    original, new = tmp_path / "old", tmp_path / "new"
    roots = {"adamw": original, "ewc": original, "cap": new, "upgd": new}
    versions = {m: BASE_VERSION if m in ("adamw", "ewc") else VERSION for m in roots}
    study = {"protocol_hash": "fixture", "code_hashes": {"optimizer": "fixture"}}
    for method, root in roots.items():
        if method in ("cap", "upgd"):
            atomic_write_json(root / "selected_configs" / f"{method}.json", {
                "protocol_hash": "fixture", "study_protocol": study, "config": center(method)})
        for seed in FINAL_SEEDS:
            path = root / "train" / method / f"seed_{seed}"
            config = {"experiment_version": versions[method], "manifest_hash": "same", "schedule_hash": str(seed),
                      "full_evaluation_bank_hash": "same", "head_initialization_hash": str(seed)}
            if method in ("cap", "upgd"):
                config.update(study_protocol=study, config=center(method))
            atomic_write_json(path / "run_config.json", config)
            segments = [{"segment": s, "cycle": (s-1)//6+1, "domain_name": DOMAINS[(s-1)%6],
                         "previous_visit_segment": "" if s <= 6 else s-6, "revisit_gap": "" if s <= 6 else 5,
                         "pre_visit_accuracy": .4, "mean_online_accuracy": .6, "sample_order_hash": f"{seed}-{s}"} for s in range(1,31)]
            evaluation = [{"checkpoint_segment": s, "evaluated_domain": d, "accuracy": .4} for s in range(1,31) for d in DOMAINS]
            online = [{"global_update": s, "update_seconds": .001, "peak_gpu_memory_bytes": 0} for s in range(1,6001)]
            for name, rows in (("stream_segments.csv", segments), ("domain_eval.csv", evaluation),
                               ("online_steps.csv", online), ("optimizer_diagnostics.csv", [{"optimizer_state_bytes": 1}])):
                atomic_write_csv(path / name, rows, tuple(rows[0]))
            (path / "checkpoint_final.pt").write_bytes(b"size fixture")
    loaded = load_runs(new, method_roots=roots, expected_versions=versions)
    assert loaded["adamw"][42]["dir"].is_relative_to(original)
    assert loaded["cap"][42]["dir"].is_relative_to(new)
    assert not (new / "train/adamw").exists()
    import analyze_dn4il_retune as combined
    monkeypatch.setattr(combined, "freeze_protocol", lambda *_: {
        "protocol_hash": "fixture", "disclosure": "Reused baselines; new CAP/UPGD only.", "sources": {}})
    output = combined.analyze(original, new)
    assert "UPGD (corrected)" in (output / "analysis_report.md").read_text(encoding="utf-8")
    assert not (original / "analysis").exists()
    provenance = json.loads((output / "source_provenance.json").read_text())
    assert provenance["methods"]["adamw"]["source_role"] == "reused original baseline"
    path = new / "train/upgd/seed_42/run_config.json"
    config = json.loads(path.read_text())
    config["config"]["lr"] *= 2
    atomic_write_json(path, config)
    with pytest.raises(ValueError, match="selected retune configuration"):
        combined.analyze(original, new)
    config["head_initialization_hash"] = "different"
    atomic_write_json(path, config)
    with pytest.raises(ValueError, match="unpaired"):
        load_runs(new, method_roots=roots, expected_versions=versions)
