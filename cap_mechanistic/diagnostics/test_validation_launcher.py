"""Targeted checks for the alternative admission launcher (no scientific runs)."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validated_mechanisms as launcher
from cap_mech.checkpointing import capture


def test_fresh_and_checkpoint_pair():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(77)
    raw = torch.randint(0, 256, (513, 28, 28), dtype=torch.uint8, generator=gen)
    labels = torch.arange(513)%10
    dense = raw.flatten(1).float().div_(255).sub_(.1307).div_(.3081)
    cls = launcher.reference_class(launcher.HERE/"reference_optimizer.py")
    model_cls = launcher.historical_model_class()
    result = launcher.paired(raw, labels, dense, cls, model_cls, 42, "b", "cpu", 12)
    assert result["exact_match"] and result["sampler_checked_against_dataloader"]
    model = launcher.MLP300()
    optimizer = launcher.CAPAdam(launcher.parameter_groups(model))
    sampler = launcher.IndexSampler(513, 42, 128)
    indices = sampler.next_indices()
    launcher.F.cross_entropy(model(dense[indices]), labels[indices]).backward()
    optimizer.step()
    checkpoint = capture(model, optimizer, sampler, source_version=launcher.SOURCE,
                         baseline_config_sha256=launcher.digest(launcher.CAP),
                         within_task_update=0, stream_metadata={"smoke": False}, seed=42, stream="b", task=50)
    assert launcher.paired(raw, labels, dense, cls, model_cls, 42, "b", "cpu", 16, checkpoint)["exact_match"]
    checkpoint["baseline_config_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="Invalid checkpoint"):
        launcher.paired(raw, labels, dense, cls, model_cls, 42, "b", "cpu", 1, checkpoint)


def test_changed_source_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "source_version", lambda: "changed")
    with pytest.raises(RuntimeError, match="core changed"):
        launcher.provenance(tmp_path, "cpu", tmp_path)


def test_failed_validation_cannot_launch(monkeypatch, tmp_path):
    def fail(args):
        raise RuntimeError("deliberate validation failure")
    monkeypatch.setattr(launcher, "validate", fail)
    monkeypatch.setattr(sys, "argv", ["launcher", "--output-dir", str(tmp_path), "--device", "cpu"])
    with pytest.raises(RuntimeError, match="deliberate validation failure"):
        launcher.main()
    assert not (tmp_path/"VALIDATION_POLICY.md").exists()
