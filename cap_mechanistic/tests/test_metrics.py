import torch
import pytest
from cap_mech.metrics import task_metrics, end_retention
from cap_mech.probes.functional_damage import antithetic_damage
from cap_mech.probes.state_features import state_features

def test_threshold_and_auc():
    a = [0.]*200+[1.]*4800
    m = task_metrics(a)
    assert m["early_auc"] == .8
    assert m["plasticity"] == .96
    assert m["steps_to_85_percent"] in (370, 371)  # FP summation at the exact .85 boundary
    assert task_metrics([0.]*5000)["steps_to_85_percent"] == 5000
    assert not task_metrics([0.]*5000)["threshold_reached"]

def test_end_retention_excludes_current():
    rows = [{"task": 200, "domain": i, "is_current": i == 0, "accuracy": 0. if i == 0 else .9} for i in range(40)]
    assert end_retention(rows) == pytest.approx(.9)
    with pytest.raises(ValueError):
        end_retention(rows[:-1])

def test_antithetic_damage_restores_on_error():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    original = p.detach().clone()
    damage, draws = antithetic_damage(p, torch.tensor([0, 1]), lambda: float(p.square().sum()), 5., 42)
    assert damage > 0
    assert torch.equal(p, original)
    def fail():
        raise ValueError("evaluation failed")
    with pytest.raises(ValueError):
        antithetic_damage(p, torch.tensor([0]), fail, 5., 42)
    assert torch.equal(p, original)

def test_features_ignore_biases(pair):
    model, opt = pair
    before = state_features(opt)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.fill_(1e6)
    assert before == state_features(opt)
