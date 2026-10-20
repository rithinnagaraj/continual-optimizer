import copy
import torch
import pytest
from cap_mech.interventions import apply, protected

@pytest.mark.parametrize("intervention", [
    {"kind": "k", "value": .00014}, {"kind": "shuffle", "seed": 17}, {"kind": "reverse"},
    {"kind": "recenter_only"}, {"kind": "importance_reset"}, {"kind": "full_cap_state_reset"},
    {"kind": "box_off"}, {"kind": "release", "value": "infinity"}])
def test_only_declared_state_changes(pair, intervention):
    model, opt = pair
    for _, p, s in protected(opt):
        s["imp"].copy_(torch.arange(p.numel()).reshape_as(p))
        s["anchor"].add_(.1)
    before = copy.deepcopy(opt.state_dict())
    x = torch.randn(2, 784)
    logits = model(x).detach().clone()
    apply(opt, intervention)
    assert torch.equal(logits, model(x))
    after = opt.state_dict()
    for key, s in before["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(s[name], after["state"][key][name])
        assert s["step"] == after["state"][key]["step"]
        if "imp" in s and intervention["kind"] in ("shuffle", "reverse"):
            assert torch.equal(s["imp"].flatten().sort().values, after["state"][key]["imp"].flatten().sort().values)
            assert torch.equal(s["anchor"], after["state"][key]["anchor"])

def test_reverse_correct_coordinates(pair):
    _, opt = pair
    for _, p, s in protected(opt):
        s["imp"].copy_(torch.arange(p.numel()).reshape_as(p))
    apply(opt, {"kind": "reverse"})
    for _, p, s in protected(opt):
        assert torch.equal(s["imp"].flatten(), torch.arange(p.numel()).flip(0).float())
