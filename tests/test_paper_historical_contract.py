"""Paper historical equations; these numerical checks require PyTorch."""
import math
import torch
from cap_adam_experimental import CAPAdam, RawImportanceTracker, protection_radius


def test_centered_and_raw_trackers_are_distinct():
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = CAPAdam([p], lr=0, passive=True, T_s=1, retention="leak", importance_evidence="centered")
    raw = RawImportanceTracker(opt)
    p.grad = torch.full_like(p, 2)
    opt.step(); raw.update()
    assert opt.state[p]["imp"].item() == 0
    assert raw.state[p].item() == 4


def test_proximal_spring_uses_implicit_solution_with_frozen_state():
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    opt = CAPAdam([p], lr=.1, betas=(0., 0.), weight_decay=0, k=.5,
                  protection="spring", spring_step="proximal", freeze_stats=True)
    p.grad = torch.zeros_like(p); opt.step()
    s = opt.state[p]
    s["anchor"].zero_(); s["imp"].fill_(2)
    p.grad = torch.ones_like(p); opt.step()
    proposal = 1 - .1 / (1 + 1e-8)
    assert math.isclose(p.item(), proposal / (1 + .1 * 2 / .5**2), rel_tol=1e-14)
    assert s["imp"].item() == 2 and s["anchor"].item() == 0


def test_conflict_release_uses_min_candidate_while_final_ignores_it():
    outputs = []
    for conflict in (False, True):
        p = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
        opt = CAPAdam([p], lr=0, k=.1, weight_decay=0, T_a=math.inf, T_s=2, T_m=2,
                      release_update="contest_blocked", paper_conflict_release=conflict)
        p.grad = torch.zeros_like(p); opt.step()
        opt.state[p]["imp"].fill_(1)
        with torch.no_grad(): p.fill_(1)
        p.grad = torch.zeros_like(p); opt.step()
        outputs.append(opt.state[p]["imp"].item())
    assert outputs == [.5, .25]


def test_importance_gated_anchor_and_selective_radius():
    p = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
    opt = CAPAdam([p], lr=0, weight_decay=0, k=1e6, T_a=2, T_s=1, c=1,
                  retention="leak", paper_anchor_gated=True)
    p.grad = torch.zeros_like(p); opt.step()
    opt.state[p]["imp"].fill_(1)
    with torch.no_grad(): p.fill_(1)
    p.grad = torch.full_like(p, 2); opt.step()
    assert opt.state[p]["anchor"].item() == .25
    group = opt.param_groups[0]
    original = protection_radius(torch.ones_like(p), group)
    group["paper_selective_box"] = True
    wider = protection_radius(torch.ones_like(p), group)
    assert (wider > original).all()


def test_adam_v_ablation_uses_corrected_second_moment_directly():
    p = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
    opt = CAPAdam([p], lr=0, weight_decay=0, importance_evidence="adam_v", retention="ratchet")
    p.grad = torch.full_like(p, 3); opt.step()
    assert torch.allclose(opt.state[p]["imp"], torch.full_like(p, 9))
