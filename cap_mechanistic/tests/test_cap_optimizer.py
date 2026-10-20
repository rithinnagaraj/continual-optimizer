import math
import torch
import pytest
from cap_mech.optim import CAPAdam

@pytest.mark.parametrize("tm", [10., math.inf])
def test_contest_blocked_release_ignores_large_candidate(tm):
    p = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
    opt = CAPAdam([p], lr=1., k=.1, T_a=2., T_s=2., T_m=tm)
    s = opt.state[p]
    s["imp"].copy_(torch.tensor([[4., 0.]], dtype=p.dtype))
    p.grad = torch.tensor([[100., .1]], dtype=p.dtype)
    radius = .1/(s["imp"].sqrt()+1e-8)
    opt.step()
    assert torch.equal(s["last_bind"], torch.tensor([[True, False]]))
    assert p[0, 0].item() == pytest.approx(-radius[0, 0].item(), abs=1e-14)
    assert s["imp"][0, 0].item() == pytest.approx(4*(1-1/tm))
    assert s["imp"][0, 1].item() == pytest.approx(.005)
    assert torch.allclose(s["anchor"], p/2)
    # The historical max(released,candidate) gives ~5002 and fails above.

def test_unbound_ratchet_never_decreases():
    p = torch.nn.Parameter(torch.zeros(1, 1))
    opt = CAPAdam([p], k=10.)
    opt.state[p]["imp"].fill_(4.)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.state[p]["imp"].item() == 4.

def test_bias_exemption_and_adam_equivalence():
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    opt = CAPAdam([{"params": [p], "protect": False}], k=1e-20)
    ref = torch.optim.AdamW([q], lr=opt.defaults["lr"], weight_decay=0., foreach=False)
    for i in range(10):
        p.grad = q.grad = torch.tensor([.2+i, -.1], dtype=p.dtype)
        opt.step()
        ref.step()
        assert torch.allclose(p, q, atol=1e-14, rtol=1e-14)
    assert "imp" not in opt.state[p]

def test_box_off_keeps_contest_observer():
    p = torch.nn.Parameter(torch.zeros(1, 1))
    opt = CAPAdam([p], lr=1., k=.01, box_enabled=False, T_m=10.)
    opt.state[p]["imp"].fill_(4.)
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.item() < -.5
    assert opt.state[p]["last_bind"].item()
    assert opt.state[p]["imp"].item() == pytest.approx(3.6)
