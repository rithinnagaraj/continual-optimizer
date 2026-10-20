"""Final contest-blocked Release; no legacy optimizer dependencies."""
import math
import torch
from torch.optim import Optimizer
from ..constants import CAP

class CAPAdam(Optimizer):
    def __init__(self, params, **kwargs):
        defaults = dict(CAP, protect=True, box_enabled=True)
        defaults.update(kwargs)
        if (defaults["importance"] != "raw_g2" or
            defaults["protection"] != "coordinate_box" or
            defaults["release_update"] != "contest_blocked" or defaults["weight_decay"] != 0):
            raise ValueError("Only the frozen raw-g2 contest-blocked, zero-decay variant is supported")
        if any(defaults[n] <= 0 for n in ("lr", "k", "eps")) or any(defaults[n] < 1 for n in ("T_a", "T_s", "T_m")):
            raise ValueError("Invalid CAP rates/timescales")
        if any(not 0 <= b < 1 for b in defaults["betas"]):
            raise ValueError("Invalid Adam betas")
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                s = self.state[p]
                s.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                if group["protect"]:
                    s.update(anchor=p.detach().clone(), imp=torch.zeros_like(p),
                             last_bind=torch.zeros_like(p, dtype=torch.bool))
        self.last_bind_rate = 0.
        self.last_bound_importance_decay = 0.

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        counts, decays, count = [], [], 0
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, s = p.grad, self.state[p]
                if g.is_sparse:
                    raise RuntimeError("Sparse gradients are unsupported")
                s["step"] += 1
                t = s["step"]
                s["exp_avg"].mul_(b1).add_(g, alpha=1-b1)
                s["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1-b2)
                denom = s["exp_avg_sq"].sqrt().div_(math.sqrt(1-b2**t)).add_(group["eps"])
                p.addcdiv_(s["exp_avg"], denom, value=-group["lr"]/(1-b1**t))
                if not group["protect"]:
                    continue
                a, imp = s["anchor"], s["imp"]
                r = group["k"]/(imp.sqrt()+group["eps"])
                lo, hi = a-r, a+r
                bind = (p < lo) | (p > hi)
                s["last_bind"].copy_(bind)
                if group["box_enabled"]:
                    p.clamp_(min=lo, max=hi)
                a.lerp_(p, 1/group["T_a"])
                # lerp preserves the arithmetic ordering of the final benchmark.
                candidate = imp.lerp(g.square(), 1/group["T_s"])
                contested = imp if math.isinf(group["T_m"]) else imp*(1-1/group["T_m"])
                decays.append(torch.where(bind, imp-contested, 0.).sum())
                imp.copy_(torch.where(bind, contested, torch.maximum(imp, candidate)))
                counts.append(bind.sum())
                count += p.numel()
        if counts:
            bound = torch.stack(counts).sum().item()
            self.last_bind_rate = bound/count
            self.last_bound_importance_decay = torch.stack(decays).sum().item()/max(1, bound)
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for s in self.state.values():
            if "last_bind" in s:
                s["last_bind"] = s["last_bind"].bool()
