"""Read-only proposed-update diagnostics around the frozen CAP implementation."""
import math
import torch
from cap_mechanistic_cifar.cifar_mech.optim.cap_final import CAPAdam


class InstrumentedCAP(CAPAdam):
    measure = False

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("This diagnostic runner requires explicit gradients")
        snapshots = []
        if self.measure:
            for group in self.param_groups:
                for p in group["params"]:
                    if not group["protect"] or p.grad is None:
                        continue
                    s = self.state[p]
                    b1, b2 = group["betas"]
                    t = s["step"] + 1
                    m = s["exp_avg"].clone().mul_(b1).add_(p.grad, alpha=1-b1)
                    v = s["exp_avg_sq"].clone().mul_(b2).addcmul_(p.grad, p.grad, value=1-b2)
                    denom = v.sqrt().div_(math.sqrt(1-b2**t)).add_(group["eps"])
                    proposal = p.clone().addcdiv_(m, denom, value=-group["lr"]/(1-b1**t))
                    radius = group["k"]/(s["imp"].sqrt()+group["eps"])
                    expected = proposal.clamp(min=s["anchor"]-radius, max=s["anchor"]+radius) if group["box_enabled"] else proposal
                    offset = (p-s["anchor"]).abs()
                    snapshots.append((p, p.clone(), proposal, expected, {
                        "layer": group["param_name"], "coordinates": p.numel(),
                        "radius_mean": radius.mean().item(),
                        "importance_mean": s["imp"].mean().item(),
                        "anchor_distance_rms": offset.square().mean().sqrt().item(),
                        "anchor_distance_over_radius_mean": (offset/radius).mean().item(),
                        "near_edge_fraction": (offset/radius >= .9).float().mean().item(),
                    }))
        result = super().step()
        self.measurements = []
        for p, before, proposal, expected, row in snapshots:
            if not torch.equal(p, expected):
                raise AssertionError("Diagnostic proposal disagrees with frozen CAP arithmetic")
            proposed = (proposal-before).norm().item()
            removed = (proposal-p).norm().item()
            row.update(proposed_norm=proposed, applied_norm=(p-before).norm().item(),
                       removed_norm=removed, removed_over_proposed=removed/max(proposed, 1e-30),
                       bind_rate=self.state[p]["last_bind"].float().mean().item())
            self.measurements.append(row)
        return result
