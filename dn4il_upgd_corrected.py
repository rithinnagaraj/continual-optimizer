"""Explicit DN4IL retune variant; the pinned upstream UPGD file stays intact."""
from __future__ import annotations

import math
import torch

from dn4il_retune_protocol import UPGD_VARIANT


class ConsistentGlobalUPGD(torch.optim.Optimizer):
    """First-order global UPGD with consistent utility bias normalization.

    u_hat = EMA(-theta * gradient) / (1 - beta**step)
    gate = sigmoid(-u_hat / max_positive(u_hat))
    theta <- (1-lr*wd)*theta - lr*gate*(gradient + sigma*noise)

    A zero/nonpositive maximum falls back to the largest absolute utility,
    floored at 1e-12. This avoids zero division and sign reversal. No adaptive
    gradient moments or momentum are added. Diagnostics are persisted in groups.
    """
    variant = UPGD_VARIANT

    def __init__(self, named_parameters, lr=.01, weight_decay=1e-4, beta_utility=.9999, sigma=.01):
        if not all(math.isfinite(v) for v in (lr, weight_decay, beta_utility, sigma)):
            raise ValueError("UPGD settings must be finite")
        if lr <= 0 or weight_decay < 0 or sigma < 0 or not 0 <= beta_utility < 1 or lr * weight_decay >= 1:
            raise ValueError("invalid UPGD settings")
        names, parameters = zip(*named_parameters)
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay,
                                         beta_utility=beta_utility, sigma=sigma, names=names))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        active = []
        for group in self.param_groups:
            for name, parameter in zip(group["names"], group["params"]):
                if "gate" in name or parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse or not torch.isfinite(grad).all():
                    raise FloatingPointError("UPGD requires finite dense gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(parameter)
                state["step"] += 1
                beta = group["beta_utility"]
                state["avg_utility"].mul_(beta).addcmul_(parameter, grad, value=-(1-beta))
                corrected = state["avg_utility"] / (1-beta ** state["step"])
                if not torch.isfinite(corrected).all():
                    raise FloatingPointError("UPGD utility became nonfinite")
                active.append((group, parameter, corrected))
        if not active:
            return loss
        maximum = torch.stack([u.max() for _, _, u in active]).max()
        max_abs = torch.stack([u.abs().max() for _, _, u in active]).max()
        scale = torch.where(maximum > 0, maximum, max_abs).clamp_min(1e-12)
        device = active[0][1].device
        stats = {id(g): torch.zeros(4, device=device) for g in self.param_groups}
        counts = {id(g): 0 for g in self.param_groups}
        for group, parameter, utility in active:
            gate = torch.sigmoid(-utility / scale)
            noise = torch.randn_like(parameter) * group["sigma"]
            perturbation = -group["lr"] * gate * noise
            update = -group["lr"] * gate * (parameter.grad + noise)
            full_update = update - group["lr"] * group["weight_decay"] * parameter
            stats[id(group)] += torch.stack((gate.sum(), (gate < .5).sum(),
                                             full_update.square().sum(), perturbation.square().sum()))
            counts[id(group)] += parameter.numel()
            parameter.mul_(1-group["lr"] * group["weight_decay"]).add_(update)
        for group in self.param_groups:
            count = max(counts[id(group)], 1)
            gate_sum, protected, update_sq, noise_sq = stats[id(group)].tolist()
            group["upgd_mean_gate"] = gate_sum / count
            group["upgd_protected_fraction"] = protected / count
            group["upgd_update_norm"] = math.sqrt(update_sq)
            group["upgd_perturbation_norm"] = math.sqrt(noise_sq)
        return loss

    def diagnostics(self):
        # This experiment supplies all named head parameters in one global group.
        return {key: self.param_groups[0].get(key, 0.0) for key in (
            "upgd_mean_gate", "upgd_protected_fraction", "upgd_update_norm", "upgd_perturbation_norm")}
