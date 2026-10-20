"""CAP-Adam: the paper's final contest-blocked Release rule (Section 3.4).

Matrix weights and convolution kernels are protected by default; one-dimensional
bias/normalization tensors receive ordinary Adam. Historical experiment variants
live in cap_adam_experimental.py and must be selected explicitly.
"""
from __future__ import annotations

import math
import torch
from torch.optim import Optimizer


class CAPAdam(Optimizer):
    """Adam with coordinate boxes, moving anchors and contest-blocked Release.

    ``params`` accepts an iterable of tensors or parameter-group dictionaries.
    Protection defaults to tensors with at least two dimensions. Explicit
    ``protect=False`` groups can exempt other tensors. Biases should not be put
    in explicit ``protect=True`` groups. Only zero weight decay is supported by
    this final paper implementation. ``box_enabled=False`` counts crossings
    against the same pre-update box while disabling only the projection.
    """

    def __init__(self, params, lr=1 / 300, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, *, k=7e-5, T_a=2000.0, T_s=2000.0,
                 T_m=1_000_000.0, box_enabled=True):
        items = list(params)
        if not items:
            raise ValueError("CAPAdam requires parameters")
        groups = []
        for item in items:
            group = dict(item) if isinstance(item, dict) else {"params": [item]}
            groups.append(group)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        k=k, T_a=T_a, T_s=T_s, T_m=T_m,
                        box_enabled=box_enabled, release_update="contest_blocked")
        super().__init__(groups, defaults)
        self.last_bind_rate = 0.0
        self.last_bound_importance_decay = 0.0

    @staticmethod
    def _validate_group(group):
        if group["weight_decay"] != 0:
            raise ValueError("The paper's final CAPAdam uses zero weight decay")
        if group.get("release_update") != "contest_blocked":
            raise ValueError("CAPAdam implements only contest-blocked Release")
        if any(not math.isfinite(group[n]) or group[n] <= 0 for n in ("lr", "k", "eps")):
            raise ValueError("lr, k and eps must be finite and positive")
        if any(not math.isfinite(group[n]) or group[n] < 1 for n in ("T_a", "T_s")):
            raise ValueError("T_a and T_s must be finite and at least one")
        if math.isnan(group["T_m"]) or group["T_m"] < 1:
            raise ValueError("T_m must be at least one or infinity")
        if len(group["betas"]) != 2 or any(not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("Adam betas must lie in [0, 1)")
        if not isinstance(group["box_enabled"], bool):
            raise TypeError("box_enabled must be a bool")

    def add_param_group(self, param_group):
        """Initialize added tensors exactly as at construction, including biases.

        Optimizer.__init__ also calls this method for the initial groups.
        Each tensor retains an independent protection choice and initial anchor.
        """
        if not isinstance(param_group, dict):
            raise TypeError("Parameter groups must be dictionaries")
        tensors = param_group["params"]
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        elif isinstance(tensors, set):
            raise TypeError("Parameters must have a deterministic order")
        else:
            tensors = list(tensors)
        names = param_group.get("param_names")
        if names is not None and len(names) != len(tensors):
            raise ValueError("param_names must match the parameter count")
        candidates = []
        existing = {id(p) for g in self.param_groups for p in g["params"]}
        for index, parameter in enumerate(tensors):
            if not isinstance(parameter, torch.Tensor):
                raise TypeError("Optimizer parameters must be tensors")
            if id(parameter) in existing:
                raise ValueError("A parameter occurs in more than one optimizer group")
            existing.add(id(parameter))
            group = {**self.defaults, **param_group, "params": [parameter],
                     "protect": bool(param_group.get("protect", parameter.ndim >= 2))}
            if names is not None:
                group["param_names"] = [names[index]]
            self._validate_group(group)
            if not parameter.is_leaf and not parameter.retains_grad:
                raise ValueError("Cannot optimize a non-leaf tensor without retained gradients")
            candidates.append(group)
        for group in candidates:
            super().add_param_group(group)
            parameter = group["params"][0]
            state = self.state[parameter]
            state.update(step=0, exp_avg=torch.zeros_like(parameter),
                         exp_avg_sq=torch.zeros_like(parameter))
            if group["protect"]:
                state.update(anchor=parameter.detach().clone(), imp=torch.zeros_like(parameter),
                             last_bind=torch.zeros_like(parameter, dtype=torch.bool))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        counts, decays, protected_count = [], [], 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad, state = parameter.grad, self.state[parameter]
                if grad.is_sparse or parameter.is_complex():
                    raise RuntimeError("CAPAdam supports dense real gradients only")
                state["step"] += 1
                t = state["step"]
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(1 - beta2**t)).add_(group["eps"])
                parameter.addcdiv_(state["exp_avg"], denominator, value=-group["lr"] / (1 - beta1**t))
                if not group["protect"]:
                    continue
                anchor, importance = state["anchor"], state["imp"]
                radius = group["k"] / (importance.sqrt() + group["eps"])
                lower, upper = anchor - radius, anchor + radius
                bind = (parameter < lower) | (parameter > upper)
                state["last_bind"].copy_(bind)
                if group["box_enabled"]:
                    parameter.clamp_(min=lower, max=upper)
                anchor.lerp_(parameter, 1 / group["T_a"])
                candidate = importance.lerp(grad.square(), 1 / group["T_s"])
                contested = importance if math.isinf(group["T_m"]) else importance * (1 - 1 / group["T_m"])
                decays.append(torch.where(bind, importance - contested, 0.0).sum())
                importance.copy_(torch.where(bind, contested, torch.maximum(importance, candidate)))
                counts.append(bind.sum())
                protected_count += parameter.numel()
        self.last_bind_rate = 0.0
        self.last_bound_importance_decay = 0.0
        if counts:
            bound = torch.stack(counts).sum().item()
            self.last_bind_rate = bound / protected_count
            self.last_bound_importance_decay = torch.stack(decays).sum().item() / max(1, bound)
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            self._validate_group(group)
        for state in self.state.values():
            if "last_bind" in state:
                state["last_bind"] = state["last_bind"].bool()


CapAdam = CAPAdam
__all__ = ["CAPAdam", "CapAdam"]
