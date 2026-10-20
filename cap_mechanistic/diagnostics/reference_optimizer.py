"""PyTorch implementation of the CAP-Adam optimizer.

CAP-Adam augments AdamW with a slowly moving parameter anchor and a
boundary-free raw squared-gradient (``g^2``) importance estimate.  See
``docs/METHOD.md`` in the repository root for the optimizer equations.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer


Protection = Literal["box", "spring"]
Retention = Literal["leak", "ratchet", "release"]
DecayMode = Literal["blend", "anchor", "zero"]
ReleaseUpdate = Literal["legacy", "contest_blocked"]
BoxGeometry = Literal["coordinate", "q4_slack"]
MomentumResponse = Literal["none", "conflict_restart"]
LossClosure = Callable[[], Tensor]


def _update_importance_(
    imp: Tensor,
    candidate: Tensor,
    bind: Tensor,
    *,
    retention: Retention,
    release_update: ReleaseUpdate,
    T_m: float,
) -> None:
    """Apply the selected importance-retention update in place."""
    if retention == "leak":
        imp.copy_(candidate)
    elif retention == "ratchet":
        torch.maximum(imp, candidate, out=imp)
    elif release_update == "legacy":
        release_rate = 0.0 if math.isinf(T_m) else 1.0 / T_m
        released = imp * (
            1.0 - bind.to(dtype=imp.dtype) * release_rate
        )
        torch.maximum(released, candidate, out=imp)
    elif math.isinf(T_m):
        torch.maximum(imp, candidate, out=imp)
    else:
        release_rate = 1.0 / T_m
        uncontested = torch.maximum(imp, candidate)
        contested = imp * (1.0 - release_rate)
        imp.copy_(torch.where(bind, contested, uncontested))


class CAPAdam(Optimizer):
    """AdamW with boundary-free anchoring and importance protection.

    Args:
        params: Parameters or parameter-group dictionaries.  A group can set
            ``protect=False`` to exempt its parameters (normally biases and
            normalization parameters) from box/spring protection.  Exempt
            parameters still maintain ``anchor`` and ``imp`` state and still
            receive anchored decay when their ``weight_decay`` is nonzero;
            conventionally their group should also set ``weight_decay=0``.
        lr: Learning rate.
        betas: Adam first- and second-moment decay rates.
        eps: Adam numerical-stability term.  It is also used in the leash
            radius denominator.
        weight_decay: Decoupled weight-decay coefficient.
        amsgrad: Use the AMSGrad second-moment maximum.
        maximize: Maximize the closure objective instead of minimizing it.
        k: Leash length in importance units.  The box radius is
            ``k / (sqrt(imp) + eps)``.
        T_a: Anchor timescale in optimizer steps.  May be ``math.inf`` for a
            fixed anchor.
        T_s: Importance-estimation timescale in optimizer steps.
        T_m: Importance-release timescale under contest.  May be
            ``math.inf`` (which makes release equivalent to a pure ratchet).
        c: Blend constant for anchor-targeted decay.  It must be positive.
            The default is only a starting point; for E5-style runs, calibrate
            it to a representative importance scale or update each layer's
            parameter-group ``c`` to the layer median between steps.
        passive: If true, parameters receive the exact AdamW update: no
            protection and decay toward zero.  CAP anchor/importance state is
            still observed and updated.
        protection: ``"box"`` for hard projection, or ``"spring"`` for the
            normalized quadratic penalty
            ``0.5 * imp * ((theta-anchor)/k)^2``.
        retention: ``"leak"`` for an EMA, ``"ratchet"`` for a non-decreasing
            estimate, or ``"release"`` for contest-triggered relaxation.
        release_update: ``"legacy"`` for the historical release rule, or
            ``"contest_blocked"`` to suppress upward importance
            reconsolidation on contested coordinates.
        box_geometry: ``"coordinate"`` for the ordinary per-coordinate box,
            or ``"q4_slack"`` to let output rows/channels with Q4 pressure at
            most one bypass that same coordinate box.
        momentum_response: ``"none"`` for ordinary Adam momentum, or
            ``"conflict_restart"`` to discard first-moment history when the
            current gradient opposes it on a coordinate bound one step ago.
        decay_mode: Active CAP weight-decay target. ``"blend"`` uses
            ``imp / (imp + c) * anchor``, ``"anchor"`` uses the full anchor,
            and ``"zero"`` matches ordinary AdamW decay toward zero.
        freeze_stats: Parameter-group switch that freezes both ``anchor`` and
            ``imp`` while parameter updates and protection continue.  This is
            intended for E2's iso-importance task-switch comparison.
        track_bind_stats: If true, populate each parameter group's
            ``last_bind_count``, ``last_bind_numel``, and ``last_bind_rate``.
            The local contest mask is computed regardless because release
            retention depends on it.  The persistent ``last_bind`` copy is
            retained when diagnostics are enabled or conflict restart needs
            it as previous-step algorithmic state.
        track_proposal: If true, expose detached ``last_proposal_scale`` state
            containing the absolute decay/Adam proposal before box projection.
            This is an experiment-only diagnostic and does not enter updates.

    The optimizer deliberately does not infer biases or normalization weights
    from tensor shapes.  Put those parameters in a group with
    ``protect=False``; shape-based inference would incorrectly exempt some
    legitimate one-dimensional weights.

    Parameters with ``grad is None`` follow PyTorch optimizer convention and
    are skipped completely.  Synthetic silence experiments that need leak or
    other state dynamics must therefore assign explicit zero-tensor gradients;
    masked zero elements within an otherwise present gradient already work.

    After each step, every group exposes ``last_bind_count``,
    ``last_bind_numel``, and ``last_bind_rate``.  They cover parameters with a
    non-None gradient in that step when tracking is enabled; otherwise the
    counts are zero and the rate is ``None``. Per-parameter pre-projection
    masks are stored as ``optimizer.state[param]["last_bind"]`` when tracking
    is enabled or conflict restart requires them algorithmically.
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        *,
        k: float = 2.0,
        T_a: float = 10_000.0,
        T_s: float = 10_000.0,
        T_m: float = math.inf,
        c: float = 1e-3,
        passive: bool = False,
        protection: Protection = "box",
        retention: Retention = "release",
        release_update: ReleaseUpdate = "legacy",
        box_geometry: BoxGeometry = "coordinate",
        momentum_response: MomentumResponse = "none",
        decay_mode: DecayMode = "blend",
        freeze_stats: bool = False,
        track_bind_stats: bool = True,
        track_proposal: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if k <= 0.0:
            raise ValueError(f"Invalid leash scale k: {k}")
        if T_a < 1.0:
            raise ValueError(f"T_a must be at least one step, got: {T_a}")
        if T_s < 1.0 or not math.isfinite(T_s):
            raise ValueError(f"T_s must be finite and at least one step, got: {T_s}")
        if T_m < 1.0:
            raise ValueError(f"T_m must be at least one step, got: {T_m}")
        if c <= 0.0:
            raise ValueError(f"Invalid decay blend constant c: {c}")
        if protection not in ("box", "spring"):
            raise ValueError(f"Unknown protection policy: {protection!r}")
        if retention not in ("leak", "ratchet", "release"):
            raise ValueError(f"Unknown retention policy: {retention!r}")
        if release_update not in ("legacy", "contest_blocked"):
            raise ValueError(f"Unknown release update: {release_update!r}")
        if box_geometry not in ("coordinate", "q4_slack"):
            raise ValueError(f"Unknown box geometry: {box_geometry!r}")
        if protection != "box" and box_geometry != "coordinate":
            raise ValueError("q4_slack box geometry requires protection='box'")
        if momentum_response not in ("none", "conflict_restart"):
            raise ValueError(f"Unknown momentum response: {momentum_response!r}")
        if momentum_response == "conflict_restart" and (
            passive or protection != "box" or box_geometry != "coordinate"
        ):
            raise ValueError(
                "conflict_restart requires active coordinate-box protection"
            )
        if decay_mode not in ("blend", "anchor", "zero"):
            raise ValueError(f"Unknown decay mode: {decay_mode!r}")
        if not isinstance(track_bind_stats, bool):
            raise TypeError("track_bind_stats must be a bool")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            k=k,
            T_a=T_a,
            T_s=T_s,
            T_m=T_m,
            c=c,
            passive=passive,
            protection=protection,
            retention=retention,
            release_update=release_update,
            box_geometry=box_geometry,
            momentum_response=momentum_response,
            decay_mode=decay_mode,
            freeze_stats=freeze_stats,
            track_bind_stats=track_bind_stats,
            track_proposal=track_proposal,
            protect=True,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group: dict) -> None:
        """Validate values that parameter groups may override."""
        if "anchor_update" in group:
            raise ValueError(
                "anchor_update parameter-group options are unsupported by CAP-v2.5"
            )
        obsolete_modes = {"leash_mode", "evidence_mode"} & group.keys()
        if obsolete_modes:
            raise ValueError(
                "rejected CAP-v2.1 parameter-group option(s): "
                + ", ".join(sorted(obsolete_modes))
            )
        beta1, beta2 = group["betas"]
        if group["lr"] < 0.0:
            raise ValueError(f"Invalid learning rate: {group['lr']}")
        if group["eps"] <= 0.0:
            raise ValueError(f"Invalid epsilon: {group['eps']}")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid betas: {group['betas']}")
        if group["weight_decay"] < 0.0:
            raise ValueError(f"Invalid weight_decay: {group['weight_decay']}")
        if group["k"] <= 0.0:
            raise ValueError(f"Invalid leash scale k: {group['k']}")
        if group["T_a"] < 1.0:
            raise ValueError(f"T_a must be at least one step, got: {group['T_a']}")
        if group["T_s"] < 1.0 or not math.isfinite(group["T_s"]):
            raise ValueError(
                f"T_s must be finite and at least one step, got: {group['T_s']}"
            )
        if group["T_m"] < 1.0:
            raise ValueError(f"T_m must be at least one step, got: {group['T_m']}")
        if group["c"] <= 0.0:
            raise ValueError(f"Invalid decay blend constant c: {group['c']}")
        if group["protection"] not in ("box", "spring"):
            raise ValueError(f"Unknown protection policy: {group['protection']!r}")
        if group["retention"] not in ("leak", "ratchet", "release"):
            raise ValueError(f"Unknown retention policy: {group['retention']!r}")
        if group["release_update"] not in ("legacy", "contest_blocked"):
            raise ValueError(f"Unknown release update: {group['release_update']!r}")
        if group["box_geometry"] not in ("coordinate", "q4_slack"):
            raise ValueError(f"Unknown box geometry: {group['box_geometry']!r}")
        if group["protection"] != "box" and group["box_geometry"] != "coordinate":
            raise ValueError("q4_slack box geometry requires protection='box'")
        if group["momentum_response"] not in ("none", "conflict_restart"):
            raise ValueError(
                f"Unknown momentum response: {group['momentum_response']!r}"
            )
        if (
            group["momentum_response"] == "conflict_restart"
            and bool(group["protect"])
            and (
                group["passive"]
                or group["protection"] != "box"
                or group["box_geometry"] != "coordinate"
            )
        ):
            raise ValueError(
                "conflict_restart requires active coordinate-box protection"
            )
        if group["decay_mode"] not in ("blend", "anchor", "zero"):
            raise ValueError(f"Unknown decay mode: {group['decay_mode']!r}")
        if not isinstance(group["track_bind_stats"], bool):
            raise TypeError("track_bind_stats must be a bool")

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state while preserving Boolean diagnostics.

        PyTorch casts most optimizer-state tensors to the parameter dtype
        during ``Optimizer.load_state_dict``.  That is appropriate for Adam
        moments, anchors, and importance, but not for the Boolean
        ``last_bind`` mask.  Normalize it after the base restore so resumed
        release-invariant checks can safely use Boolean operations.
        """
        compatible_state = dict(state_dict)
        compatible_groups = []
        for group_index, saved_group in enumerate(state_dict["param_groups"]):
            group_copy = dict(saved_group)
            old_leash = group_copy.pop("leash_mode", "legacy")
            old_evidence = group_copy.pop("evidence_mode", "legacy")
            if old_leash != "legacy" or old_evidence != "legacy":
                raise RuntimeError(
                    "checkpoint was created with a rejected CAP-v2.1 "
                    "selective-leash or separated-evidence mechanism"
                )
            saved_geometry = group_copy.get("box_geometry", "coordinate")
            if group_index < len(self.param_groups):
                current_geometry = self.param_groups[group_index]["box_geometry"]
                if saved_geometry != current_geometry:
                    raise RuntimeError(
                        "checkpoint box_geometry mismatch: "
                        f"saved {saved_geometry!r}, current {current_geometry!r}"
                    )
            group_copy["box_geometry"] = saved_geometry
            saved_momentum_response = group_copy.get("momentum_response", "none")
            if group_index < len(self.param_groups):
                current_momentum_response = self.param_groups[group_index][
                    "momentum_response"
                ]
                if saved_momentum_response != current_momentum_response:
                    raise RuntimeError(
                        "checkpoint momentum_response mismatch: "
                        f"saved {saved_momentum_response!r}, "
                        f"current {current_momentum_response!r}"
                    )
            group_copy["momentum_response"] = saved_momentum_response
            anchor_update = group_copy.pop("anchor_update", "moving")
            if anchor_update != "moving":
                raise ValueError(
                    "importance-gated anchor checkpoints are unsupported by CAP-v2.5"
                )
            if group_copy.get("release_update") == "conflict_gated":
                group_copy["release_update"] = "contest_blocked"
            compatible_groups.append(group_copy)
        compatible_state["param_groups"] = compatible_groups
        super().load_state_dict(compatible_state)
        for group in self.param_groups:
            # Checkpoints created before the E4 protocol preconditions used the
            # current blend decay and always-on bind diagnostics implicitly.
            group.setdefault("decay_mode", "blend")
            group.setdefault("track_bind_stats", True)
            group.setdefault("release_update", "legacy")
            group.setdefault("box_geometry", "coordinate")
            group.setdefault("momentum_response", "none")
            self._validate_group(group)
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                needs_bind_state = bool(group["track_bind_stats"]) or (
                    group["momentum_response"] == "conflict_restart"
                    and bool(group["protect"])
                    and not bool(group["passive"])
                )
                if needs_bind_state:
                    if "last_bind" not in state:
                        if group["momentum_response"] == "conflict_restart":
                            raise RuntimeError(
                                "conflict_restart checkpoint lacks last_bind state"
                            )
                        state["last_bind"] = torch.zeros_like(
                            parameter,
                            dtype=torch.bool,
                            memory_format=torch.preserve_format,
                        )
                    elif state["last_bind"].shape != parameter.shape:
                        raise RuntimeError(
                            "checkpoint last_bind shape does not match parameter"
                        )
                    else:
                        state["last_bind"] = state["last_bind"].to(
                            device=parameter.device, dtype=torch.bool
                        )
                elif "last_bind" in state:
                    del state["last_bind"]

    @torch.no_grad()
    def step(self, closure: Optional[LossClosure] = None) -> Optional[Tensor]:
        """Perform one optimization step and return the closure loss, if any."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._validate_group(group)
            group["last_bind_count"] = 0
            group["last_bind_numel"] = 0
            group["last_bind_rate"] = (
                0.0 if group["track_bind_stats"] else None
            )
            group["last_q4_group_count"] = 0
            group["last_q4_coordinate_contested_groups"] = 0
            group["last_q4_slack_accepted_groups"] = 0
            group["last_q4_overloaded_groups"] = 0
            group["last_previous_bind_count"] = 0
            group["last_momentum_conflict_count"] = 0
            group["last_momentum_restart_count"] = 0
            group["last_rebind_count"] = 0
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            k = group["k"]
            anchor_rate = 0.0 if math.isinf(group["T_a"]) else 1.0 / group["T_a"]
            imp_rate = 1.0 / group["T_s"]
            passive = group["passive"]
            protected = bool(group["protect"]) and not passive

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.is_complex():
                    raise RuntimeError("CAPAdam does not support complex parameters")
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("CAPAdam does not support sparse gradients")
                effective_grad = -grad if group["maximize"] else grad

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    state["anchor"] = parameter.detach().clone(
                        memory_format=torch.preserve_format
                    )
                    state["imp"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    needs_bind_state = bool(group["track_bind_stats"]) or (
                        protected
                        and group["momentum_response"] == "conflict_restart"
                    )
                    if needs_bind_state:
                        state["last_bind"] = torch.zeros_like(
                            parameter,
                            dtype=torch.bool,
                            memory_format=torch.preserve_format,
                        )

                # Backward-compatible with checkpoints saved before bind
                # diagnostics were added.
                needs_bind_state = bool(group["track_bind_stats"]) or (
                    protected and group["momentum_response"] == "conflict_restart"
                )
                if needs_bind_state and "last_bind" not in state:
                    state["last_bind"] = torch.zeros_like(
                        parameter, dtype=torch.bool, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step_number = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                anchor = state["anchor"]
                imp = state["imp"]

                previous_bind = state.get("last_bind")
                momentum_gradient_conflict = None
                if protected and previous_bind is not None:
                    momentum_gradient_conflict = previous_bind & (
                        exp_avg * effective_grad < 0
                    )
                    previous_bind_count = int(previous_bind.count_nonzero().item())
                    conflict_count = int(
                        momentum_gradient_conflict.count_nonzero().item()
                    )
                    group["last_previous_bind_count"] += previous_bind_count
                    group["last_momentum_conflict_count"] += conflict_count
                    if group["momentum_response"] == "conflict_restart":
                        exp_avg.masked_fill_(momentum_gradient_conflict, 0.0)
                        group["last_momentum_restart_count"] += conflict_count

                exp_avg.mul_(beta1).add_(effective_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    effective_grad, effective_grad, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1**step_number
                bias_correction2 = 1.0 - beta2**step_number
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    second_moment = max_exp_avg_sq
                else:
                    second_moment = exp_avg_sq
                denom = second_moment.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                # Evaluate all forces at theta_t.  Cloning is needed only for
                # the spring because its force is applied after AdamW's two
                # in-place update operations.
                spring_force = None
                if protected and group["protection"] == "spring":
                    spring_force = (parameter - anchor).mul(imp).div(k * k)

                # E4b can opt in to observing the unconstrained update
                # pressure.  This detached snapshot is diagnostic only: it
                # never enters an optimizer formula and is captured before
                # decay/Adam update, as required by the E4 protocol.
                theta_before = None
                if group.get("track_proposal", False):
                    theta_before = parameter.detach().clone(
                        memory_format=torch.preserve_format
                    )

                if passive:
                    # This is exactly AdamW's decoupled decay toward zero.
                    parameter.mul_(1.0 - lr * wd)
                elif wd != 0.0:
                    decay_mode = group["decay_mode"]
                    if decay_mode == "zero":
                        parameter.mul_(1.0 - lr * wd)
                    elif decay_mode == "anchor":
                        parameter.add_(parameter - anchor, alpha=-lr * wd)
                    else:
                        blend = imp / (imp + group["c"])
                        target = blend * anchor
                        parameter.add_(parameter - target, alpha=-lr * wd)

                # Keep the same operation ordering as PyTorch's single-tensor
                # AdamW path.
                parameter.addcdiv_(
                    exp_avg, denom, value=-(lr / bias_correction1)
                )
                if spring_force is not None:
                    parameter.add_(spring_force, alpha=-lr)

                if theta_before is not None:
                    proposal = state.get("last_proposal_scale")
                    if proposal is None:
                        proposal = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                        state["last_proposal_scale"] = proposal
                    proposal.copy_((parameter - theta_before).abs())

                # Contest is observed against the same box for both active
                # protection forms.  Passive/exempt groups never contest, so
                # passive tracking cannot accidentally trigger release.
                radius = k / (imp.sqrt() + eps)
                lower = anchor - radius
                upper = anchor + radius
                coord_exceed = (parameter < lower) | (parameter > upper)
                overloaded = None
                if (
                    protected
                    and group["protection"] == "box"
                    and group["box_geometry"] == "q4_slack"
                    and parameter.ndim >= 2
                ):
                    z = (parameter - anchor) / radius
                    z_groups = z.reshape(z.shape[0], -1)
                    q4 = z_groups.abs().pow(4).mean(dim=1).pow(0.25)
                    group_overloaded = q4 > 1.0
                    broadcast_shape = (parameter.shape[0],) + (1,) * (
                        parameter.ndim - 1
                    )
                    overloaded = group_overloaded.reshape(broadcast_shape)
                    coordinate_contested = coord_exceed.reshape(
                        parameter.shape[0], -1
                    ).any(dim=1)
                    group["last_q4_group_count"] += int(group_overloaded.numel())
                    group["last_q4_coordinate_contested_groups"] += int(
                        coordinate_contested.count_nonzero().item()
                    )
                    group["last_q4_slack_accepted_groups"] += int(
                        (coordinate_contested & ~group_overloaded)
                        .count_nonzero()
                        .item()
                    )
                    group["last_q4_overloaded_groups"] += int(
                        group_overloaded.count_nonzero().item()
                    )
                    bind = coord_exceed & overloaded
                elif protected:
                    bind = coord_exceed
                else:
                    bind = torch.zeros_like(parameter, dtype=torch.bool)
                if protected and previous_bind is not None:
                    rebind = bind & previous_bind
                    group["last_rebind_count"] += int(
                        rebind.count_nonzero().item()
                    )
                if group["track_bind_stats"]:
                    group["last_bind_count"] += int(bind.count_nonzero().item())
                    group["last_bind_numel"] += bind.numel()
                if needs_bind_state:
                    state["last_bind"].copy_(bind)

                # Experiment-only warm-up switch: keep contest observation,
                # release, anchor, importance, and counters active while
                # suppressing only the hard projection.  The key is absent in
                # ordinary runs, so the default optimizer path is unchanged.
                if (
                    protected
                    and group["protection"] == "box"
                    and not bool(group.get("disable_box_clamp", False))
                ):
                    if overloaded is None:
                        parameter.clamp_(min=lower, max=upper)
                    else:
                        clamped = torch.maximum(torch.minimum(parameter, upper), lower)
                        parameter.copy_(torch.where(overloaded, clamped, parameter))

                if not group["freeze_stats"]:
                    if anchor_rate != 0.0:
                        anchor.lerp_(parameter, anchor_rate)

                    raw_squared_grad = effective_grad.square()
                    candidate = imp.lerp(raw_squared_grad, imp_rate)
                    _update_importance_(
                        imp,
                        candidate,
                        bind,
                        retention=group["retention"],
                        release_update=group["release_update"],
                        T_m=group["T_m"],
                    )

            if group["last_bind_numel"]:
                group["last_bind_rate"] = (
                    group["last_bind_count"] / group["last_bind_numel"]
                )

        return loss


# Conventional CamelCase spelling for callers that prefer it.
CapAdam = CAPAdam


class RawImportanceTracker:
    """External raw-``g^2`` tracker retained for historical E3 analysis.

    Instantiate it before training and call :meth:`update` immediately after
    ``optimizer.step()`` and before gradients are cleared.  It mirrors each
    It mirrors each CAPAdam group's ``T_s``, retention policy, contest mask,
    and ``freeze_stats`` setting.  Production CAPAdam now also uses ``g^2``,
    so the two states should match under the same update schedule.  Values are
    available in ``tracker.state[param]``.
    """

    def __init__(self, optimizer: CAPAdam) -> None:
        if not isinstance(optimizer, CAPAdam):
            raise TypeError("RawImportanceTracker requires a CAPAdam optimizer")
        self.optimizer = optimizer
        self.state: Dict[Tensor, Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        """Update raw importance from gradients still on the parameters."""
        for group in self.optimizer.param_groups:
            if group["freeze_stats"]:
                continue
            imp_rate = 1.0 / group["T_s"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "RawImportanceTracker does not support sparse gradients"
                    )
                raw_imp = self.state.get(parameter)
                if raw_imp is None:
                    raw_imp = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    self.state[parameter] = raw_imp

                candidate = raw_imp.lerp(grad.square(), imp_rate)
                retention = group["retention"]
                if retention == "release":
                    bind = self.optimizer.state[parameter]["last_bind"]
                else:
                    bind = torch.zeros_like(parameter, dtype=torch.bool)
                _update_importance_(
                    raw_imp,
                    candidate,
                    bind,
                    retention=retention,
                    release_update=group["release_update"],
                    T_m=group["T_m"],
                )

    def state_dict(self) -> dict:
        """Return tracker state in optimizer parameter-group order."""
        values: List[List[Optional[Tensor]]] = []
        for group in self.optimizer.param_groups:
            values.append(
                [
                    None
                    if parameter not in self.state
                    else self.state[parameter].clone()
                    for parameter in group["params"]
                ]
            )
        return {"imp_raw": values}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        values = state_dict["imp_raw"]
        if len(values) != len(self.optimizer.param_groups):
            raise ValueError("Raw tracker state has a different number of groups")
        restored: Dict[Tensor, Tensor] = {}
        for group, group_values in zip(self.optimizer.param_groups, values):
            if len(group_values) != len(group["params"]):
                raise ValueError("Raw tracker state has a different group size")
            for parameter, value in zip(group["params"], group_values):
                if value is not None:
                    if value.shape != parameter.shape:
                        raise ValueError(
                            "Raw tracker tensor shape does not match parameter"
                        )
                    restored[parameter] = value.to(
                        device=parameter.device, dtype=parameter.dtype
                    ).clone()
        self.state = restored


__all__ = ["CAPAdam", "CapAdam", "RawImportanceTracker"]
