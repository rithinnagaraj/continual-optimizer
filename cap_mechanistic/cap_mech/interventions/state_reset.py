import torch

@torch.no_grad()
def reset(optimizer, kind):
    from . import protected
    if kind not in ("recenter_only", "importance_reset", "full_cap_state_reset"):
        raise ValueError(f"Unknown reset: {kind}")
    for _, p, s in protected(optimizer):
        if kind in ("recenter_only", "full_cap_state_reset"):
            s["anchor"].copy_(p)
        if kind in ("importance_reset", "full_cap_state_reset"):
            s["imp"].zero_()
