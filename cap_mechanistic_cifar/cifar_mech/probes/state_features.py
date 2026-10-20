import math
import torch
from ..interventions import protected

@torch.no_grad()
def state_features(optimizer, trailing_bind_rate=0., previous_task_accuracy=0.):
    imps, radii, zs, weights = [], [], [], []
    for group, p, s in protected(optimizer):
        r = group["k"]/(s["imp"].sqrt()+group["eps"])
        imps.append(s["imp"].flatten())
        radii.append(r.flatten())
        zs.append(((p-s["anchor"]).abs()/r).flatten())
        weights.append(p.flatten())
    imp, r, z, w = [torch.cat(v) for v in (imps, radii, zs, weights)]
    mi, mr = imp.median().item(), r.median().item()
    return {"median_importance": mi, "median_radius": mr,
            "log_median_importance": math.log(max(mi, 1e-30)),
            "log_median_radius": math.log(max(mr, 1e-30)),
            "mean_normalized_displacement": z.mean().item(),
            "median_normalized_displacement": z.median().item(),
            "boundary_fraction": (z > .9).float().mean().item(),
            "trailing_bind_rate": trailing_bind_rate,
            "parameter_norm": w.norm().item(),
            "previous_task_accuracy": previous_task_accuracy}
