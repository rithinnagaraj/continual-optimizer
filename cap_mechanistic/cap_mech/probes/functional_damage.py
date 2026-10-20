import numpy as np
import torch

@torch.no_grad()
def antithetic_damage(parameter, indices, loss_fn, baseline, seed, draws=3, noise_scale=.1):
    """Restore the exact original weights even if evaluation raises."""
    original = parameter.detach().clone()
    gen = torch.Generator().manual_seed(seed)
    sigma = noise_scale*original.std().item()
    results = []
    try:
        for _ in range(draws):
            noise = torch.zeros_like(parameter).flatten()
            noise[indices] = torch.randn(len(indices), generator=gen).to(parameter.device)*sigma
            noise = noise.view_as(parameter)
            parameter.copy_(original+noise)
            plus = loss_fn()
            parameter.copy_(original-noise)
            minus = loss_fn()
            results.append((plus+minus)/2-baseline)
    finally:
        parameter.copy_(original)
    return float(np.mean(results)), results
