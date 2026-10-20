import torch
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from ..data import stratified_indices
from ..streams import permutation
from ..constants import STREAMS

def empirical_fisher(model, data, device, count=200, seed=510000, microbatch=16):
    """True per-example squared loss gradients, stratified within every domain."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    fisher = {name: torch.zeros_like(p) for name, p in params.items() if p.ndim >= 2}
    def loss_fn(p, x, y):
        return F.cross_entropy(functional_call(model, (p, buffers), (x.unsqueeze(0),)), y.unsqueeze(0))
    grad_fn = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    was_training = model.training
    model.eval()
    try:
        for domain in range(STREAMS["pool_size"]):
            indices = stratified_indices(data.test_y, count, seed+domain)
            perm = permutation("b", domain)
            for start in range(0, count, microbatch):
                idx = indices[start:start+microbatch]
                grads = grad_fn(params, data.test_x[idx][:, perm].to(device), data.test_y[idx].to(device))
                for name in fisher:
                    fisher[name].add_(grads[name].detach().square().sum(0)/(STREAMS["pool_size"]*count))
    finally:
        model.train(was_training)
    return fisher
