import torch

@torch.no_grad()
def shuffle(optimizer, seed):
    from . import protected
    generator = torch.Generator().manual_seed(seed)
    for _, _, s in protected(optimizer):
        imp = s["imp"].flatten()
        order = torch.randperm(imp.numel(), generator=generator).to(imp.device)
        imp.copy_(imp[order])

@torch.no_grad()
def reverse(optimizer):
    from . import protected
    for _, _, s in protected(optimizer):
        imp = s["imp"].flatten()
        order = torch.argsort(imp, stable=True)
        original = imp.clone()
        imp[order] = original[order.flip(0)]
