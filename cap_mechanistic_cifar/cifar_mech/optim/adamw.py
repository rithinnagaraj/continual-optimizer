"""Ordinary AdamW control, with the reference zero weight decay."""
from torch.optim import AdamW

def adamw(params, lr=0.0033333333333333335):
    return AdamW(params, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
