import torch
from .constants import STREAMS

class Data:
    def __init__(self, root, smoke=False):
        if smoke:
            gen = torch.Generator().manual_seed(9876)
            self.train_x = torch.randn(32, 784, generator=gen)
            self.train_y = torch.arange(32) % 10
            self.test_x = torch.randn(20, 784, generator=gen)
            self.test_y = torch.arange(20) % 10
            self.test_indices = torch.arange(20)
        else:
            from torchvision.datasets import MNIST
            train = MNIST(root, train=True, download=True)
            test = MNIST(root, train=False, download=True)
            self.train_x = train.data.flatten(1).float().div_(255).sub_(.1307).div_(.3081)
            self.train_y = train.targets
            self.test_indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(999))[:2000]
            self.test_x = test.data[self.test_indices].flatten(1).float().div_(255).sub_(.1307).div_(.3081)
            self.test_y = test.targets[self.test_indices]

    def batch(self, indices, perm, device):
        return self.train_x[indices][:, perm].to(device), self.train_y[indices].to(device)

def stratified_indices(labels, count, seed):
    gen = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    result = []
    for i, c in enumerate(classes):
        candidates = (labels == c).nonzero().flatten()
        n = count//len(classes) + (i < count % len(classes))
        if len(candidates) < n:
            raise ValueError("Insufficient examples for stratification")
        result.append(candidates[torch.randperm(len(candidates), generator=gen)[:n]])
    return torch.cat(result)
