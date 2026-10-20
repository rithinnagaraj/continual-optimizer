import torch
from .constants import STREAMS

class Data:
    def __init__(self, root, smoke=False):
        if smoke:
            gen = torch.Generator().manual_seed(9876)
            self.train_x = torch.randn(32, 3072, generator=gen)
            self.train_y = torch.arange(32) % 100
            self.test_x = torch.randn(10, 3072, generator=gen)
            self.test_y = torch.arange(10)
            self.test_indices = torch.arange(10)
        else:
            from torchvision.datasets import CIFAR100
            train = CIFAR100(root, train=True, download=True)
            test = CIFAR100(root, train=False, download=True)
            self.train_x = normalize(torch.from_numpy(train.data))
            self.train_y = torch.tensor(train.targets)
            self.test_indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(999))[:2000]
            self.test_x = normalize(torch.from_numpy(test.data)[self.test_indices])
            self.test_y = torch.tensor(test.targets)[self.test_indices]

    def batch(self, indices, perm, device):
        return self.train_x[indices][:, perm].to(device), self.train_y[indices].to(device)

def normalize(x):
    x = x.permute(0,3,1,2).float().div_(255)
    mean = torch.tensor(STREAMS['normalization_mean']).view(1,3,1,1)
    std = torch.tensor(STREAMS['normalization_std']).view(1,3,1,1)
    return ((x-mean)/std).flatten(1)

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
