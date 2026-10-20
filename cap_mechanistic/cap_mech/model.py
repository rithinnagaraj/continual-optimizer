from torch import nn

class MLP300(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 300)
        self.linear2 = nn.Linear(300, 300)
        self.linear3 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        return self.linear3(x)

def parameter_groups(model):
    return [{"params": [p], "param_name": n, "protect": p.ndim >= 2}
            for n, p in model.named_parameters()]
