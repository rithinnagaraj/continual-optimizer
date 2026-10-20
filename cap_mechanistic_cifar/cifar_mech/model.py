from torch import nn
from . import protocol

class MLP300(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 300)
        self.linear2 = nn.Linear(300, 300)
        self.linear3 = nn.Linear(300, 100)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        return self.linear3(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a,b in ((3,32),(32,64),(64,128)):
            layers.extend([nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(2048,256),nn.ReLU(),nn.Linear(256,100))

    def forward(self,x):
        return self.classifier(self.features(x.reshape(-1,3,32,32)))

def make_model():
    return MLP300() if protocol.ARCHITECTURE == 'mlp' else SmallCNN()

def parameter_groups(model):
    return [{"params": [p], "param_name": n, "protect": p.ndim >= 2}
            for n, p in model.named_parameters()]
