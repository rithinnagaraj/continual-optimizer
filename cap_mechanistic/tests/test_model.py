import torch
from cap_mech.model import MLP300

def test_architecture_and_initialization():
    torch.manual_seed(42)
    model = MLP300()
    torch.manual_seed(42)
    reference = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(784, 300), torch.nn.ReLU(),
                                    torch.nn.Linear(300, 300), torch.nn.ReLU(), torch.nn.Linear(300, 10))
    x = torch.randn(3, 1, 28, 28)
    assert torch.equal(model(x), reference(x))
    assert sum(p.numel() for p in model.parameters()) == 328810
