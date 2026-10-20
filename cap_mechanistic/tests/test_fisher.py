from types import SimpleNamespace
import torch
from cap_mech.probes.fisher import empirical_fisher
from cap_mech.streams import permutation

def test_fisher_matches_per_example_gradients_without_mutation():
    model = torch.nn.Linear(784, 10)
    data = SimpleNamespace(test_x=torch.randn(10, 784), test_y=torch.arange(10))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = empirical_fisher(model, data, "cpu", count=10, microbatch=3)
    expected = torch.zeros_like(model.weight)
    for domain in range(40):
        for i in range(10):
            logits = model(data.test_x[i:i+1, permutation("b", domain)])
            loss = torch.nn.functional.cross_entropy(logits, data.test_y[i:i+1])
            gradient = torch.autograd.grad(loss, model.weight)[0]
            expected += gradient.square()/400
    assert torch.allclose(result["weight"], expected, atol=1e-6)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in model.parameters())
