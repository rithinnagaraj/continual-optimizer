"""Numerical checks of the public paper optimizer; requires a working PyTorch."""
import math
import copy
import pytest
import torch
from cap_adam import CAPAdam


def test_initial_state_and_section_34_defaults():
    weight = torch.nn.Parameter(torch.tensor([[0.3, -0.4]], dtype=torch.float64))
    optimizer = CAPAdam([weight])
    group = optimizer.param_groups[0]
    assert {n: group[n] for n in ("k", "T_a", "T_s", "T_m", "weight_decay", "release_update")} == {
        "k": 7e-5, "T_a": 2000.0, "T_s": 2000.0, "T_m": 1e6,
        "weight_decay": 0.0, "release_update": "contest_blocked"}
    assert torch.equal(optimizer.state[weight]["anchor"], weight)
    assert not optimizer.state[weight]["imp"].any()
    assert optimizer.state[weight]["anchor"].data_ptr() != weight.data_ptr()


def test_bias_uses_plain_adam_and_is_excluded_from_bind_denominator():
    weight = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float64))
    reference = torch.nn.Parameter(bias.detach().clone())
    optimizer = CAPAdam([weight, bias], lr=0.1, k=0.01)
    adam = torch.optim.Adam([reference], lr=0.1, foreach=False)
    optimizer.state[weight]["imp"].fill_(1)
    for gradient in ([1.0, -2.0], [-0.3, 0.7], [0.0, 1.0]):
        weight.grad = torch.ones_like(weight)
        bias.grad = torch.tensor(gradient, dtype=torch.float64)
        reference.grad = bias.grad.clone()
        optimizer.step(); adam.step()
        assert torch.allclose(bias, reference, rtol=1e-14, atol=1e-15)
        assert optimizer.last_bind_rate == 1.0
    assert "anchor" not in optimizer.state[bias]
    assert "imp" not in optimizer.state[bias]


def test_contested_importance_ignores_large_candidate_and_infinity_holds():
    for timescale, expected in ((10.0, 0.9), (math.inf, 1.0)):
        weight = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
        optimizer = CAPAdam([weight], lr=1.0, k=0.01, T_s=2, T_m=timescale)
        optimizer.state[weight]["imp"].fill_(1)
        weight.grad = torch.full_like(weight, 100.0)
        optimizer.step()
        assert optimizer.state[weight]["imp"].item() == expected


def test_box_off_observes_crossings_without_clipping_and_keeps_observers():
    weight = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
    optimizer = CAPAdam([weight], lr=1.0, k=0.01, T_a=2, T_m=10, box_enabled=False)
    optimizer.state[weight]["imp"].fill_(1)
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    assert abs(weight.item()) > 0.9
    assert optimizer.last_bind_rate == 1.0
    assert optimizer.state[weight]["last_bind"].all()
    assert torch.equal(optimizer.state[weight]["anchor"], weight / 2)
    assert optimizer.state[weight]["imp"].item() == 0.9


def test_added_groups_initialize_matrices_and_biases_and_resume():
    initial = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
    optimizer = CAPAdam([initial])
    weight = torch.nn.Parameter(torch.tensor([[.3, -.4]], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
    optimizer.add_param_group({"params": (p for p in (weight, bias)), "lr": .01})
    assert torch.equal(optimizer.state[weight]["anchor"], weight)
    assert not optimizer.state[weight]["imp"].any()
    assert "anchor" not in optimizer.state[bias]
    for parameter in (initial, weight, bias):parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(saved)
    optimizer.step()
    assert optimizer.state[weight]["step"] == 2
    assert optimizer.state[bias]["step"] == 2


def test_added_group_rejects_invalid_settings_before_mutation():
    optimizer = CAPAdam([torch.nn.Parameter(torch.zeros(1, 1))])
    candidate = torch.nn.Parameter(torch.ones(1, 1))
    count = len(optimizer.param_groups)
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [candidate], "weight_decay": .01})
    assert len(optimizer.param_groups) == count
    assert candidate not in optimizer.state


def test_replay_metric_observes_incoming_data_before_mixing():
    from phase1_stream_common import training_step
    model = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():model.weight.copy_(torch.tensor([[1.], [-1.]], dtype=torch.float64))
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    incoming = torch.ones(4, 1, dtype=torch.float64)
    targets = torch.zeros(4, dtype=torch.long)
    hooks = {"batch_mixer": lambda x,y: (x, torch.ones_like(y)), "grad_hook": None}
    loss, accuracy = training_step(model=model, optimizer=optimizer, hooks=hooks,
                                   inputs=incoming, targets=targets)
    assert accuracy == 1.0  # The deliberately wrong replay labels must not enter P.
    assert loss.item() > 1
