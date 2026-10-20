import torch
from cap_mech.checkpointing import capture, restore, atomic_save, load
from cap_mech.streams import IndexSampler
from cap_mech.model import MLP300, parameter_groups
from cap_mech.optim import CAPAdam

def test_identical_counterfactual_continuations(pair, tmp_path):
    model, opt = pair
    sampler = IndexSampler(25, 42, 8)
    x, y = torch.randn(25, 784), torch.arange(25)%10
    def step(m, o, s):
        idx = s.next_indices()
        o.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(m(x[idx]), y[idx]).backward()
        o.step()
    step(model, opt, sampler)
    path = tmp_path/"checkpoint.pt"
    atomic_save(path, capture(model, opt, sampler))
    other = MLP300()
    other_opt = CAPAdam(parameter_groups(other))
    other_sampler = IndexSampler(25, 42, 8)
    restore(load(path), other, other_opt, other_sampler)
    for _ in range(10):
        step(model, opt, sampler)
        step(other, other_opt, other_sampler)
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.equal(p, q)
        for key in opt.state[p]:
            a, b = opt.state[p][key], other_opt.state[q][key]
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b

def test_journal_trims_uncommitted_rows(tmp_path):
    from cap_mech.records import Journal
    j = Journal(tmp_path)
    j.add("online", {"step": 1})
    counts = dict(j.counts)
    j.add("online", {"step": 2})
    (tmp_path/"online.jsonl").open("a").write('{"partial":')
    j = Journal(tmp_path, counts)
    assert j.rows("online") == [{"step": 1}]
