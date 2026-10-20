import torch
from torch.utils.data import DataLoader, TensorDataset
from cap_mech.streams import IndexSampler, permutation, schedule, old_tasks

def test_exact_streams():
    for task in (1, 20, 200):
        assert torch.equal(permutation("a", task), torch.randperm(784, generator=torch.Generator().manual_seed(100000+task)))
    for domain in range(40):
        assert torch.equal(permutation("b", domain), torch.randperm(784, generator=torch.Generator().manual_seed(200001+domain)))
    gen = torch.Generator().manual_seed(555)
    assert torch.equal(schedule(), torch.cat([torch.randperm(40, generator=gen) for _ in range(5)]))
    for block in schedule().reshape(5, 40):
        assert torch.equal(block.sort().values, torch.arange(40))

def test_sampler_matches_reference_loader_across_epochs_and_short_batches():
    sampler = IndexSampler(23, 42, 7, persistent_workers=False)
    loader = DataLoader(TensorDataset(torch.arange(23)), shuffle=True, batch_size=7,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    for _ in range(4):
        for (indices,) in loader:
            assert torch.equal(indices, sampler.next_indices())

def test_sampler_matches_persistent_worker_reference():
    sampler = IndexSampler(23, 42, 7, persistent_workers=True)
    loader = DataLoader(TensorDataset(torch.arange(23)), shuffle=True, batch_size=7,
                        generator=torch.Generator().manual_seed(42), num_workers=1,
                        persistent_workers=True)
    try:
        for _ in range(4):
            for (indices,) in loader:
                assert torch.equal(indices, sampler.next_indices())
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

def test_sampler_resume():
    a, b = IndexSampler(23, 42, 7), IndexSampler(23, 99, 7)
    for _ in range(7):
        a.next_indices()
    b.load_state_dict(a.state_dict())
    for _ in range(20):
        assert torch.equal(a.next_indices(), b.next_indices())

def test_probe_tasks():
    assert old_tasks(20) == [19, 18, 17, 16, 15, 1, 3, 6, 8, 10]
    for b in (20, 50, 100, 150):
        assert len(old_tasks(b)) == len(set(old_tasks(b))) == 10
        assert all(1 <= t < b for t in old_tasks(b))
