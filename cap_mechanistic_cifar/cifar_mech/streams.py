import numpy as np
import torch
from .constants import STREAMS
from . import protocol

def schedule():
    generator = torch.Generator().manual_seed(555)
    return torch.cat([torch.randperm(STREAMS["pool_size"], generator=generator)
                      for _ in range(STREAMS["tasks"]//STREAMS["pool_size"])])

def permutation(stream, task_or_pool):
    seed = (100000 if stream == "a" else 200001) + int(task_or_pool)
    p = protocol.PATCH_SIZE
    grid = 32//p
    order = torch.randperm(grid*grid, generator=torch.Generator().manual_seed(seed))
    # Each output patch receives one intact input patch, identically in RGB.
    indices = torch.arange(3072).reshape(3,grid,p,grid,p).permute(0,1,3,2,4)
    patches = indices.reshape(3,grid*grid,p,p)[:,order]
    return patches.reshape(3,grid,grid,p,p).permute(0,1,3,2,4).reshape(-1)

def task_domain(stream, task):
    return task if stream == "a" else int(schedule()[task-1])

def old_tasks(b):
    recent = [b-i for i in range(1, 6) if b-i >= 1]
    older = [round(x) for x in np.linspace(1, max(1, b-10), 5)]
    return recent + sorted(set(older)-set(recent))

class IndexSampler:
    """Explicit replay of DataLoader(RandomSampler), including worker lifetime.

    The iterator consumes a base seed, then RandomSampler consumes one full
    permutation and one unused permutation on exhaustion. Last short batches
    are retained; epochs cross task boundaries uninterrupted. Persistent workers
    consume the base seed only on the first epoch, matching the final runner's
    num_workers=8/persistent_workers=True default without needing actual workers.
    """
    def __init__(self, size, seed, batch_size=128, persistent_workers=STREAMS["persistent_workers"]):
        self.size, self.batch_size = size, batch_size
        self.persistent_workers = persistent_workers
        self.generator = torch.Generator().manual_seed(seed)
        self.order = None
        self.position, self.epoch = 0, -1

    def next_indices(self):
        if self.order is None or self.position == self.size:
            if self.order is not None:
                torch.randperm(self.size, generator=self.generator)
            if self.order is None or not self.persistent_workers:
                torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            self.order = torch.randperm(self.size, generator=self.generator)
            self.position, self.epoch = 0, self.epoch+1
        end = min(self.position+self.batch_size, self.size)
        result = self.order[self.position:end].clone()
        self.position = end
        return result

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "epoch": self.epoch,
                "position": self.position, "order": self.order,
                "generator": self.generator.get_state(), "persistent_workers": self.persistent_workers}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size or state["persistent_workers"] != self.persistent_workers:
            raise ValueError("Sampler dataset/batch size changed")
        self.epoch, self.position = state["epoch"], state["position"]
        self.order = None if state["order"] is None else state["order"].cpu().clone()
        self.generator.set_state(state["generator"].cpu())
