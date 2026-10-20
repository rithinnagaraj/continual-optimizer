import torch
import pytest
from cap_mech.model import MLP300, parameter_groups
from cap_mech.optim import CAPAdam

torch.set_num_threads(1)

@pytest.fixture
def pair():
    torch.manual_seed(42)
    model = MLP300()
    return model, CAPAdam(parameter_groups(model))
