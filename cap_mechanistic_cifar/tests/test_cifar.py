import copy
import json
from pathlib import Path
import pytest
import torch
from cifar_mech import protocol
from cifar_mech.streams import permutation, IndexSampler
from cifar_mech.model import make_model, parameter_groups
from cifar_mech.optim import CAPAdam
from cifar_mech.interventions import apply
from cifar_mech.checkpointing import capture, restore, gate_payload, require_gate, digest, source_version, atomic_json, file_hash
from cifar_mech.constants import SEEDS, CAP
from cifar_mech.data import Data
from cifar_mech.probes.fisher import empirical_fisher
from cifar_mech.probes.functional_damage import antithetic_damage

@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    protocol.configure('mlp',4)

@pytest.mark.parametrize('size',[2,4,8,16])
def test_patch_mapping(size):
    protocol.configure('cnn',size)
    indices=permutation('a',1)
    assert torch.equal(indices.sort().values,torch.arange(3072))
    image=indices.reshape(3,32,32)
    assert torch.equal(image[1]-1024,image[0])
    grid=32//size
    expected=torch.randperm(grid*grid,generator=torch.Generator().manual_seed(100001))
    for row in range(grid):
        for col in range(grid):
            source=int(expected[row*grid+col]); sr,sc=divmod(source,grid)
            block=torch.arange(1024).reshape(32,32)[sr*size:(sr+1)*size,sc*size:(sc+1)*size]
            assert torch.equal(image[0,row*size:(row+1)*size,col*size:(col+1)*size],block)
    assert torch.equal(indices,permutation('a',1))
    assert not torch.equal(indices,permutation('a',2))

@pytest.mark.parametrize('architecture',['mlp','cnn'])
def test_models_interventions_resume_and_probes(architecture):
    protocol.configure(architecture,4)
    data=Data('.',True); model=make_model(); opt=CAPAdam(parameter_groups(model)); sampler=IndexSampler(32,42,8)
    def step(m,o,s):
        x,y=data.batch(s.next_indices(),permutation('a',1),'cpu'); o.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(m(x),y).backward();o.step()
    step(model,opt,sampler)
    assert model(data.test_x[:2]).shape==(2,100)
    cp=capture(model,opt,sampler)
    other=make_model(); other_opt=CAPAdam(parameter_groups(other)); other_sampler=IndexSampler(32,42,8)
    restore(cp,other,other_opt,other_sampler)
    step(model,opt,sampler);step(other,other_opt,other_sampler)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters()))
    for intervention in [{'kind':'k','value':.00014},{'kind':'box_off'},{'kind':'recenter_only'},
                         {'kind':'importance_reset'},{'kind':'full_cap_state_reset'},
                         {'kind':'shuffle','seed':77},{'kind':'reverse'}]:
        restore(cp,other,other_opt,other_sampler)
        before=other(data.test_x[:2]).detach().clone(); moments=[s['exp_avg'].clone() for s in other_opt.state.values()]
        old_importance=[s['imp'].flatten().sort().values.clone() for s in other_opt.state.values() if 'imp' in s]
        apply(other_opt,intervention)
        assert torch.equal(before,other(data.test_x[:2]))
        assert all(torch.equal(a,s['exp_avg']) for a,s in zip(moments,other_opt.state.values()))
        if intervention['kind'] in ('shuffle','reverse'):
            assert all(torch.equal(a,s['imp'].flatten().sort().values) for a,s in zip(old_importance,[s for s in other_opt.state.values() if 'imp' in s]))
    fisher=empirical_fisher(model,data,'cpu',count=2)
    assert all(torch.isfinite(v).all() and (v>=0).all() for v in fisher.values())
    p=next(model.parameters()); original=p.detach().clone()
    loss=lambda:float(model(data.test_x[:2]).square().mean().detach())
    damage,_=antithetic_damage(p,torch.arange(3),loss,loss(),42,draws=1)
    assert torch.equal(p,original)

def test_gate_never_uses_mnist_targets_and_rejects_other_model(tmp_path):
    observed={s:{'P_sampled_100step':.02,'R':.01} for s in SEEDS}
    gate=gate_payload(observed,source_version=source_version(),protocol=protocol.specification())
    assert gate['passed'] and 'expected' not in gate['rows'][0]
    assert not gate_payload({42:observed[42]})['passed']
    atomic_json(tmp_path/'reproduction/reproduction_gate.json',gate)
    protocol.configure('cnn',4)
    with pytest.raises(RuntimeError,match='wrong protocol'):
        require_gate(tmp_path)

def test_cli_separates_architectures(capsys,tmp_path):
    from cifar_mech.cli import main
    main(['reference','--model','both','--dry-run','--output-dir',str(tmp_path)])
    out=capsys.readouterr().out
    assert 'mlp' in out and 'cnn' in out and out.count('"executes_training": false')==2

def test_reference_integrity_checks(tmp_path):
    bank=tmp_path/'reproduction/checkpoints'; bank.mkdir(parents=True)
    hashes={}
    for seed in SEEDS:
        for stream,tasks in [('a',[20,50,100,150,200]),('b',[50,100,150,200])]:
            for task in tasks:
                p=bank/f'{stream}_seed{seed}_task{task}.pt';p.write_bytes(b'test-checkpoint')
                hashes[p.name]=file_hash(p)
    gate=gate_payload({s:{'P_sampled_100step':.02,'R':.01} for s in SEEDS},
                      source_version=source_version(),protocol=protocol.specification(),checkpoint_hashes=hashes)
    atomic_json(tmp_path/'reproduction/reproduction_gate.json',gate)
    assert require_gate(tmp_path)
    protocol.configure('mlp',8)
    with pytest.raises(RuntimeError,match='wrong protocol'):require_gate(tmp_path)
    protocol.configure('mlp',4)
    (bank/next(iter(hashes))).write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checkpoint changed'):require_gate(tmp_path)
