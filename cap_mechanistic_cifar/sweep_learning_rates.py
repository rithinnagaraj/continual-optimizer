"""Ten CAP rates per model; one seed/task; train-only validation split; no test-set access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torchvision.datasets import CIFAR100
from torch.nn import functional as F
from cifar_mech import protocol
from cifar_mech.constants import CAP
from cifar_mech.data import normalize
from cifar_mech.model import make_model, parameter_groups
from cifar_mech.optim import CAPAdam
from cifar_mech.streams import IndexSampler, permutation
from cifar_mech.seed import seed_everything
from cifar_mech.checkpointing import source_version, atomic_json, file_hash
from cifar_mech.lr_selection import signature, winner

RATES=[1e-5,3e-5,1e-4,2e-4,3e-4,5e-4,1e-3,2e-3,1/300,1e-2]

@torch.no_grad()
def score(model,x,y,perm,device):
    model.eval(); correct=0; loss=0.
    for start in range(0,len(y),128):
        yy=y[start:start+128].to(device)
        logits=model(x[start:start+128][:,perm].to(device))
        correct+=(logits.argmax(1)==yy).sum().item()
        loss+=F.cross_entropy(logits,yy,reduction='sum').item()
    model.train()
    return correct/len(y),loss/len(y)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('../data'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--patch-size',type=int,choices=(2,4,8,16),default=4)
    p.add_argument('--work-dir',type=Path,default=Path('lr_sweep'))
    p.add_argument('--selection',type=Path,default=Path(__file__).with_name('selected_lrs.json'))
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    torch.set_num_threads(1)
    seed_everything(args.seed)
    # No CIFAR test file is opened. Require an already-downloaded dataset.
    dataset=CIFAR100(args.data_dir,train=True,download=False)
    y=torch.tensor(dataset.targets)
    gen=torch.Generator().manual_seed(777)
    train_idx=[];val_idx=[]
    for label in range(100):
        idx=(y==label).nonzero().flatten()
        assert len(idx)==500,'Expected standard CIFAR-100 training split'
        idx=idx[torch.randperm(len(idx),generator=gen)]
        val_idx.append(idx[:50]);train_idx.append(idx[50:])
    train_idx=torch.cat(train_idx).sort().values;val_idx=torch.cat(val_idx).sort().values
    raw=torch.from_numpy(dataset.data)
    tx=normalize(raw[train_idx]);ty=y[train_idx]
    vx=normalize(raw[val_idx]);vy=y[val_idx]
    del raw,dataset
    config={'source_version':source_version(),'script_sha256':file_hash(__file__),
            'cap_without_lr':{k:v for k,v in CAP.items() if k!='lr'},
            'seed':args.seed,'steps':args.steps,'patch_size':args.patch_size,'learning_rates':RATES,
            'task':'Stream A task 1','split_seed':777,'train_examples':45000,'validation_examples':5000,
            'validation_indices_sha256':hashlib.sha256(val_idx.numpy().tobytes()).hexdigest(),
            'train_file_sha256':file_hash(args.data_dir/'cifar-100-python/train'),
            'selection_rule':'highest final validation accuracy; ties: lowest loss, then lowest LR',
            'python':sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'device':args.device,'gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None}
    config_hash=signature(config)
    work=args.work_dir/config_hash[:16];work.mkdir(parents=True,exist_ok=True)
    atomic_json(work/'config.json',config)
    if args.selection.exists():
        old=json.loads(args.selection.read_text())
        if old.get('config_hash')!=config_hash:
            raise RuntimeError('Selection file belongs to another pilot. Use --selection with a new filename; do not overwrite a selection used by experiments.')
    results={};selected={}
    print(f'45,000 train / 5,000 validation; test set untouched. {args.steps} updates per candidate.',flush=True)
    print(f'Pilot records: {work.resolve()}',flush=True)
    for architecture in ('mlp','cnn'):
        protocol.configure(architecture,args.patch_size)
        perm=permutation('a',1)
        rows=[]
        for i,lr in enumerate(RATES):
            path=work/f'{architecture}_{i:02d}.json'
            if path.exists():
                row=json.loads(path.read_text())
                if row['config_hash']!=config_hash or row['lr']!=lr:raise RuntimeError('Cached candidate mismatch')
                print(f'Reusing {architecture} lr={lr:g}: {row["status"]}',flush=True)
                rows.append(row);continue
            seed_everything(args.seed)
            model=make_model().to(args.device)
            opt=CAPAdam(parameter_groups(model),lr=lr)
            sampler=IndexSampler(len(ty),args.seed,128)
            status='ok';history=[]
            for step in range(1,args.steps+1):
                idx=sampler.next_indices()
                opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(tx[idx][:,perm].to(args.device)),ty[idx].to(args.device))
                if not torch.isfinite(loss):status='nonfinite';break
                loss.backward();opt.step()
                if step%1000==0 or step==args.steps:
                    acc,ce=score(model,vx,vy,perm,args.device)
                    if not torch.isfinite(torch.tensor([acc,ce])).all():status='nonfinite';break
                    history.append({'step':step,'validation_accuracy':acc,'validation_loss':ce})
                    print(f'{architecture:3} CAP lr={lr:.7g} step={step:5} val={acc:.2%} loss={ce:.4f}',flush=True)
            row={'config_hash':config_hash,'model':architecture,'lr':lr,'status':status,'history':history,
                 'validation_accuracy':history[-1]['validation_accuracy'] if status=='ok' else None,
                 'validation_loss':history[-1]['validation_loss'] if status=='ok' else None}
            atomic_json(path,row);rows.append(row)
            del model,opt
        results[architecture]=rows;selected[architecture]=winner(rows)
        print(f'SELECTED {architecture}: lr={selected[architecture]:.10g}',flush=True)
    payload={'schema':'cifar-cap-lr-selection-v1','config':config,'config_hash':config_hash,
             'selected':selected,'results':results}
    if args.selection.exists() and json.loads(args.selection.read_text())!=payload:
        raise RuntimeError('Refusing to replace a different completed selection')
    atomic_json(args.selection,payload)
    print(f'Saved {args.selection.resolve()}. Next: bash run_all.sh',flush=True)

if __name__=='__main__':main()
