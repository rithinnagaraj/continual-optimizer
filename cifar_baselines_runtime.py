"""CIFAR baseline worker; immutable protocol, exact resume, and timed bounded jobs."""
import os
from pathlib import Path
import time
import uuid
import math
import torch
from torch.nn import functional as F

import upgd_rebuttal as shared
import upgd_rebuttal_runtime as common
import cifar_baselines as protocol
from cap_mechanistic_cifar.cifar_mech.model import MLP300, SmallCNN, parameter_groups
from cap_mechanistic_cifar.cifar_mech.data import normalize
from cap_mechanistic_cifar.cifar_mech.streams import IndexSampler
from cap_mechanistic_cifar.cifar_mech.seed import seed_everything, rng_state, restore_rng

FIELDS=dict(scalars=['step','task','accuracy','loss'],
            tasks=['task','current_domain','online_accuracy','task_end_accuracy','task_end_loss'],
            pool=['task','domain','accuracy','loss','is_current'],
            validation=['step','accuracy','loss'])


def split_indices(y,count,seed):
    """Match the historical CAP pilot's sorted, stratified training split."""
    g=torch.Generator().manual_seed(seed)
    train,val=[],[]
    classes=y.unique(sorted=True)
    if count%len(classes): raise ValueError('Holdout must divide equally across classes')
    for label in classes:
        indices=(y==label).nonzero().flatten()
        indices=indices[torch.randperm(len(indices),generator=g)]
        take=count//len(classes)
        if take>=len(indices): raise ValueError('Insufficient training examples')
        val.append(indices[:take]);train.append(indices[take:])
    return torch.cat(train).sort().values,torch.cat(val).sort().values


def load_data(p,stage,root):
    if p['smoke']:
        gen=torch.Generator().manual_seed(9876)
        raw=torch.randn(60,3072,generator=gen);labels=torch.arange(60)%10
        if stage=='eval':
            ti,vi=torch.arange(60),torch.arange(20)
            tx,ty=raw,labels
            ex,ey=torch.randn(20,3072,generator=gen),torch.arange(20)%10
        else:
            ti,vi=split_indices(labels,p['holdout_examples'],p['split_seed'])
            tx,ty,ex,ey=raw[ti],labels[ti],raw[vi],labels[vi]
    else:
        from torchvision.datasets import CIFAR100
        dataset=CIFAR100(str(root),train=True,download=True)
        raw=torch.from_numpy(dataset.data);labels=torch.tensor(dataset.targets)
        if stage=='eval':
            test=CIFAR100(str(root),train=False,download=True)
            ti=torch.arange(len(labels))
            vi=torch.randperm(len(test),generator=torch.Generator().manual_seed(p['test_subset_seed']))[:p['test_examples']]
            tx,ty=normalize(raw),labels
            ex,ey=normalize(torch.from_numpy(test.data)[vi]),torch.tensor(test.targets)[vi]
        else:
            ti,vi=split_indices(labels,p['holdout_examples'],p['split_seed'])
            if stage=='pilot':
                # Cost-only heldout subset, not official test evaluation or model selection.
                chosen=torch.randperm(len(vi),generator=torch.Generator().manual_seed(p['pilot_subset_seed']))[:p['test_examples']]
                vi=vi[chosen]
            tx,ty,ex,ey=normalize(raw[ti]),labels[ti],normalize(raw[vi]),labels[vi]
    ident=dict(smoke=p['smoke'],train_examples=len(ty),evaluation_examples=len(ey),
        evaluation_source='official_test' if stage=='eval' else 'train_holdout',
        train_indices_sha256=common.tensor_hash(ti),eval_indices_sha256=common.tensor_hash(vi),
        train_x_sha256=common.tensor_hash(tx),train_y_sha256=common.tensor_hash(ty),
        eval_x_sha256=common.tensor_hash(ex),eval_y_sha256=common.tensor_hash(ey))
    return tx,ty,ex,ey,ident,ti,vi


def patch_permutation(p,stream,domain):
    gen=torch.Generator().manual_seed(p['a_permutation_seed' if stream=='a' else 'b_permutation_seed']+domain)
    size=p['patch_size'];grid=32//size
    order=torch.randperm(grid*grid,generator=gen)
    indices=torch.arange(3072).reshape(3,grid,size,grid,size).permute(0,1,3,2,4)
    patches=indices.reshape(3,grid*grid,size,size)[:,order]
    return patches.reshape(3,grid,grid,size,size).permute(0,1,3,2,4).reshape(-1)


def stream_spec(p,job):
    if job['stream']=='a':
        schedule=list(range(1,job['tasks']+1))
        domains=schedule
    else:
        gen=torch.Generator().manual_seed(p['schedule_seed'])
        rounds=math.ceil(job['tasks']/p['pool_size'])
        schedule=torch.cat([torch.randperm(p['pool_size'],generator=gen) for _ in range(rounds)]).tolist()[:job['tasks']]
        domains=list(range(p['pool_size']))
    return schedule,{d:patch_permutation(p,job['stream'],d) for d in domains}


def make_model(job):
    return MLP300() if job['model']=='mlp' else SmallCNN()


def make_optimizer(p,job,model):
    if job['arm']=='adamw':
        return torch.optim.AdamW(model.parameters(),lr=job['lr'],**p['adamw'])
    if job['arm']=='upgd':
        return common.make_optimizer(model,dict(arm='upgd',config=dict(lr=job['lr'],**p['upgd'])))
    from cap_mechanistic_cifar.cifar_mech.optim import CAPAdam
    return CAPAdam(parameter_groups(model),lr=job['lr'])


@torch.no_grad()
def evaluate(model,x,y,permutations,domains,device,batch_size):
    model.eval();rows=[]
    try:
        for domain in domains:
            correct,total=0,0.
            for start in range(0,len(y),batch_size):
                labels=y[start:start+batch_size].to(device)
                logits=model(x[start:start+batch_size][:,permutations[domain]].to(device))
                loss=F.cross_entropy(logits,labels,reduction='sum').item()
                if not math.isfinite(loss): raise common.NumericalDivergence('Nonfinite evaluation loss')
                total+=loss;correct+=(logits.argmax(1)==labels).sum().item()
            rows.append(dict(domain=domain,accuracy=correct/len(y),loss=total/len(y)))
    finally:
        model.train()
    return rows


def metrics(p,job,state):
    if state['step']!=job['updates'] or [r['task'] for r in state['tasks']]!=list(range(1,job['tasks']+1)):
        raise RuntimeError('Incomplete task coverage')
    if [r['step'] for r in state['scalars']]!=list(range(p['observation_interval'],job['updates']+1,p['observation_interval'])):
        raise RuntimeError('Incomplete scalar coverage')
    if job['stream']=='b':
        actual=[(r['task'],r['domain']) for r in state['pool']]
        expected={(t,d) for t in range(1,job['tasks']+1) for d in range(p['pool_size'])}
        if len(actual)!=len(expected) or set(actual)!=expected: raise RuntimeError('Incomplete pool coverage')
        for task in state['tasks']:
            current=[r['domain'] for r in state['pool'] if r['task']==task['task'] and r['is_current']]
            if current!=[task['current_domain']]: raise RuntimeError('Wrong current-domain label')
    if job['stage']=='pilot': return {}  # timing only; never eligible for LR/scientific selection
    if job['stage']=='search':
        if [r['step'] for r in state['validation']]!=list(range(p['validation_interval'],job['updates']+1,p['validation_interval'])):
            raise RuntimeError('Incomplete validation history')
        final=state['validation'][-1]
        return dict(validation_accuracy=final['accuracy'],validation_loss=final['loss'])
    if job['stream']=='a':
        return dict(plasticity=sum(r['accuracy'] for r in state['scalars'] if r['step']>job['updates']//2)/(job['updates']//2//p['observation_interval']))
    retained=[r['accuracy'] for r in state['pool'] if r['task']==job['tasks'] and not r['is_current']]
    return dict(retention=sum(retained)/len(retained))


def run_job(output,job,*,data_dir,device='cuda',resume=False,checkpoint_every=1000,stop_after_updates=None):
    if checkpoint_every<=0 or (stop_after_updates is not None and stop_after_updates<=0):
        raise ValueError('Work/checkpoint bounds must be positive')
    p=protocol.campaign(output)
    if job not in protocol.jobs(p,output,job['stage'],(job['arm'],)):
        raise RuntimeError('Job outside frozen protocol')
    cell=Path(output)/'jobs'/job['id']
    with shared.file_lock(cell/'worker.lock'):
        existing=protocol.result(output,job,required=False)
        if existing:
            if not resume: raise FileExistsError('Use --resume to verify and skip completed jobs')
            print(f'{job["id"]}: verified {existing["status"]}; skipped',flush=True)
            return existing
        progress=cell/'latest.pt'
        if (progress.exists() or (cell/'job.json').exists()) and not resume:
            raise FileExistsError('Use --resume for partial job')
        shared.freeze_json(cell/'job.json',job)
        device=torch.device(device)
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);seed_everything(job['seed'])
        env={**common.environment(device), 'cudnn_deterministic':torch.backends.cudnn.deterministic,
             'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
        shared.freeze_json(Path(output)/'environment.json',env)
        shared.freeze_json(cell/'environment.json',env)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        attempt,start=uuid.uuid4().hex,time.perf_counter()
        state=dict(step=0,tasks=[],pool=[],scalars=[],validation=[],task_accuracy_sum=0.,task_count=0,
                   training_seconds=0.,evaluation_seconds=0.,pilot_single_domain_seconds=0.,status='running')
        def event(kind,**extra):
            common.synchronize(device)
            shared.append_event(cell/'timing.jsonl',dict(attempt=attempt,event=kind,unix_time=time.time(),
                elapsed_seconds=time.perf_counter()-start,step=state['step'],**extra))
        event('start',stop_after_updates=stop_after_updates,checkpoint_every=checkpoint_every)
        try:
            tx,ty,ex,ey,data_id,ti,vi=load_data(p,job['stage'],data_dir)
            shared.freeze_json(Path(output)/'data_identity'/f'{job["stage"]}.json',data_id)
            shared.freeze_json(cell/'data_identity.json',data_id)
            model=make_model(job).to(device);optimizer=make_optimizer(p,job,model)
            sampler=IndexSampler(len(ty),job['seed'],p['batch_size'])
            schedule,permutations=stream_spec(p,job)
            identity=dict(job=job,environment=env,data=data_id)
            initial_hash=shared.digest({n:common.tensor_hash(v) for n,v in model.state_dict().items()})
            if progress.exists():
                saved=torch.load(progress,map_location='cpu',weights_only=False)
                if saved['identity']!=identity or saved['schedule']!=schedule: raise RuntimeError('Resume identity changed')
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                sampler.load_state_dict(saved['sampler']);state=saved['progress'];restore_rng(saved['rng'])
                initial_hash=saved['initial_model_hash']
            start_step=state['step'];last_commit_step=start_step
            def checkpoint(block_seconds=0.):
                nonlocal last_commit_step
                before=time.perf_counter()
                common.atomic_checkpoint(progress,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    sampler=sampler.state_dict(),rng=rng_state(),progress=state,schedule=schedule,
                    parameter_names=[n for n,_ in model.named_parameters()],train_indices=ti,evaluation_indices=vi,
                    initial_model_hash=initial_hash))
                event('checkpoint',block_updates=state['step']-last_commit_step,block_training_seconds=block_seconds,
                      checkpoint_seconds=time.perf_counter()-before,training_seconds=state['training_seconds'],
                      evaluation_seconds=state['evaluation_seconds'])
                last_commit_step=state['step']
            if not progress.exists(): checkpoint()
            event('ready',source_stage=job['stage'])
            segment=time.perf_counter()
            while state['step']<job['updates'] and state['status']=='running':
                task=state['step']//job['task_updates']+1;domain=schedule[task-1]
                indices=sampler.next_indices()
                optimizer.zero_grad(set_to_none=True)
                labels=ty[indices].to(device)
                logits=model(tx[indices][:,permutations[domain]].to(device))
                loss=F.cross_entropy(logits,labels);loss_value=loss.item()
                if not math.isfinite(loss_value):
                    state['status']='diverged';state['failure_reason']='Nonfinite training loss';break
                accuracy=(logits.argmax(1)==labels).float().mean().item()
                loss.backward()
                with torch.device(device): optimizer.step()
                state['step']+=1;step=state['step']
                state['task_accuracy_sum']+=accuracy;state['task_count']+=1
                if step%p['observation_interval']==0:
                    state['scalars'].append(dict(step=step,task=task,accuracy=accuracy,loss=loss_value))
                boundary=step%job['task_updates']==0
                validate=job['stage']=='search' and step%p['validation_interval']==0
                pause=stop_after_updates is not None and step-start_step>=stop_after_updates
                if not (boundary or validate or pause or step%checkpoint_every==0): continue
                common.synchronize(device);block=time.perf_counter()-segment
                state['training_seconds']+=block
                if boundary or validate:
                    before=time.perf_counter()
                    domains=list(range(p['pool_size'])) if job['stream']=='b' else [domain]
                    try:
                        evaluated=evaluate(model,ex,ey,permutations,domains,device,p['eval_batch_size'])
                    except common.NumericalDivergence as error:
                        state['status']='diverged';state['failure_reason']=str(error);evaluated=[]
                    common.synchronize(device);state['evaluation_seconds']+=time.perf_counter()-before
                    if evaluated:
                        current=next(r for r in evaluated if r['domain']==domain)
                        if validate: state['validation'].append(dict(step=step,accuracy=current['accuracy'],loss=current['loss']))
                        if boundary:
                            state['tasks'].append(dict(task=task,current_domain=domain,
                                online_accuracy=state['task_accuracy_sum']/state['task_count'],
                                task_end_accuracy=current['accuracy'],task_end_loss=current['loss']))
                            if job['stream']=='b': state['pool'].extend(dict(task=task,**r,is_current=r['domain']==domain) for r in evaluated)
                            state['task_accuracy_sum']=0.;state['task_count']=0
                            print(f'{job["id"]}: task {task}/{job["tasks"]}, step {step}/{job["updates"]}',flush=True)
                    if job['stage']=='pilot' and evaluated:
                        before=time.perf_counter()
                        evaluate(model,ex,ey,permutations,[domain],device,p['eval_batch_size'])
                        common.synchronize(device);state['pilot_single_domain_seconds']+=time.perf_counter()-before
                if step==job['updates'] and state['status']=='running': state['status']='complete'
                checkpoint(block)
                if pause and state['status']=='running':
                    event('paused',invocation_updates=step-start_step)
                    return dict(status='paused',step=step)
                segment=time.perf_counter()
            if state['status']=='diverged':
                common.synchronize(device);block=time.perf_counter()-segment
                state['training_seconds']+=block;checkpoint(block)
            values=metrics(p,job,state) if state['status']=='complete' else {}
            for name,fields in FIELDS.items(): shared.write_csv(cell/f'{name}.csv',state[name],fields)
            event(state['status'],invocation_updates=state['step']-start_step)
            result=dict(job=job,status=state['status'],step=state['step'],smoke=p['smoke'],
                metrics=values,failure_reason=state.get('failure_reason'),initial_model_hash=initial_hash,
                training_seconds=state['training_seconds'],evaluation_seconds=state['evaluation_seconds'],
                pilot_single_domain_seconds=state['pilot_single_domain_seconds'],timing=shared.timing_summary(cell/'timing.jsonl'),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                artifact_hashes={name:shared.file_hash(cell/name) for name in
                    ('latest.pt','tasks.csv','pool.csv','scalars.csv','validation.csv','environment.json','data_identity.json')})
            shared.atomic_json(cell/'result.json',result)
            return result
        except BaseException as error:
            event('error',error_type=type(error).__name__,message=str(error));raise
