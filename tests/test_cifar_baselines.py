"""CIFAR scientific identity, selection gates, and actual numerical recovery."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

import cifar_baselines as p
import cifar_baselines_runtime as runtime
import upgd_rebuttal as shared
from cap_mechanistic_cifar.cifar_mech import protocol as original_protocol
from cap_mechanistic_cifar.cifar_mech.streams import permutation as original_permutation, schedule as original_schedule
from cap_mechanistic_cifar.cifar_mech.data import normalize
from cap_mechanistic_cifar.cifar_mech.model import parameter_groups
from cap_mechanistic_cifar.cifar_mech.optim import CAPAdam


class ProtocolTests(unittest.TestCase):
    def test_plan_is_stdlib_only(self):
        code="import sys; from experiment_cifar_baselines import main; main(['plan']); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=p.ROOT,check=True,capture_output=True)

    def test_protocol_budget_and_development_seed_separation(self):
        plan=p.protocol()
        self.assertNotIn(plan['development_seed'],plan['report_seeds'])
        self.assertEqual(len(p.jobs(plan,None,'pilot')),4)
        self.assertEqual(sum(j['updates'] for j in p.jobs(plan,None,'search')),200000)
        self.assertEqual(len(p.jobs(plan,None,'search')),40)
        self.assertEqual(plan['eval_tasks']*plan['updates_per_task'],400000)

    def test_patch_permutations_and_schedule_match_frozen_cifar(self):
        plan=p.protocol()
        with patch.object(original_protocol,'PATCH_SIZE',4):
            for stream,domain in [('a',1),('a',80),('b',0),('b',39)]:
                self.assertTrue(torch.equal(runtime.patch_permutation(plan,stream,domain),original_permutation(stream,domain)))
        job=p.spec(plan,'eval','cnn','upgd',42,'b',lr=.001)
        schedule,_=runtime.stream_spec(plan,job)
        self.assertEqual(schedule,original_schedule().tolist()[:80])
        self.assertEqual(sorted(schedule[:40]),list(range(40)))
        self.assertEqual(sorted(schedule[40:]),list(range(40)))

    def test_split_matches_original_sorted_stratification(self):
        labels=torch.arange(1000)%100
        a,b=runtime.split_indices(labels,200,777)
        self.assertFalse(set(a.tolist())&set(b.tolist()))
        self.assertEqual(len(a),800)
        self.assertTrue(torch.equal(torch.bincount(labels[b]),torch.full((100,),2)))
        gen=torch.Generator().manual_seed(777);expected=[]
        for c in range(100):
            indices=(labels==c).nonzero().flatten()
            expected.append(indices[torch.randperm(len(indices),generator=gen)[:2]])
        self.assertTrue(torch.equal(b,torch.cat(expected).sort().values))

    def test_pilot_and_search_never_request_official_test(self):
        calls=[]
        class Fake:
            def __init__(self,root,train,download):
                calls.append(train)
                if not train: raise AssertionError('Official test accessed')
                self.data=np.zeros((60,32,32,3),dtype=np.uint8)
                self.targets=list(np.arange(60)%10)
        plan=p.protocol(True);plan['smoke']=False
        with patch.dict(sys.modules,{'torchvision.datasets':SimpleNamespace(CIFAR100=Fake)}):
            for stage in ('pilot','search'):
                *_,ident,ti,vi=runtime.load_data(plan,stage,'unused')
                self.assertEqual(ident['evaluation_source'],'train_holdout')
        self.assertEqual(calls,[True,True])

    def test_selection_is_final_validation_only_and_requires_all_results(self):
        plan=p.protocol(True)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RuntimeError): p.select(plan,out)
            def fake(output,job):
                i=job['lr_index']
                return dict(job=job,status='diverged' if i==9 else 'complete',
                    metrics=dict(validation_accuracy=.8 if i in (2,3,4) else .7,
                                 validation_loss=.5 if i in (3,4) else .6))
            with patch.object(p,'result',side_effect=fake):
                choice=p.select(plan,out)
            self.assertTrue(all(c['lr_index']==3 for c in choice['selected'].values()))

    def test_archived_cap_prefix_has_expected_scope(self):
        ref=shared.read_json(p.CAP_REFERENCE)
        self.assertEqual(ref['eval_tasks'],80)
        self.assertEqual(len(ref['per_seed']),10)
        self.assertEqual({r['arm'] for r in ref['per_seed']},{'cap_archived'})
        for row in ref['per_seed']:
            self.assertEqual(row['balanced'],min(row['plasticity'],row['retention']))
        self.assertIn('not a same-environment',ref['per_seed'][0]['provenance'])


class NumericalTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def worker(self,out,job,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runtime.run_job(out,job,data_dir=Path(out)/'unused',device='cpu',checkpoint_every=3,**kwargs)

    def cp(self,out,job):
        return torch.load(Path(out)/'jobs'/job['id']/'latest.pt',weights_only=False,map_location='cpu')

    def nested_equal(self,a,b):
        if isinstance(a,torch.Tensor): self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.nested_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.nested_equal(x,y)
        elif isinstance(a,np.ndarray):self.assertTrue(np.array_equal(a,b))
        else:self.assertEqual(a,b)

    def test_actual_resume_preserves_adamw_and_upgd_both_architectures(self):
        for model in p.MODELS:
            for arm in p.ARMS:
                with self.subTest(model=model,arm=arm),tempfile.TemporaryDirectory() as full,tempfile.TemporaryDirectory() as resumed:
                    plan=p.campaign(full,True,True);p.campaign(resumed,True,True)
                    job=p.spec(plan,'search',model,arm,lr_index=6)
                    self.worker(full,job)
                    paused=self.worker(resumed,job,stop_after_updates=5)
                    self.assertEqual(paused['status'],'paused')
                    self.worker(resumed,job,resume=True)
                    a,b=self.cp(full,job),self.cp(resumed,job)
                    for k in ('model','optimizer','sampler','rng'):self.nested_equal(a[k],b[k])
                    for k in ('scalars','tasks','pool','validation'):self.assertEqual(a['progress'][k],b['progress'][k])

    def test_pilot_all_four_cells_are_excluded_from_selection_and_timed(self):
        with tempfile.TemporaryDirectory() as out:
            plan=p.campaign(out,True,True)
            results=[self.worker(out,j) for j in p.jobs(plan,out,'pilot')]
            self.assertTrue(all(r['metrics']=={} for r in results))
            self.assertTrue(all(r['peak_cuda_allocated_bytes']==0 for r in results))
            report=p.pilot_report(plan,out)
            self.assertEqual(len(report['rows']),4)
            self.assertGreater(report['projected_baseline_evaluation_hours'],0)
            with self.assertRaises(RuntimeError):p.select(plan,out)
            for model in p.MODELS:
                hashes={r['initial_model_hash'] for r in results if r['job']['model']==model}
                self.assertEqual(len(hashes),1)

    def test_interrupted_pool_evaluation_and_completed_publication_recovery(self):
        with tempfile.TemporaryDirectory() as full,tempfile.TemporaryDirectory() as resumed:
            plan=p.campaign(full,True,True);p.campaign(resumed,True,True)
            job=p.spec(plan,'pilot','cnn','upgd')
            self.worker(full,job)
            with patch.object(runtime,'evaluate',side_effect=RuntimeError('interrupted pool evaluation')):
                with self.assertRaises(RuntimeError):self.worker(resumed,job)
            self.worker(resumed,job,resume=True)
            for key in ('model','optimizer','sampler','rng'):self.nested_equal(self.cp(full,job)[key],self.cp(resumed,job)[key])
            self.assertEqual(self.cp(full,job)['progress']['pool'],self.cp(resumed,job)['progress']['pool'])
            (Path(resumed)/'jobs'/job['id']/'result.json').unlink()
            self.worker(resumed,job,resume=True)
            self.assertEqual(p.result(resumed,job)['status'],'complete')

    def test_invalid_jobs_and_corrupt_exports_are_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            plan=p.campaign(out,True,True);job=p.jobs(plan,out,'pilot')[0]
            bad=copy.deepcopy(job);bad['updates']-=1
            with self.assertRaises(RuntimeError):self.worker(out,bad)
            self.worker(out,job)
            (Path(out)/'jobs'/job['id']/'tasks.csv').write_text('changed')
            with self.assertRaises(RuntimeError):p.result(out,job)

    def test_nonfinite_trial_is_recorded_and_not_a_score(self):
        with tempfile.TemporaryDirectory() as out:
            plan=p.campaign(out,True,True);job=p.jobs(plan,out,'search')[0]
            original=runtime.make_optimizer
            def factory(plan,job,model):
                opt=original(plan,job,model);step=opt.step
                def corrupt():
                    step()
                    with torch.no_grad():next(model.parameters()).fill_(float('nan'))
                opt.step=corrupt
                return opt
            with patch.object(runtime,'make_optimizer',side_effect=factory):result=self.worker(out,job)
            self.assertEqual(result['status'],'diverged');self.assertEqual(result['metrics'],{})

    def test_full_small_search_selection_evaluation_report(self):
        with tempfile.TemporaryDirectory() as out:
            plan=p.campaign(out,True,True)
            for job in p.jobs(plan,out,'search'):self.worker(out,job)
            selected=p.select(plan,out)
            self.assertEqual(len(selected['selected']),4)
            with self.assertRaises(RuntimeError):p.report(plan,out)
            for job in p.jobs(plan,out,'eval'):self.worker(out,job)
            report=p.report(plan,out)
            self.assertEqual(len(report),4)
            self.assertTrue(all(r['seeds']==5 for r in report))
            with self.assertRaises(ValueError):p.report(plan,out,include_archived_cap=True)
            job=p.jobs(plan,out,'eval')[0]
            state=self.cp(out,job)['progress']
            state['scalars'].pop()
            with self.assertRaises(RuntimeError):runtime.metrics(plan,job,state)

    def test_optional_cap_constructor_matches_existing_cifar_optimizer(self):
        plan=p.protocol(True)
        for model in p.MODELS:
            job=p.spec(plan,'eval',model,'cap',42,'a',lr=plan['cap_learning_rates'][model])
            torch.manual_seed(42);a=runtime.make_model(job)
            b=copy.deepcopy(a)
            left=runtime.make_optimizer(plan,job,a);right=CAPAdam(parameter_groups(b),lr=job['lr'])
            x=torch.randn(4,3072);labels=torch.arange(4)
            for net,opt in ((a,left),(b,right)):
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(net(x),labels).backward();opt.step()
            self.nested_equal(a.state_dict(),b.state_dict())


if __name__=='__main__':unittest.main()
