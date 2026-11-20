"""Scientific estimator checks, actual pause/recovery, and source integrity gates."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import upgd_rebuttal as parent
import upgd_rebuttal_runtime as training
import upgd_m1 as protocol
import upgd_m1_runtime as runtime
from model import MLP300
from cap_mechanistic.cap_mech.probes.functional_damage import antithetic_damage


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_plan_does_not_import_torch(self):
        code = "import sys; from experiment_upgd_m1 import main; main(['plan']); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], cwd=protocol.ROOT, check=True, capture_output=True)

    def test_fisher_matches_individual_gradients_not_squared_mean(self):
        torch.manual_seed(7)
        model = MLP300(784, 10).eval()
        x, y = torch.randn(4, 784), torch.tensor([0, 1, 2, 3])
        actual = runtime.fisher_domain(model, x, y, torch.arange(784), torch.arange(4), 4, 'cpu', 2)
        expected = {n:torch.zeros_like(p) for n,p in model.named_parameters() if n in protocol.LAYERS}
        for i in range(4):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
            for n,p in model.named_parameters():
                if n in expected:
                    expected[n] += p.grad.square()/4
        for n in expected:
            torch.testing.assert_close(actual[n], expected[n], atol=1e-8, rtol=1e-5)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        self.assertFalse(torch.allclose(actual[protocol.LAYERS[-1]], dict(model.named_parameters())[protocol.LAYERS[-1]].grad.square()))

    def test_noise_matches_original_m1_and_restores_on_exception(self):
        p = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3,4)/10)
        original = p.detach().clone()
        indices = torch.tensor([0, 3, 8])
        loss = lambda: float(p.square().sum().item())
        baseline = loss()
        expected, draws = antithetic_damage(p, indices, loss, baseline, 1234, 3, .1)
        actual = runtime.perturb(p, indices, lambda:dict(losses=[loss()]), [baseline], 1234, 3, .1)
        self.assertEqual([x[0] for x in actual['draw_damages_by_domain']], draws)
        self.assertTrue(torch.equal(p, original))
        def fail():
            raise RuntimeError('interrupted evaluation')
        with self.assertRaises(RuntimeError):
            runtime.perturb(p, indices, fail, [baseline], 1234, 3, .1)
        self.assertTrue(torch.equal(p, original))

    def test_signed_utility_order_and_random_group_sizes(self):
        p = protocol.settings()
        specs = protocol.groups(p, 42)
        self.assertEqual(len(specs), 72)
        values = torch.tensor([-10., -2., 0., 1., 3.])
        spec = {**specs[0], 'coordinate_count':2}
        scores = {'utility':{spec['layer']:values}}
        self.assertEqual(runtime.selected_indices(spec, scores, 5).tolist(), [3,4])
        self.assertEqual(runtime.selected_indices({**spec, 'group':'bottom'}, scores, 5).tolist(), [0,1])
        self.assertEqual([s['coordinate_count'] for s in specs[::24]], [2352,900,30])
        for s in specs:
            if s['comparator']=='random':
                ids = runtime.selected_indices(s, {}, 235200 if s['layer']==protocol.LAYERS[0] else 90000 if s['layer']==protocol.LAYERS[1] else 3000)
                self.assertEqual(len(ids.unique()), s['coordinate_count'])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.tmp.name)/'source'
        pp = parent.initialize(cls.source, smoke=True)
        cls.jobs = [parent.job_spec(pp, 'eval', 22, s, 'b', 'selected') for s in pp['report_seeds']]
        parent.atomic_json(cls.source/'selected_config.json', dict(config_index=22, config=pp['configurations'][22]))
        with patch.object(parent, 'jobs', return_value=cls.jobs), contextlib.redirect_stdout(io.StringIO()):
            for job in cls.jobs:
                training.run_job(cls.source, job, data_dir=cls.source/'unused', device='cpu')
            cls.p = protocol.build_manifest(cls.source)
        cls.source_hashes = {str(p.relative_to(cls.source)):parent.file_hash(p) for p in cls.source.rglob('*') if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, output, seed=42, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runtime.run_job(self.p, self.source, output, seed, data_dir=self.source/'unused', device='cpu', **kwargs)

    def test_full_smoke_all_seeds_report_and_duplicate_skip(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RuntimeError):
                protocol.report(self.p, out)
            for seed in self.p['seeds']:
                self.run_worker(out, seed)
            result = protocol.report(self.p, out)
            self.assertTrue(result['smoke'])
            self.assertEqual(result['completed_seeds'], 5)
            self.assertEqual(len(result['summary']), 3)
            log = protocol.directory(out,42)/'timing.jsonl'
            before = log.read_bytes()
            self.run_worker(out, resume=True)
            self.assertEqual(before, log.read_bytes())
            self.assertEqual(self.source_hashes, {str(p.relative_to(self.source)):parent.file_hash(p) for p in self.source.rglob('*') if p.is_file()})
            # Changed raw result evidence must not be admitted to a report.
            group = protocol.directory(out,42)/'groups/000.json'
            data = parent.read_json(group)
            data['draw_damages_by_domain'][0][0] += .1
            parent.atomic_json(group,data)
            with self.assertRaises(RuntimeError):
                protocol.report(self.p,out)

    def test_resumed_fisher_and_groups_equal_uninterrupted(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed:
            self.run_worker(full)
            first = self.run_worker(resumed, max_fisher_domains=1)
            self.assertEqual(first['stage'],'fisher')
            second = self.run_worker(resumed, max_groups=3, resume=True)
            self.assertEqual(second['committed_groups'],3)
            self.run_worker(resumed, resume=True)
            for name in ['baseline.json', 'rankings.csv', *[f'groups/{i:03d}.json' for i in range(18)]]:
                self.assertEqual((protocol.directory(full,42)/name).read_bytes(), (protocol.directory(resumed,42)/name).read_bytes())
            for comp in ('utility','fisher'):
                a = torch.load(protocol.directory(full,42)/'scores.pt', weights_only=True)[comp]
                b = torch.load(protocol.directory(resumed,42)/'scores.pt', weights_only=True)[comp]
                for n in a:
                    self.assertTrue(torch.equal(a[n],b[n]))
            # Recover a completed computation that failed before result publication.
            (protocol.directory(resumed,42)/'result.json').unlink()
            self.run_worker(resumed, resume=True)
            self.assertIsNotNone(protocol.verify_result(self.p,resumed,42))

    def test_failure_during_group_replays_only_uncommitted_work(self):
        with tempfile.TemporaryDirectory() as out:
            self.run_worker(out,max_groups=2)
            with patch.object(runtime,'perturb',side_effect=RuntimeError('injected interruption')):
                with self.assertRaises(RuntimeError):
                    self.run_worker(out,resume=True)
            self.assertEqual(len(protocol.completed_rows(self.p,out,42)),2)
            self.run_worker(out,resume=True)
            self.assertIsNotNone(protocol.verify_result(self.p,out,42))

    def test_wrong_source_hash_and_corrupt_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            wrong = copy.deepcopy(self.p)
            wrong['sources']['42']['checkpoint_hash'] = '0'*64
            with self.assertRaises(RuntimeError):
                runtime.run_job(wrong,self.source,out,42,data_dir=self.source/'unused',device='cpu')
        with tempfile.TemporaryDirectory() as out:
            self.run_worker(out,max_fisher_domains=1)
            path = protocol.directory(out,42)/'fisher/000.pt'
            path.write_bytes(path.read_bytes()+b'corruption')
            with self.assertRaises(RuntimeError):
                self.run_worker(out,resume=True)

    def test_manifest_is_frozen_and_source_environment_is_checked(self):
        with tempfile.TemporaryDirectory() as out:
            with patch.object(parent,'jobs',return_value=self.jobs):
                protocol.campaign(self.source,out,initialize=True)
                protocol.campaign(self.source,out)
                changed = parent.read_json(Path(out)/'campaign.json')
                changed['noise_scale'] = 1.
                parent.atomic_json(Path(out)/'campaign.json',changed)
                with self.assertRaises(RuntimeError):
                    protocol.campaign(self.source,out)
        with tempfile.TemporaryDirectory() as out:
            wrong=copy.deepcopy(self.p)
            wrong['environment']['torch']='different'
            with self.assertRaises(RuntimeError):
                runtime.run_job(wrong,self.source,out,42,data_dir=self.source/'unused',device='cpu')


if __name__=='__main__':
    unittest.main()
