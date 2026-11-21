"""Verify extension provenance, merged selection, and unchanged numerical recovery."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import cifar_baselines as base
import cifar_baselines_runtime as runtime
import cifar_lr_extension as ext
import upgd_rebuttal as shared


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name)/'parent'
        self.out=Path(self.temp.name)/'extended'
        self.old=base.campaign(self.parent,initialize=True,smoke=True)
        # Synthetic terminal exports exercise real file/checksum validation without
        # repeating 40 numerical parent jobs already covered by the base suite.
        for job in base.jobs(self.old,self.parent,'search'):
            self.fake_result(self.parent,job,accuracy=.8 if job['lr_index']==4 else .1)
        base.select(self.old,self.parent)
        self.p=ext.campaign(self.out,initialize=True,parent=self.parent)

    def fake_result(self,out,job,accuracy=.2,status='complete'):
        cell=out/'jobs'/job['id']
        cell.mkdir(parents=True,exist_ok=True)
        artifacts={}
        for name in ('latest.pt','tasks.csv','pool.csv','scalars.csv','validation.csv','environment.json','data_identity.json'):
            (cell/name).write_bytes(b'test-only synthetic export\n')
            artifacts[name]=shared.file_hash(cell/name)
        r=dict(job=job,status=status,step=job['updates'],smoke=True,artifact_hashes=artifacts,
            metrics=dict(validation_accuracy=accuracy,validation_loss=1.) if status=='complete' else {})
        shared.atomic_json(cell/'result.json',r)
        return r

    def finish_extension(self,winner_index=11):
        for j in ext.jobs(self.p,self.out,'search'):
            self.fake_result(self.out,j,accuracy=.9 if j['arm']=='upgd' and j['lr_index']==winner_index else .2)

    def test_budget_and_original_source_preservation(self):
        jobs=ext.jobs(self.p,self.out,'search')
        self.assertEqual(len(jobs),12)
        self.assertEqual({j['lr'] for j in jobs},{.03,.1,.3})
        self.assertEqual(self.p['learning_rates'],base.RATES+[.03,.1,.3])
        self.assertEqual(base.campaign(self.parent),self.old)
        self.assertEqual(ext.campaign(self.out),self.p)
        self.assertEqual(self.p['parent_campaign_hash'],shared.digest(self.old))
        with self.assertRaises(RuntimeError): ext.select(self.p,self.out)

    def test_merge_divergence_and_changed_parent_gates(self):
        self.finish_extension()
        j=ext.jobs(self.p,self.out,'search')[0]
        self.fake_result(self.out,j,status='diverged')
        selected=ext.select(self.p,self.out)
        self.assertEqual(len(selected['result_hashes']),52)
        self.assertEqual(selected['selected']['mlp-upgd']['lr'],.1)
        self.assertEqual(selected['selected']['cnn-adamw']['source'],'parent')
        jobs=ext.jobs(self.p,self.out,'eval')
        self.assertEqual(len(jobs),40)
        self.assertEqual({j['lr'] for j in jobs if j['arm']=='upgd'},{.1})
        artifact=self.parent/'jobs/search-mlp-adamw-00/validation.csv'
        artifact.write_bytes(b'changed')
        with self.assertRaises(RuntimeError): ext.select(self.p,self.out,write=False)

    def test_boundary_blocks_evaluation_and_test_access_blocks_init(self):
        self.finish_extension(winner_index=12)
        self.assertTrue(ext.select(self.p,self.out)['selected']['mlp-upgd']['at_grid_boundary'])
        with self.assertRaisesRegex(RuntimeError,'boundary'): ext.jobs(self.p,self.out,'eval')
        shared.atomic_json(self.parent/'jobs/eval-mlp-adamw-42-a/job.json',{})
        with self.assertRaisesRegex(RuntimeError,'evaluation already started'):
            ext.campaign(Path(self.temp.name)/'other',initialize=True,parent=self.parent)

    def test_numerical_pause_resume_and_protocol_restoration(self):
        other=Path(self.temp.name)/'uninterrupted'
        ext.campaign(other,initialize=True,parent=self.parent)
        with contextlib.redirect_stdout(io.StringIO()),ext.worker_protocol():
            for model in ('mlp','cnn'):
                job=next(j for j in ext.jobs(self.p,self.out,'search') if j['model']==model and j['arm']=='upgd' and j['lr_index']==10)
                paused=runtime.run_job(self.out,job,data_dir=self.temp.name,device='cpu',stop_after_updates=3)
                self.assertEqual(paused['status'],'paused')
                resumed=runtime.run_job(self.out,job,data_dir=self.temp.name,device='cpu',resume=True)
                direct=runtime.run_job(other,job,data_dir=self.temp.name,device='cpu')
                self.assertEqual(resumed['metrics'],direct['metrics'])
                a=torch.load(self.out/'jobs'/job['id']/'latest.pt',weights_only=False)
                b=torch.load(other/'jobs'/job['id']/'latest.pt',weights_only=False)
                for name in a['model']: self.assertTrue(torch.equal(a['model'][name],b['model'][name]))
                self.assertEqual(ext.result(self.out,job)['status'],'complete')
        self.assertIs(runtime.protocol,base)
        with self.assertRaises(ValueError),ext.worker_protocol(): raise ValueError('restore on exception')
        self.assertIs(runtime.protocol,base)

    def test_evaluation_report_path(self):
        self.finish_extension();ext.select(self.p,self.out)
        with contextlib.redirect_stdout(io.StringIO()),ext.worker_protocol():
            for job in ext.jobs(self.p,self.out,'eval'):
                runtime.run_job(self.out,job,data_dir=self.temp.name,device='cpu')
        summary=ext.report(self.p,self.out)
        self.assertEqual(len(summary),4)
        self.assertTrue(all(r['seeds']==5 for r in summary))
        self.assertEqual(base.campaign(self.parent),self.old)


if __name__=='__main__': unittest.main()
