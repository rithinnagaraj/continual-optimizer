"""Stdlib-only selection and launch wiring tests; run with python -m unittest test_lr_selection."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from cifar_mech.lr_selection import winner, signature, load_selection
from cifar_mech.constants import CAP

class SelectionTests(unittest.TestCase):
    def payload(self):
        rates=[.00001*(i+1) for i in range(10)]
        cfg={'source_version':'test-source','patch_size':4,
             'cap_without_lr':{k:v for k,v in CAP.items() if k!='lr'},'learning_rates':rates}
        rows=[{'lr':lr,'status':'ok','validation_accuracy':.1+i*.01,'validation_loss':3.} for i,lr in enumerate(rates)]
        cnn=copy.deepcopy(rows)
        cnn[0]['validation_accuracy']=.9
        return {'schema':'cifar-cap-lr-selection-v1','config':cfg,'config_hash':signature(cfg),
                'results':{'mlp':rows,'cnn':cnn},'selected':{'mlp':rates[-1],'cnn':rates[0]}}

    def test_selection_and_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection.json';v=self.payload();p.write_text(json.dumps(v))
            self.assertEqual(load_selection(p,'mlp',4,'test-source',CAP)[0],v['selected']['mlp'])
            with self.assertRaises(ValueError):load_selection(p,'mlp',8,'test-source',CAP)
            with self.assertRaises(ValueError):load_selection(p,'mlp',4,'changed',CAP)
            v['selected']['mlp']=.2;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):load_selection(p,'mlp',4,'test-source',CAP)

    def test_failed_candidates_and_ties(self):
        rows=[{'lr':.1,'status':'nonfinite','validation_accuracy':None,'validation_loss':None},
              {'lr':.03,'status':'ok','validation_accuracy':.5,'validation_loss':2.},
              {'lr':.01,'status':'ok','validation_accuracy':.5,'validation_loss':2.}]
        self.assertEqual(winner(rows),.01)
        with self.assertRaises(ValueError):winner(rows[:1])

    def test_both_model_launch_rates_and_output_isolation(self):
        torch_stub=types.ModuleType('torch');torch_stub.set_num_threads=lambda n:None
        cp=types.ModuleType('cifar_mech.checkpointing');cp.source_version=lambda:'test-source'
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,{'torch':torch_stub,'cifar_mech.checkpointing':cp}):
            from cifar_mech import cli, protocol
            p=Path(d)/'selected.json';v=self.payload();p.write_text(json.dumps(v))
            calls=[]
            try:
                with patch.object(cli,'execute',lambda a:calls.append((protocol.ARCHITECTURE,CAP['lr'],str(a.output_dir),protocol.LR_SELECTION))):
                    cli.main(['reference','--model','both','--lr-selection',str(p),'--output-dir',d])
                self.assertEqual([c[1] for c in calls],[v['selected']['mlp'],v['selected']['cnn']])
                self.assertNotEqual(calls[0][2],calls[1][2])
                self.assertTrue(all('lrsel_' in c[2] and c[3] for c in calls))
            finally:
                CAP['lr']=1/300;protocol.LR_SELECTION=None;protocol.configure('mlp',4)

    def test_fixed_paper_rates_do_not_reselect(self):
        torch_stub=types.ModuleType('torch');torch_stub.set_num_threads=lambda n:None
        with patch.dict(sys.modules,{'torch':torch_stub}):
            from cifar_mech import cli, protocol
            calls=[]
            try:
                with patch.object(cli,'execute',lambda a:calls.append((CAP['lr'],str(a.output_dir),protocol.LR_SELECTION))):
                    cli.main(['reference','--model','both','--paper-settings'])
                self.assertEqual([c[0] for c in calls],[2e-4,3e-4])
                self.assertTrue(all('paper_fixed' in c[1] and c[2]=='paper-fixed-rates-v1' for c in calls))
                with self.assertRaises(ValueError):
                    cli.main(['reference','--model','mlp','--paper-settings','--lr-selection','unused.json'])
            finally:
                CAP['lr']=1/300;protocol.LR_SELECTION=None;protocol.configure('mlp',4)

if __name__=='__main__':unittest.main()
