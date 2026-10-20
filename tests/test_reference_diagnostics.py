"""Endpoint reporting must preserve failed gates and reject incomplete inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.reference_diagnostics import SEEDS, METRICS, source_hash, validate_reference


class ReferenceDiagnosticTests(unittest.TestCase):
    def fixture(self, root):
        core=root/"core";(core/"configs").mkdir(parents=True)
        baseline={"k":7e-5};streams={"tasks":200}
        for name,value in (("final_cap",baseline),("streams",streams)):
            (core/"configs"/(name+".json")).write_text(json.dumps(value),encoding="utf-8")
        output=root/"output";directory=output/"reproduction";(directory/"checkpoints").mkdir(parents=True)
        hashes={}
        for seed in SEEDS:
            for stream,tasks in (("a",(20,50,100,150,200)),("b",(50,100,150,200))):
                for task in tasks:
                    name=f"{stream}_seed{seed}_task{task}.pt"
                    (directory/"checkpoints"/name).write_bytes(b"fixture, not a model")
                    hashes[name]=hashlib.sha256(b"fixture, not a model").hexdigest()
        config={"smoke":False,"seeds":list(SEEDS),"baseline":baseline,"streams":streams,
                "threads":1,"source_version":source_hash(core)}
        gate={"passed":False,"valid_reference":True,"source_version":source_hash(core),
              "baseline_config_sha256":hashlib.sha256(json.dumps(baseline,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
              "tolerance":.0005,"checkpoint_hashes":hashes,
              "rows":[{"seed":s,"metric":m,"observed":.8,"expected":.9,
                       "absolute_error":abs(.8-.9),"passed":False} for s in SEEDS for m in METRICS]}
        (directory/"run_config.json").write_text(json.dumps(config),encoding="utf-8")
        (directory/"reproduction_gate.json").write_text(json.dumps(gate),encoding="utf-8")
        return output,core,config,gate

    def test_failed_tolerance_is_reportable_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            output,core,_,_=self.fixture(Path(temp))
            path=output/"reproduction/reproduction_gate.json";before=path.read_bytes()
            self.assertFalse(validate_reference(output,core)["passed"])
            self.assertEqual(path.read_bytes(),before)

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output,core,_,gate=self.fixture(Path(temp));gate["rows"].pop()
            (output/"reproduction/reproduction_gate.json").write_text(json.dumps(gate),encoding="utf-8")
            with self.assertRaises(ValueError):validate_reference(output,core)

    def test_changed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output,core,_,_=self.fixture(Path(temp))
            next((output/"reproduction/checkpoints").glob("*.pt")).write_bytes(b"changed")
            with self.assertRaises(ValueError):validate_reference(output,core)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output,core,_,_=self.fixture(Path(temp));(core/"changed.py").write_text("x=1",encoding="utf-8")
            with self.assertRaises(ValueError):validate_reference(output,core)

    def test_synthetic_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output,core,config,_=self.fixture(Path(temp));config["smoke"]=True
            (output/"reproduction/run_config.json").write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaises(ValueError):validate_reference(output,core)
