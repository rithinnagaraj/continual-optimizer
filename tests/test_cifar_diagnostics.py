import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from cap_mechanistic_cifar.cifar_mech.optim.cap_final import CAPAdam
from cap_mechanistic_cifar.cifar_mech.model import make_model, parameter_groups
from cap_mechanistic_cifar.cifar_mech import protocol
from cap_mechanistic_cifar.cifar_mech.interventions import apply
from cap_mechanistic_cifar.cifar_mech.checkpointing import capture, atomic_save, file_hash, digest, load, state_equal
from cap_mechanistic_cifar.cifar_mech.streams import IndexSampler
from cap_mechanistic_cifar.cifar_mech.data import Data
from cifar_update_probe import InstrumentedCAP
import experiment_cifar_diagnostics as runner


class DiagnosticsTests(unittest.TestCase):
    def test_exact_optimizer_parity_all_branches_both_architectures(self):
        torch.set_num_threads(1)
        for arch in ["mlp", "cnn"]:
            protocol.configure(arch, 4)
            for branch in runner.BRANCHES:
                torch.manual_seed(23)
                a = make_model()
                b = copy.deepcopy(a)
                oa = CAPAdam(parameter_groups(a), k=1e-5)
                for group in oa.param_groups:
                    for p in group["params"]:
                        if group["protect"]:
                            oa.state[p]["imp"].fill_(.1)
                ob = InstrumentedCAP(parameter_groups(b))
                ob.load_state_dict(copy.deepcopy(oa.state_dict()))
                apply(oa, {"kind": branch})
                apply(ob, {"kind": branch})
                for step in range(3):
                    for pa, pb in zip(a.parameters(), b.parameters()):
                        pa.grad = torch.randn_like(pa)
                        pb.grad = pa.grad.clone()
                    before = torch.get_rng_state().clone()
                    ob.measure = True
                    oa.step()
                    ob.step()
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    self.assertTrue(state_equal(a.state_dict(), b.state_dict()))
                    self.assertTrue(state_equal(oa.state_dict(), ob.state_dict()))
                    if branch == "box_off":
                        self.assertTrue(all(r["removed_norm"] == 0 for r in ob.measurements))

    def test_matched_batches_and_exact_midtask_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            j = ("mlp", 42, 20, "baseline")
            cp_path = root/"bank/mlp/scientific/reproduction/checkpoints/a_seed42_task20.pt"
            data = Data("unused", smoke=True)
            factory = lambda: torch.nn.Sequential(torch.nn.Linear(3072, 100))
            model = factory()
            config = runner.reference()["mlp"]["run_config.json"]["content"]
            opt = CAPAdam(parameter_groups(model), **config["baseline"])
            sampler = IndexSampler(len(data.train_y), 42)
            atomic_save(cp_path, capture(model, opt, sampler, task=20, seed=42, stream="a",
                        within_task_update=0, stream_metadata={"smoke": False},
                        source_version=config["source_version"], protocol=config["protocol"],
                        baseline_config_sha256=digest(config["baseline"])))
            refs = runner.reference()
            refs["mlp"]["reproduction_gate.json"]["content"]["checkpoint_hashes"][cp_path.name] = file_hash(cp_path)
            args = SimpleNamespace(output=root/"full", bank_root=root/"bank", device="cpu", resume=True, stop_after_updates=7)
            with patch.object(runner, "reference", return_value=refs), patch.object(runner, "make_model", side_effect=factory):
                runner.run_job(args, data, j, {"test": True})
                full = load(args.output/"jobs"/runner.job_id(j)/"resume.pt")
                args.output = root/"split"
                args.stop_after_updates = 3
                runner.run_job(args, data, j, {"test": True})
                args.stop_after_updates = 4
                runner.run_job(args, data, j, {"test": True})
                split = load(args.output/"jobs"/runner.job_id(j)/"resume.pt")
                for key in ["model", "optimizer", "sampler"]:
                    self.assertTrue(state_equal(full[key], split[key]), key)
                for key in ["python", "cpu", "cuda"]:
                    self.assertTrue(state_equal(full["rng"][key], split["rng"][key]), key)
                np.testing.assert_equal(full["rng"]["numpy"], split["rng"]["numpy"])
                for key in ["accuracies", "online", "traces", "retention"]:
                    self.assertEqual(full["record"][key], split["record"][key])
                args.output = root/"reset"
                args.stop_after_updates = 7
                runner.run_job(args, data, (*j[:3], "importance_reset"), {"test": True})
                reset = load(args.output/"jobs"/runner.job_id((*j[:3], "importance_reset"))/"resume.pt")
                self.assertEqual([r["batch_sha256"] for r in full["record"]["online"]],
                                 [r["batch_sha256"] for r in reset["record"]["online"]])


if __name__ == "__main__":
    unittest.main()
