"""Protocol gates and genuine interrupted-versus-uninterrupted numerical tests."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import upgd_rebuttal as p


class ProtocolTests(unittest.TestCase):
    def test_grid_contains_incumbent_and_preserves_separation(self):
        plan = p.protocol()
        self.assertEqual(len(p.grid()), 64)
        self.assertEqual(len({p.digest(c) for c in p.grid()}), 64)
        self.assertIn(p.INCUMBENT, p.grid())
        self.assertFalse(set(plan["report_seeds"]) & {plan["search_seed"], plan["confirmation_seed"]})
        self.assertEqual(plan["updates_per_task"]*plan["search_tasks"], 200000)
        self.assertEqual(plan["updates_per_task"]*plan["eval_tasks"], 1000000)

    def test_plan_does_not_import_torch_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/"absent"
            code = ("import sys; from experiment_upgd_tuning import main; "
                    f"main(['plan','--output-dir',{str(destination)!r}]); "
                    "assert 'torch' not in sys.modules")
            subprocess.run([sys.executable, "-c", code], cwd=p.ROOT, check=True, capture_output=True)
            self.assertFalse(destination.exists())

    def test_manifest_changes_and_missing_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p.initialize(tmp, True)
            with self.assertRaises(RuntimeError):
                p.initialize(tmp, False)
            with self.assertRaises(RuntimeError):
                p.make_shortlist(tmp)
            value = p.read_json(Path(tmp)/"campaign.json")
            value["split_seed"] += 1
            p.atomic_json(Path(tmp)/"campaign.json", value)
            with self.assertRaises(RuntimeError):
                p.campaign(tmp)

    def test_lock_prevents_duplicate_worker_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp)/"worker.lock"
            with p.file_lock(lock):
                with self.assertRaises(RuntimeError):
                    with p.file_lock(lock):
                        pass
            with p.file_lock(lock):
                pass

    def test_timing_accounts_restarts_and_truncated_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/"timing.jsonl"
            for row in (dict(attempt="a", event="start", elapsed_seconds=0),
                        dict(attempt="a", event="checkpoint", elapsed_seconds=5)):
                p.append_event(log, row)
            with log.open("ab") as handle:
                handle.write(b'{"incomplete":')
            p.append_event(log, dict(attempt="b", event="complete", elapsed_seconds=3))
            observed = p.timing_summary(log)
            self.assertEqual(observed["observed_wall_seconds"], 8)
            self.assertEqual(observed["interrupted_attempts"], 1)

    def test_selection_averages_seed_balanced_scores_not_averaged_p_and_r(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = p.initialize(tmp, True)
            p.atomic_json(Path(tmp)/"shortlist.json", {})
            confirm_jobs = [p.job_spec(plan, "confirm", i) for i in (0, 1, 2)]
            def fake_result(output, job):
                index = job["config_index"]
                primary = job["stage"] == "search"
                pv, rv = ((1., .4) if primary else (.4, 1.)) if index == 0 else (.6, .6)
                return dict(job=job, status="diverged" if index == 2 else "complete",
                            metrics=dict(plasticity=pv, retention=rv, balanced=min(pv, rv)))
            with patch.object(p, "jobs", return_value=confirm_jobs), patch.object(p, "result", side_effect=fake_result):
                selected = p.choose(tmp, write=False)
            self.assertEqual(selected["config_index"], 1)
            self.assertEqual(selected["excluded_confirmation_indices"], [2])
            self.assertEqual(selected["balanced"], .6)


try:
    import torch
    import upgd_rebuttal_runtime as runtime
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Numerical tests require PyTorch")
class RuntimeTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        elif hasattr(a, "shape"):
            self.assertTrue((a == b).all())
        else:
            self.assertEqual(a, b)

    def run_worker(self, root, job, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return runtime.run_job(root, job, data_dir=Path(root)/"unused", device="cpu", checkpoint_every=3, **kw)

    def checkpoint(self, root, job):
        return torch.load(Path(root)/"jobs"/job["id"]/"latest.pt", weights_only=False, map_location="cpu")

    def test_split_is_stratified_disjoint_and_repeatable(self):
        labels = torch.arange(100) % 10
        train, val = runtime.split_indices(labels, 20, 777)
        self.assertEqual(len(set(train.tolist()) & set(val.tolist())), 0)
        self.assertEqual(len(train)+len(val), 100)
        self.assertTrue(torch.equal(torch.bincount(labels[val]), torch.full((10,), 2)))
        self.assertTrue(torch.equal(val, runtime.split_indices(labels, 20, 777)[1]))

    def test_resume_mid_task_preserves_utility_rng_sampler_and_metrics(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as resumed:
            p.initialize(full, True)
            p.initialize(resumed, True)
            job = p.jobs(full, "search")[p.grid().index(p.INCUMBENT)]
            completed = self.run_worker(full, job)
            paused = self.run_worker(resumed, job, stop_after_updates=5)
            self.assertEqual(paused["step"], 5)
            actual = self.run_worker(resumed, job, resume=True)
            self.assertEqual(completed["metrics"], actual["metrics"])
            left, right = self.checkpoint(full, job), self.checkpoint(resumed, job)
            for key in ("model", "optimizer", "sampler", "rng"):
                self.assert_nested_equal(left[key], right[key])
            for name in ("scalars", "tasks", "pool"):
                self.assertEqual(left["progress"][name], right["progress"][name])
            self.assertEqual(actual["timing"]["attempts"], 2)
            log = Path(resumed)/"jobs"/job["id"]/"timing.jsonl"
            before = log.read_bytes()
            self.run_worker(resumed, job, resume=True)
            self.assertEqual(before, log.read_bytes())
            # Final checkpoint is enough to recover a crash before result publication.
            (Path(resumed)/"jobs"/job["id"]/"result.json").unlink()
            recovered = self.run_worker(resumed, job, resume=True)
            self.assertEqual(recovered["metrics"], completed["metrics"])

    def test_error_during_evaluation_replays_only_uncommitted_updates(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as interrupted:
            p.initialize(full, True)
            p.initialize(interrupted, True)
            job = p.jobs(full, "search")[0]
            self.run_worker(full, job)
            with patch.object(runtime, "evaluate", side_effect=RuntimeError("injected evaluation interruption")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self.run_worker(interrupted, job)
            self.assertEqual(self.checkpoint(interrupted, job)["progress"]["step"], 3)
            self.run_worker(interrupted, job, resume=True)
            a, b = self.checkpoint(full, job), self.checkpoint(interrupted, job)
            for key in ("model", "optimizer", "sampler", "rng"):
                self.assert_nested_equal(a[key], b[key])

    def test_job_tampering_and_output_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p.initialize(tmp, True)
            job = p.jobs(tmp, "search")[0]
            bad = copy.deepcopy(job)
            bad["updates"] -= 1
            with self.assertRaises(RuntimeError):
                self.run_worker(tmp, bad)
            self.run_worker(tmp, job)
            path = Path(tmp)/"jobs"/job["id"]/"pool.csv"
            path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "artifact"):
                p.result(tmp, job)

    def test_numerical_divergence_is_a_terminal_failure_not_a_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            p.initialize(tmp, True)
            job = p.jobs(tmp, "search")[0]
            original_factory = runtime.make_optimizer
            def corrupting_factory(model, spec):
                optimizer = original_factory(model, spec)
                original_step = optimizer.step
                def step():
                    original_step()
                    with torch.no_grad():
                        next(model.parameters()).fill_(float("nan"))
                optimizer.step = step
                return optimizer
            with patch.object(runtime, "make_optimizer", side_effect=corrupting_factory):
                observed = self.run_worker(tmp, job)
            self.assertEqual(observed["status"], "diverged")
            self.assertEqual(observed["metrics"], {})
            self.assertLess(observed["step"], job["updates"])
            self.assertEqual(p.result(tmp, job)["status"], "diverged")

    def test_full_synthetic_selection_and_evaluation_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = p.initialize(tmp, True)
            for job in p.jobs(tmp, "search"):
                self.run_worker(tmp, job)
            shortlist = p.make_shortlist(tmp)
            self.assertIn(p.grid().index(p.INCUMBENT), shortlist["indices"])
            self.assertLessEqual(len(shortlist["indices"]), 4)
            for job in p.jobs(tmp, "confirm"):
                self.run_worker(tmp, job)
            selected = p.choose(tmp)
            self.assertIn(selected["config_index"], shortlist["indices"])
            for job in p.jobs(tmp, "reference"):
                self.run_worker(tmp, job)
            for job in p.jobs(tmp, "eval"):
                self.run_worker(tmp, job)
                if job["stream"] == "b":
                    for task in plan["evaluation_checkpoint_tasks"]:
                        cp = torch.load(Path(tmp)/"jobs"/job["id"]/f"task_{task:03d}.pt", weights_only=False)
                        self.assertEqual(cp["progress"]["step"], task*plan["updates_per_task"])
                        self.assertTrue(all("avg_utility" in s for s in cp["optimizer"]["state"].values()))
            report = p.evaluation_report(tmp, ["selected"])
            self.assertEqual(report[0]["seeds"], 5)
            p.tuning_report(tmp)
            self.assertTrue((Path(tmp)/"analysis/search_budget_curve.csv").exists())
            # The committed final rows must obey their coverage contract.
            job = p.jobs(tmp, "eval")[0]
            state = self.checkpoint(tmp, job)["progress"]
            state["scalars"].pop()
            with self.assertRaises(RuntimeError):
                runtime.validate_rows(plan, job, state)
            b_job = next(j for j in p.jobs(tmp, "eval") if j["stream"] == "b")
            (Path(tmp)/"jobs"/b_job["id"]/"task_002.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                p.result(tmp, b_job)


if __name__ == "__main__":
    unittest.main()
