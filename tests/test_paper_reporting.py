"""Check paper reporting semantics using synthetic scalar records, without torch."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import experiment_paper_benchmark as benchmark
import experiment_paper_development as development


class PaperReportingTests(unittest.TestCase):
    def test_fixed_paper_settings(self):
        self.assertEqual(benchmark.CONFIGS["cap"]["k"], 7e-5)
        self.assertEqual(benchmark.CONFIGS["cap"]["release_update"], "contest_blocked")
        self.assertEqual(benchmark.CONFIGS["upgd"]["lr"], 1e-3)
        self.assertEqual(benchmark.CONFIGS["adam_replay"]["replay_count"], 96)

    def test_gap_excludes_current_and_never_seen_domains(self):
        rows = []
        for task, current in ((1, 0), (2, 1), (3, 0)):
            for domain in range(3):
                rows.append({"seed": 42, "task": task, "pool_index": domain,
                             "is_current": domain == current, "accuracy": [.7, .5, .1][domain]})
        result = development.gap_rows(rows, "test", 42)
        self.assertEqual(result[0]["observations"], 2)
        self.assertAlmostEqual(result[0]["retention"], .6)

    def test_complete_benchmark_aggregates_per_seed_min_and_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scalars = [{"seed": seed, "step": step,
                        "batch_accuracy": (.8 if step <= 500000 else .6) if seed==42 else .9}
                       for seed in (42,123) for step in range(100, 1000001, 100)]
            pool = [{"seed": seed, "task": task, "pool_index": domain,
                     "is_current": domain == (task-1)%40,
                     "accuracy": .99 if domain == (task-1)%40 else (.7 if seed==42 else .5)}
                    for seed in (42,123) for task in range(1, 201) for domain in range(40)]
            development.write(root/"cap/stream_a/stream_scalars.csv", scalars)
            development.write(root/"cap/stream_b/streamb_pool.csv", pool)
            benchmark.analyze(root, ["cap"], [42,123])
            result = development.read(root/"analysis/per_seed_metrics.csv")[0]
            self.assertAlmostEqual(float(result["plasticity"]), .6)
            self.assertAlmostEqual(float(result["retention"]), .7)
            self.assertAlmostEqual(float(result["balanced"]), .6)
            horizons = development.read(root/"analysis/horizon_plasticity.csv")
            self.assertAlmostEqual(float(horizons[0]["plasticity"]), .8)
            self.assertEqual(len(horizons), 10)
            summary=development.read(root/"analysis/table03_benchmark.csv")[0]
            self.assertAlmostEqual(float(summary["balanced"]), .55)

    def test_development_windows_and_within_task_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            scalars=[{"seed":42,"step":step,"batch_accuracy":step/1000000,
                      "bind_rate":.2,"median_imp":.01,"median_radius":.02}
                     for step in range(100,1000001,100)]
            pool=[{"seed":42,"task":task,"pool_index":domain,
                   "is_current":domain==(task-1)%40,"accuracy":.7}
                  for task in range(1,201) for domain in range(40)]
            development.write(root/"final/stream_a/stream_scalars.csv",scalars)
            development.write(root/"final/stream_b/streamb_pool.csv",pool)
            development.analyze(root,["final"],[42])
            temporal=development.read(root/"analysis/table21_temporal.csv")
            self.assertEqual(len(temporal),5)
            self.assertAlmostEqual(float(temporal[0]["batch_accuracy"]),.05005)
            within=development.read(root/"analysis/table23_within_task.csv")
            self.assertEqual([int(r["within_task_update"]) for r in within],[100,300,1000,2000,3000,5000])

    def test_every_paper_target_has_a_local_pipeline_or_protocol(self):
        root=Path(benchmark.__file__).parent
        targets=json.loads((root/"paper_targets.json").read_text())["targets"]
        self.assertEqual(len(targets),34)
        for name,target in targets.items():
            self.assertIn(target["status"],{"data-pipeline","protocol"},name)
            self.assertTrue(target["artifacts"],name)
            self.assertTrue(target["commands"] or target["status"]=="protocol",name)
            for command in target["commands"]:
                if command[0]!="-c":self.assertTrue((root/command[0]).is_file(),(name,command))

    def test_incomplete_trajectory_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            development.write(root/"cap/stream_a/stream_scalars.csv", [{"seed":42,"step":100,"batch_accuracy":.8}])
            development.write(root/"cap/stream_b/streamb_pool.csv", [{"seed":42,"task":1,"pool_index":0,"is_current":True,"accuracy":.8}])
            with self.assertRaises(ValueError):
                benchmark.analyze(root, ["cap"], [42])
