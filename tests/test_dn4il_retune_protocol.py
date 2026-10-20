"""Selection/isolation tests runnable without PyTorch: python -m unittest ..."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dn4il_retune_protocol import (
    BASE_VERSION, DEVELOPMENT_SEEDS, FINAL_SEEDS, METHODS, grid, center,
    candidate_summary, select, freeze_protocol, validate_output_path, read_json, write_json,
)
from experiment_dn4il_retune import parse_args, require_liveness


class RetuneProtocolTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "original"
        write_json(source / "manifest/dn4il_manifest.json", {"manifest_hash": "data"})
        write_json(source / "qualification/qualification.json", {"qualified": True})
        for seed in (*DEVELOPMENT_SEEDS, *FINAL_SEEDS):
            write_json(source / "schedules" / f"seed_{seed}.json", {"seed": seed})
        for method in ("adamw", "ewc"):
            write_json(source / "selected_configs" / f"{method}.json", {})
            for seed in FINAL_SEEDS:
                folder = source / "train" / method / f"seed_{seed}"
                write_json(folder / "run_config.json", {"experiment_version": BASE_VERSION,
                    "method": method, "batch_size": 128, "updates_per_segment": 200, "max_segments": 30})
                for name in ("stream_segments.csv", "online_steps.csv", "domain_eval.csv", "optimizer_diagnostics.csv"):
                    (folder / name).write_text("fixture\n")
                (folder / "checkpoint_final.pt").write_bytes(b"size fixture")
        return source

    def test_mnist_style_grids_have_equal_budgets_and_expected_values(self):
        for method in METHODS:
            configs = grid(method)
            self.assertEqual(len(configs), 36)
            self.assertIn(center(method), configs)
            self.assertEqual(len({tuple(sorted(c.items())) for c in configs}), 36)
        self.assertIn(dict(lr=.03, sigma=.01, beta_utility=.9999, weight_decay=1e-4), grid("upgd"))
        self.assertEqual({c["T_s"] for c in grid("cap")}, {2000})
        self.assertEqual({c["T_a"] for c in grid("cap")}, {2000, 10000})
        self.assertEqual({c["T_m"] for c in grid("cap")}, {10000, 50000})
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS))

    def test_maximum_of_mean_per_seed_min_not_min_of_means(self):
        a = candidate_summary(grid("cap")[0], [{"P_scale": .9, "R_scale": .4}, {"P_scale": .4, "R_scale": .9}])
        b = candidate_summary(grid("cap")[1], [{"P_scale": .5, "R_scale": .5}] * 2)
        self.assertAlmostEqual(a["mean_metrics"]["B_scale"], .4)
        self.assertIs(select([a, b], "cap")[0], b)

    def test_tie_prefers_lower_variability_then_central(self):
        a = candidate_summary(grid("cap")[0], [{"P_scale": .4, "R_scale": .4}, {"P_scale": .6, "R_scale": .6}])
        b = candidate_summary(center("cap"), [{"P_scale": .5, "R_scale": .5}] * 2)
        self.assertIs(select([a, b], "cap")[0], b)
        a = candidate_summary(grid("cap")[0], [{"P_scale": .5, "R_scale": .5}] * 2)
        self.assertIs(select([a, b], "cap")[0], b)

    def test_invalid_metrics_are_not_selected(self):
        with self.assertRaises(ValueError):
            candidate_summary({}, [{"P_scale": .8, "R_scale": .5}])
        for bad in (float("nan"), float("inf"), -1, 1.1):
            with self.assertRaises(ValueError):
                candidate_summary({}, [{"P_scale": bad, "R_scale": .5}] * 2)

    def test_freeze_reuses_original_files_and_rejects_source_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = self.fixture(root); output = root / "new"
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            p = freeze_protocol(source, output)
            self.assertEqual(p, freeze_protocol(source, output))
            self.assertEqual(p, read_json(output / "protocol.json"))
            self.assertEqual(before, {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()})
            (source / "train/adamw/seed_42/domain_eval.csv").write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                freeze_protocol(source, output)

    def test_output_cannot_overlap_original(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for source, output in ((root, root), (root, root / "new"), (root / "old", root)):
                with self.assertRaises(ValueError):
                    validate_output_path(source, output)
            validate_output_path(root / "old", root / "new")

    def test_cli_cannot_train_baselines_or_tune_on_reporting_seeds(self):
        for argv in (["--train", "--method", "adamw"], ["--tune", "--seeds", "42"], ["--smoke", "--all-seeds"]):
            with self.assertRaises(SystemExit):
                parse_args(argv)
        self.assertTrue(parse_args(["--train", "--all-methods", "--all-seeds"]).train)

    def test_learning_gate_must_pass_with_the_same_code(self):
        with tempfile.TemporaryDirectory() as name:
            args = SimpleNamespace(output_dir=Path(name))
            protocol = {"protocol_hash": "fixture"}
            study = {"code_hashes": {"optimizer": "first"}}
            with self.assertRaisesRegex(RuntimeError, "--smoke"):
                require_liveness(args, protocol, study)
            gate = {**protocol, "study_protocol": study, "passed": False}
            path = args.output_dir / "smoke/upgd_liveness.json"
            write_json(path, gate)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                require_liveness(args, protocol, study)
            gate["passed"] = True
            write_json(path, gate)
            require_liveness(args, protocol, study)
            with self.assertRaisesRegex(RuntimeError, "protocol/code"):
                require_liveness(args, protocol, {"code_hashes": {"optimizer": "changed"}})


if __name__ == "__main__":
    unittest.main()
