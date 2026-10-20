from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from post_phase1_common import (
    EXPECTED_GRID_COUNTS,
    METHODS,
    POST_DIR,
    normalize_config,
    parse_manifest,
    render_manifest,
    retention_grids,
)

try:
    import torch
    from torch import nn

    from phase1_arms import build_arm
    from post_phase1_select import _extend_value, choose

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


class PostPhase1Tests(unittest.TestCase):
    def test_v2_has_an_isolated_output_root(self) -> None:
        self.assertEqual(POST_DIR.name, "post_phase1_v2")

    def test_pp1_a1_c_removes_result_inputs_from_v2_computation(self) -> None:
        root = Path(__file__).resolve().parent
        selection = (root / "post_phase1_select.py").read_text(encoding="utf-8")
        streams = (root / "post_phase1_streams.py").read_text(encoding="utf-8")
        grid = (root / "post_phase1_grid.py").read_text(encoding="utf-8")
        report = (root / "post_phase1_report.py").read_text(encoding="utf-8")
        for forbidden in ("PHASE1_DIR", "tuning_v5", "_primary_candidates"):
            self.assertNotIn(forbidden, selection)
        self.assertNotIn("CORRECTED_RESULTS", selection)
        for forbidden in ("PHASE1_DIR", "STREAM_A_DIR", "reused_frozen"):
            self.assertNotIn(forbidden, streams)
        self.assertNotIn("assert_required_frozen_evidence", grid)
        self.assertNotIn("PHASE1_DIR", report)
        self.assertNotIn("STREAM_A_DIR", report)
        forbidden_paths = (
            "results/phase1",
            "results/stream_a",
            "results/post_phase1_v1",
            "POST_V1_DIR",
            "verify_frozen_inputs",
            "frozen_guard",
        )
        for path in root.glob("post_*.py"):
            if path.name == Path(__file__).name:
                continue
            source = path.read_text(encoding="utf-8")
            for forbidden in forbidden_paths:
                self.assertNotIn(forbidden, source, f"{forbidden} in {path.name}")

    def test_registered_grid_counts_and_fixed_replay_capacity(self) -> None:
        grids = retention_grids()
        self.assertEqual(
            {method: len(grids[method]) for method in METHODS},
            EXPECTED_GRID_COUNTS,
        )
        self.assertEqual(sum(map(len, grids.values())), 114)
        self.assertEqual(
            {cell["buffer_capacity"] for cell in grids["adam_replay"]},
            {5_000},
        )
        self.assertEqual(
            {cell["replay_count"] for cell in grids["adam_replay"]},
            {32, 96},
        )
        for method in METHODS:
            self.assertEqual(
                {cell["lr"] for cell in grids[method]},
                {1e-4, 3e-4, 1e-3},
            )
            self.assertEqual(
                {cell["weight_decay"] for cell in grids[method]},
                {0.0, 1e-2},
            )

    def test_cap_liveness_amendment_is_tracked(self) -> None:
        root = Path(__file__).resolve().parent
        amendment = root / "post_phase1_amendment_POST-P1-A2.md"
        source = (root / "post_phase1_grid.py").read_text(encoding="utf-8")
        self.assertTrue(amendment.is_file())
        self.assertIn("prime_cap_state=True", source)
        self.assertIn('"state_priming_amendment"', source)
        self.assertIn('"POST-P1-A2"', source)

    @unittest.skipUnless(HAS_TORCH, "scientific PyTorch runtime unavailable")
    def test_cap_k_is_live_from_identical_primed_state(self) -> None:
        from post_phase1_grid import _liveness_distance

        low = {
            "lr": 1e-4,
            "weight_decay": 0.0,
            "k": 1e-5,
            "T_a": 2_000,
            "T_m": 50_000,
            "T_s": 2_000,
            "rho": 0.01,
            "c_min": 0.01,
            "beta_utility": 0.9999,
            "retention": "release",
        }
        high = dict(low, k=7e-5)
        self.assertGreater(
            _liveness_distance(
                "cap_release", low, high, 10, prime_cap_state=True
            ),
            0.0,
        )

    @unittest.skipUnless(HAS_TORCH, "scientific PyTorch runtime unavailable")
    def test_new_constructor_knobs_are_consumed_and_unknown_keys_fail(self) -> None:
        model = nn.Linear(4, 2)
        artifact = {
            "methods": {
                "adamw": {
                    "config": {
                        "lr": 1e-3,
                        "weight_decay": 0.0,
                    }
                }
            }
        }
        optimizer, _ = build_arm(
            "adamw", model, artifact, 42, torch.device("cpu")
        )
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)
        artifact["methods"]["adamw"]["config"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "unconsumed configuration key"):
            build_arm("adamw", model, artifact, 42, torch.device("cpu"))

    @unittest.skipUnless(HAS_TORCH, "scientific PyTorch runtime unavailable")
    def test_replay_protocol_cap_is_enforced(self) -> None:
        model = nn.Linear(4, 2)
        config = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "replay_count": 96,
            "buffer_capacity": 4_999,
        }
        with self.assertRaisesRegex(ValueError, "fixed protocol constant"):
            build_arm(
                "adam_replay",
                model,
                {"methods": {"adam_replay": {"config": config}}},
                42,
                torch.device("cpu"),
            )

    def test_normalization_adds_only_frozen_defaults(self) -> None:
        self.assertEqual(
            normalize_config("adamw", {"lr": 1e-3}),
            {"lr": 1e-3, "weight_decay": 0.01},
        )
        cap = normalize_config(
            "cap_ratchet", {"k": 3e-5, "T_a": 2_000}
        )
        self.assertEqual(cap["T_s"], 2_000)
        self.assertEqual(cap["weight_decay"], 0.01)
        self.assertEqual(cap["retention"], "ratchet")

    def test_manifest_is_sorted_and_excludes_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "MANIFEST.sha256").write_text("ignored", encoding="utf-8")
            rendered = render_manifest(root)
            manifest = root / "post_manifest.sha256"
            manifest.write_text(rendered, encoding="utf-8")
            parsed = parse_manifest(manifest)
            self.assertEqual(list(parsed), ["a.txt", "z.txt"])

    @unittest.skipUnless(HAS_TORCH, "scientific PyTorch runtime unavailable")
    def test_edge_extension_caps(self) -> None:
        self.assertEqual(
            _extend_value(
                "adam_replay", "replay_count", [32, 96], "high", []
            ),
            127,
        )
        self.assertIsNone(
            _extend_value(
                "adamw", "weight_decay", [0, 1e-2], "low", []
            )
        )
        self.assertIsNone(
            _extend_value("upgd", "sigma", [1e-4, 1e-2], "low", [])
        )

    @unittest.skipUnless(HAS_TORCH, "scientific PyTorch runtime unavailable")
    def test_v2_selection_maximizes_the_balanced_minimum(self) -> None:
        rows = [
            {
                "validation_retention_metric": 0.95,
                "validation_plasticity": 0.60,
                "selection_order": 0,
                "status": "OK",
                "diverged": False,
            },
            {
                "validation_retention_metric": 0.80,
                "validation_plasticity": 0.79,
                "selection_order": 1,
                "status": "OK",
                "diverged": False,
            },
        ]
        self.assertEqual(choose(rows)["selection_order"], 1)


if __name__ == "__main__":
    unittest.main()
