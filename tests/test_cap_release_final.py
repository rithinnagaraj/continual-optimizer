from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cap_release_final as final


class CapReleaseFinalTests(unittest.TestCase):
    def test_grid_is_bounded_and_changes_only_k_and_tm(self) -> None:
        grid = final._grid()
        self.assertEqual(len(grid), 9)
        self.assertEqual({config["k"] for config in grid}, set(final.K_GRID))
        self.assertEqual({config["T_m"] for config in grid}, set(final.TM_GRID))

        center = final._full_cap_config(final.CENTER)
        corner = final._full_cap_config(grid[0])
        changed = {key for key in center if center[key] != corner[key]}
        self.assertEqual(changed, {"k", "T_m"})
        self.assertEqual(center["release_update"], "contest_blocked")
        self.assertEqual(center["momentum_response"], "none")
        self.assertEqual(center["box_geometry"], "coordinate")
        self.assertEqual(center["anchor_update"], "moving")
        self.assertEqual(center["importance_evidence"], "raw_g_squared")
        self.assertEqual(center["weight_decay"], 0.0)
        with self.assertRaises(ValueError):
            final.complete_cap_config({"lr": 1e-3})

    def test_confirmation_seeds_are_development_only(self) -> None:
        self.assertFalse(set(final.CONFIRMATION_SEEDS) & set(final.FINAL_SEEDS))
        self.assertEqual(
            final.DEVELOPMENT_SEEDS,
            (final.PRIMARY_DEVELOPMENT_SEED, *final.CONFIRMATION_SEEDS),
        )

    def test_cli_modes_and_final_seed_guard(self) -> None:
        with self.assertRaises(SystemExit):
            final.parse_args([])
        with self.assertRaises(SystemExit):
            final.parse_args(["--retune", "--train"])
        with self.assertRaises(SystemExit):
            final.parse_args(["--retune", "--seeds", "42"])
        with self.assertRaises(SystemExit):
            final.parse_args(["--train", "--seeds", "42"])

        reordered = [str(seed) for seed in reversed(final.FINAL_SEEDS)]
        args = final.parse_args(["--train", "--seeds", *reordered])
        self.assertTrue(args.train)

    def test_retune_config_hash_and_preregistered_rules(self) -> None:
        payload = final._retune_config()
        claimed = payload.pop("config_hash")
        self.assertEqual(claimed, final._hash_payload(payload))
        self.assertEqual(payload["grid"], final._grid())
        self.assertIn("prefer center configuration", payload["tie_breaks"])

    def test_pareto_filter_requires_a_strict_improvement(self) -> None:
        points = {
            "balanced": (0.8, 0.8),
            "plastic": (0.9, 0.7),
            "dominated": (0.7, 0.7),
            "duplicate": (0.8, 0.8),
        }
        self.assertEqual(
            final._pareto_ids(points), {"balanced", "plastic", "duplicate"}
        )

    def test_selected_config_is_hash_checked_and_grid_locked(self) -> None:
        selected_config = dict(final.CENTER)
        payload = {
            "experiment_version": final.EXPERIMENT_VERSION,
            "selected_config_id": final._config_id(selected_config),
            "selected_k": selected_config["k"],
            "selected_T_m": selected_config["T_m"],
            "cap_config": final._full_cap_config(selected_config),
            "development_selection_score": 0.5,
            "development_seeds": list(final.DEVELOPMENT_SEEDS),
            "confirmation_seeds": list(final.CONFIRMATION_SEEDS),
            "final_reporting_seeds": list(final.FINAL_SEEDS),
            "selection_rank": 1,
            "boundary_selected": False,
            "protocol_hash": final.protocol_hash(),
        }
        payload["config_hash"] = final._hash_payload(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selected_config.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            loaded = final._load_selected(root)
            self.assertEqual(loaded["selected_config_id"], final._config_id(final.CENTER))

            tampered = dict(payload)
            tampered["selected_k"] = 123.0
            (root / "selected_config.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                final._load_selected(root)


if __name__ == "__main__":
    unittest.main()
