from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import analyze_stream_horizon_plasticity as horizon


def _complete_values(offset: float = 0.0) -> dict[int, dict[int, float]]:
    return {
        seed: {
            step: offset + seed * 1e-7 + step * 1e-9
            for step in range(
                horizon.SCALAR_LOG_EVERY,
                horizon.HORIZONS[-1] + 1,
                horizon.SCALAR_LOG_EVERY,
            )
        }
        for seed in horizon.FINAL_SEEDS
    }


class StreamHorizonPlasticityTests(unittest.TestCase):
    def test_horizon_set_and_exact_open_closed_windows(self) -> None:
        self.assertEqual(
            horizon.HORIZONS,
            (100_000, 250_000, 500_000, 750_000, 1_000_000),
        )
        rows = horizon.horizon_plasticity(_complete_values(), "cap")
        for row in rows:
            expected_count = int(row["horizon"]) // 2 // horizon.SCALAR_LOG_EVERY
            self.assertEqual(row["window_start"], int(row["horizon"]) // 2)
            self.assertEqual(row["window_end"], row["horizon"])
            self.assertEqual(row["observation_count"], expected_count)

        row = next(
            row
            for row in rows
            if row["seed"] == horizon.FINAL_SEEDS[0]
            and row["horizon"] == 100_000
        )
        selected_steps = range(50_100, 100_001, 100)
        expected = sum(
            horizon.FINAL_SEEDS[0] * 1e-7 + step * 1e-9
            for step in selected_steps
        ) / 500
        self.assertAlmostEqual(row["plasticity"], expected)

    def test_loader_deduplicates_identical_rows_and_rejects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stream_scalars.csv"
            fields = ("arm", "seed", "step", "batch_accuracy")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {"arm": "cap", "seed": 42, "step": 100, "batch_accuracy": 0.5}
                )
                writer.writerow(
                    {"arm": "cap", "seed": 42, "step": 100, "batch_accuracy": 0.5}
                )
            loaded = horizon.load_trajectory(
                horizon.MethodSpec("cap", "CAP", path, "cap")
            )
            self.assertEqual(loaded.duplicate_rows_removed, 1)
            self.assertEqual(loaded.values[42], {100: 0.5})

            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(
                    {"arm": "cap", "seed": 42, "step": 100, "batch_accuracy": 0.6}
                )
            with self.assertRaisesRegex(RuntimeError, "conflicting duplicate"):
                horizon.load_trajectory(
                    horizon.MethodSpec("cap", "CAP", path, "cap")
                )

    def test_coverage_requires_all_steps_and_final_seeds(self) -> None:
        loaded = horizon.LoadedTrajectory(
            horizon.MethodSpec("cap", "CAP", Path("unused"), "cap"),
            Path("unused"),
            _complete_values(),
            0,
        )
        horizon.validate_trajectory(loaded)

        missing_seed = _complete_values()
        del missing_seed[horizon.FINAL_SEEDS[-1]]
        loaded.values = missing_seed
        with self.assertRaisesRegex(RuntimeError, "seed mismatch"):
            horizon.validate_trajectory(loaded)

        missing_step = _complete_values()
        del missing_step[horizon.FINAL_SEEDS[0]][100_000]
        loaded.values = missing_step
        with self.assertRaisesRegex(RuntimeError, "incomplete/noncanonical"):
            horizon.validate_trajectory(loaded)

    def test_pairing_and_delta_summary(self) -> None:
        cap_rows = horizon.horizon_plasticity(_complete_values(0.2), "cap")
        upgd_rows = horizon.horizon_plasticity(_complete_values(0.1), "upgd")
        deltas = horizon.paired_deltas([*cap_rows, *upgd_rows])
        self.assertEqual(len(deltas), len(horizon.FINAL_SEEDS) * len(horizon.HORIZONS))
        for row in deltas:
            self.assertAlmostEqual(row["delta_cap_minus_upgd"], 0.1)
            self.assertAlmostEqual(row["mean_paired_delta"], 0.1)

        with self.assertRaisesRegex(RuntimeError, "pairing is incomplete"):
            horizon.paired_deltas([*cap_rows, *upgd_rows[:-1]])

    def test_one_million_equivalence_is_per_seed_and_mean_checked(self) -> None:
        cap_rows = horizon.horizon_plasticity(_complete_values(0.2), "cap")
        upgd_rows = horizon.horizon_plasticity(_complete_values(0.1), "upgd")
        rows = [*cap_rows, *upgd_rows]
        final_rows = [row for row in rows if row["horizon"] == 1_000_000]
        reference = {"cap": {}, "upgd": {}}
        for row in final_rows:
            reference[str(row["method"])][int(row["seed"])] = float(row["plasticity"])
        expected_means = {
            method: sum(values.values()) / len(values)
            for method, values in reference.items()
        }
        observed = horizon.validate_one_million_equivalence(
            rows, reference, expected_means=expected_means
        )
        for method in ("cap", "upgd"):
            self.assertAlmostEqual(observed[method], expected_means[method])

        reference["cap"][horizon.FINAL_SEEDS[0]] += 0.01
        with self.assertRaisesRegex(RuntimeError, "1M metric mismatch"):
            horizon.validate_one_million_equivalence(
                rows, reference, expected_means=expected_means
            )

    def test_prefix_equivalence_is_verified_from_current_runner(self) -> None:
        metadata = horizon.prefix_equivalence_metadata()
        self.assertTrue(metadata["verified"])
        self.assertTrue(metadata["reused_1m_prefixes"])
        self.assertFalse(metadata["dedicated_shorter_runs_required"])
        self.assertTrue(all(metadata["checks"].values()))

    def test_analysis_writes_all_artifacts_from_existing_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cap_dir = root / "cap"
            upgd_dir = root / "upgd"
            output_dir = root / "analysis"
            cap_dir.mkdir()
            upgd_dir.mkdir()
            fields = ("arm", "seed", "step", "batch_accuracy")
            for directory, arm, accuracy in (
                (cap_dir, "cap_arm", horizon.EXPECTED_FINAL_MEANS["cap"]),
                (upgd_dir, "upgd_arm", horizon.EXPECTED_FINAL_MEANS["upgd"]),
            ):
                with (directory / "stream_scalars.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for seed in horizon.FINAL_SEEDS:
                        for step in range(
                            horizon.SCALAR_LOG_EVERY,
                            horizon.HORIZONS[-1] + 1,
                            horizon.SCALAR_LOG_EVERY,
                        ):
                            writer.writerow(
                                {
                                    "arm": arm,
                                    "seed": seed,
                                    "step": step,
                                    "batch_accuracy": accuracy,
                                }
                            )
            final_metrics = root / "final_metrics.csv"
            with final_metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("method", "seed", "plasticity")
                )
                writer.writeheader()
                for method, accuracy in (
                    ("cap_tuned", horizon.EXPECTED_FINAL_MEANS["cap"]),
                    ("upgd", horizon.EXPECTED_FINAL_MEANS["upgd"]),
                ):
                    for seed in horizon.FINAL_SEEDS:
                        writer.writerow(
                            {
                                "method": method,
                                "seed": seed,
                                "plasticity": accuracy,
                            }
                        )

            args = horizon.parse_args(
                [
                    "--cap-dir",
                    str(cap_dir),
                    "--upgd-dir",
                    str(upgd_dir),
                    "--cap-arm",
                    "cap_arm",
                    "--upgd-arm",
                    "upgd_arm",
                    "--final-metrics",
                    str(final_metrics),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            def fake_plot(_rows, path: Path) -> None:
                path.write_bytes(b"figure")

            with mock.patch.object(horizon, "_plot_primary", side_effect=fake_plot), mock.patch.object(
                horizon, "_plot_delta", side_effect=fake_plot
            ):
                horizon.run(args)

            expected = {
                "horizon_plasticity_per_seed.csv",
                "horizon_plasticity_summary.csv",
                "cap_vs_upgd_horizon_deltas.csv",
                "stream_horizon_plasticity.png",
                "stream_horizon_delta.png",
                "horizon_analysis_metadata.json",
                "horizon_analysis.md",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)


if __name__ == "__main__":
    unittest.main()
