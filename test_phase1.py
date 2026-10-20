"""Focused protocol tests for the Phase-1 implementation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import experiment_e4b
import experiment_stream_b
from experiment_e5 import DRAIN_STEPS, grade_drain, grade_sanity
from experiment_e6 import ARMS as E6_ARMS
from experiment_e6 import REPEATS as E6_REPEATS
from experiment_e6 import WORKLOADS as E6_WORKLOADS
from experiment_e6 import grade as grade_e6
from experiment_e6 import summarize as summarize_e6
from experiment_stream_b import revisit_gap_distribution, stream_b_schedule
from phase1_arms import (
    CONDITIONAL_RETUNE_ARM,
    FINAL_STREAM_AMENDMENT,
    PHASE1_AMENDMENT,
    P1_A7_PRIMARY_CAP_CONFIG,
    P1_A7_RETUNE_CAP_CONFIG,
    RETENTION_FOCUSED_LABEL,
    UPGD_DEFAULTS,
    build_arm,
    l2_descope_guard,
    load_conditional_retune_request,
    load_tuned_configs,
    cap_release_matches_core,
    upgd_provenance,
    validate_p1_a7_artifact,
)
from phase1_retention_validation import build_v7_artifact, validation_schedule
from phase1_grade import conditional_retune_request, grade_phase1
from phase1_tuning import (
    PRE_AMENDMENT_SMOKE,
    preserve_pre_amendment_evidence,
    registered_grids,
)
from phase1_stream_common import training_step
from cap_adam_experimental import CAPAdam


def tuned_artifact() -> dict:
    endpoint_data = {
        "adamw": (
            1,
            {"lr": 1e-3},
            0.9775945964,
            0.9130625,
            0,
            {"lr": 3e-4},
            0.9775393490,
            0.96585,
        ),
        "adam_replay": (
            2,
            {"lr": 3e-4, "replay_count": 64},
            0.98503,
            0.96,
            1,
            {"lr": 3e-4, "replay_count": 32},
            0.98,
            0.9743,
        ),
        "upgd": (
            10,
            {
                "lr": 0.03,
                "weight_decay": 0.01,
                "sigma": 0.01,
                "beta_utility": 0.9999,
            },
            0.9272321875,
            0.2261,
            4,
            {
                "lr": 0.01,
                "weight_decay": 0.01,
                "sigma": 0.1,
                "beta_utility": 0.9999,
            },
            0.9075823047,
            0.29750625,
        ),
        "ewc_oracle": (
            0,
            {"lambda": 1e3, "gamma": 0.95},
            0.869732,
            0.9748875,
            0,
            {"lambda": 1e3, "gamma": 0.95},
            0.869732,
            0.9748875,
        ),
        "l2_init": (
            0,
            {"lambda": 1e-4},
            0.983879,
            0.2773375,
            0,
            {"lambda": 1e-4},
            0.983879,
            0.2773375,
        ),
        "cap_release": (
            4,
            dict(P1_A7_PRIMARY_CAP_CONFIG),
            0.960763,
            0.94,
            6,
            dict(P1_A7_RETUNE_CAP_CONFIG),
            0.95,
            0.96855,
        ),
    }
    methods = {}
    sensitivity = {}
    for method, values in endpoint_data.items():
        (
            p_index,
            p_config,
            p_plasticity,
            p_retention,
            r_index,
            r_config,
            r_plasticity,
            r_retention,
        ) = values
        tried = [
            {
                "config_index": p_index,
                "config": dict(p_config),
                "plasticity_metric": p_plasticity,
                "validation_retention_metric": p_retention,
            }
        ]
        if r_index != p_index:
            tried.append(
                {
                    "config_index": r_index,
                    "config": dict(r_config),
                    "plasticity_metric": r_plasticity,
                    "validation_retention_metric": r_retention,
                }
            )
        methods[method] = {
            "config": dict(p_config),
            "selection_metric": p_plasticity,
            "winner_config_index": p_index,
            "primary": {
                "config": dict(p_config),
                "plasticity_metric": p_plasticity,
                "tie_break_metric": 0.0,
                "config_index": p_index,
            },
            "retention_focused": {
                "config": dict(r_config),
                "plasticity_metric": r_plasticity,
                "validation_retention_metric": r_retention,
                "config_index": r_index,
            },
            "endpoints_coincide": p_config == r_config,
            "tried_configs": tried,
        }
        if p_config != r_config:
            arm = method + "_retention_focused"
            sensitivity[arm] = {
                "method": method,
                "endpoint": "retention_focused",
                "label": RETENTION_FOCUSED_LABEL,
                "seeds": (
                    [42, 123, 2026, 451, 789]
                    if method == "cap_release"
                    else [42, 123, 2026]
                ),
                "config": dict(r_config),
            }
    artifact = {"methods": methods}
    artifact.update(
        {
            "schema_version": 7,
            "amendment": PHASE1_AMENDMENT,
            "amendments": ["P1-A3", "P1-A4", "P1-A5"],
            "sensitivity_arms": sensitivity,
            "kill_gate_retune_arm": CONDITIONAL_RETUNE_ARM,
            "stream_a_release_arm": "cap_release_tuned",
            "upgd": upgd_provenance(),
            "pre_amendment_upgd_smoke_gate": PRE_AMENDMENT_SMOKE,
        }
    )
    cap = dict(P1_A7_PRIMARY_CAP_CONFIG)
    artifact["derived_configs"] = {
        "cap_ratchet": {**cap, "retention": "ratchet"},
        "cap_leak": {**cap, "retention": "leak"},
    }
    return artifact


def grading_rows(
    *,
    primary_stream_a: float = 0.80,
    primary_stream_b: float = 0.40,
    include_conditional: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Construct the exact five-seed rows consumed by the P1-A7 grader."""
    stream_a_values = {
        "cap_release_tuned": primary_stream_a,
        "adamw": 0.81,
        "upgd": 0.79,
        "cap_leak": 0.82,
    }
    if include_conditional:
        stream_a_values[CONDITIONAL_RETUNE_ARM] = 0.81
    scalar_rows = [
        {
            "arm": arm,
            "seed": seed,
            "step": step,
            "batch_accuracy": accuracy,
        }
        for arm, accuracy in stream_a_values.items()
        for seed in (42, 123, 2026, 451, 789)
        for step in range(500_100, 1_000_001, 100)
    ]

    retention = {
        "adamw": 0.20,
        "adam_replay": 0.50,
        "upgd": 0.25,
        "ewc_oracle": 0.35,
        "cap_release": primary_stream_b,
        "cap_ratchet": 0.38,
        "cap_leak": 0.22,
    }
    if include_conditional:
        retention[CONDITIONAL_RETUNE_ARM] = 0.40
    pool_rows = [
        {
            "arm": arm,
            "seed": seed,
            "task": 200,
            "pool_index": pool_index,
            "accuracy": accuracy,
            "is_current": pool_index == 0,
        }
        for arm, accuracy in retention.items()
        for seed in (42, 123, 2026, 451, 789)
        for pool_index in range(10)
    ]
    return scalar_rows, pool_rows


class Phase1ProtocolTests(unittest.TestCase):
    def test_bind_diagnostics_flag_is_equivalent_and_omits_state(self) -> None:
        base = nn.Linear(4, 2)
        enabled_model = nn.Linear(4, 2)
        disabled_model = nn.Linear(4, 2)
        enabled_model.load_state_dict(base.state_dict())
        disabled_model.load_state_dict(base.state_dict())
        enabled = CAPAdam(
            enabled_model.parameters(), k=3e-5, track_bind_stats=True
        )
        disabled = CAPAdam(
            disabled_model.parameters(), k=3e-5, track_bind_stats=False
        )
        for _step in range(2):
            inputs = torch.arange(20, dtype=torch.float32).view(5, 4) / 20
            targets = torch.arange(5) % 2
            for model, optimizer in (
                (enabled_model, enabled),
                (disabled_model, disabled),
            ):
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(inputs), targets).backward()
                optimizer.step()
        for enabled_parameter, disabled_parameter in zip(
            enabled_model.parameters(), disabled_model.parameters()
        ):
            self.assertTrue(torch.equal(enabled_parameter, disabled_parameter))
            self.assertIn("last_bind", enabled.state[enabled_parameter])
            self.assertNotIn("last_bind", disabled.state[disabled_parameter])
        self.assertIsNone(disabled.param_groups[0]["last_bind_rate"])

    def test_registered_grid_sizes_and_order(self) -> None:
        grids = registered_grids()
        self.assertEqual(
            {method: len(grid) for method, grid in grids.items()},
            {
                "adamw": 3,
                "adam_replay": 9,
                "upgd": 12,
                "ewc_oracle": 6,
                "l2_init": 3,
                "cap_release": 8,
            },
        )
        self.assertEqual(sum(map(len, grids.values())), 41)
        self.assertEqual(grids["adamw"][0], {"lr": 3e-4})
        self.assertEqual(
            grids["cap_release"][0],
            {
                "k": 3e-5,
                "T_m": 10_000,
                "T_a": 2_000,
                "retention": "release",
            },
        )
        self.assertEqual(
            UPGD_DEFAULTS,
            {
                "lr": 0.01,
                "weight_decay": 0.01,
                "beta_utility": 0.9999,
                "sigma": 0.1,
            },
        )
        self.assertEqual(
            grids["upgd"][0],
            {
                "lr": 0.01 / 3.0,
                "weight_decay": 0.01,
                "sigma": 0.1,
                "beta_utility": 0.9999,
            },
        )
        self.assertEqual(grids["upgd"][-1]["lr"], 0.03)
        self.assertAlmostEqual(grids["upgd"][-1]["sigma"], 0.01)
        self.assertAlmostEqual(
            grids["upgd"][-1]["beta_utility"], 0.99999
        )
        provenance = upgd_provenance()
        self.assertEqual(provenance["amendment"], "P1-A3")
        self.assertIn("statistics_input_permuted_mnist.py", provenance["task_config_source"])

    def test_p1_a3_preserves_pre_amendment_smoke_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "run_config.json"
            report_path = root / "report.md"
            old_config = {
                "experiment": "phase1_tuning",
                "upgd_smoke_gate": PRE_AMENDMENT_SMOKE,
            }
            config_path.write_text(json.dumps(old_config), encoding="utf-8")
            report_path.write_text("original failure\n", encoding="utf-8")
            self.assertTrue(
                preserve_pre_amendment_evidence(
                    config_path=config_path,
                    report_path=report_path,
                    existing=old_config,
                )
            )
            archive = root / "p1_a3_pre_amendment"
            self.assertEqual(
                (archive / "run_config.json").read_bytes(),
                config_path.read_bytes(),
            )
            self.assertEqual(
                (archive / "report.md").read_bytes(),
                report_path.read_bytes(),
            )

    def test_hook_contract_and_derived_cap_configs(self) -> None:
        artifact = tuned_artifact()
        self.assertFalse(cap_release_matches_core(artifact))
        validate_p1_a7_artifact(artifact)
        for arm in (
            "adamw",
            "adam_replay",
            "upgd",
            "ewc_oracle",
            "l2_init",
            "cap_release",
            "cap_ratchet",
            "cap_leak",
        ):
            model = nn.Linear(4, 2)
            _optimizer, hooks = build_arm(
                arm, model, artifact, seed=42, device=torch.device("cpu")
            )
            self.assertEqual(
                set(hooks),
                {"batch_mixer", "grad_hook", "on_boundary", "is_cap"},
            )
            self.assertEqual(bool(hooks["is_cap"]), arm.startswith("cap_"))

    def test_upgd_uses_every_parameter(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        optimizer, _hooks = build_arm(
            "upgd", model, tuned_artifact(), 42, torch.device("cpu")
        )
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(list(model.parameters())))

    def test_every_arm_takes_a_finite_training_step(self) -> None:
        artifact = tuned_artifact()
        for arm in (
            "adamw",
            "adam_replay",
            "upgd",
            "ewc_oracle",
            "l2_init",
            "cap_release",
            "cap_ratchet",
            "cap_leak",
        ):
            model = nn.Linear(4, 2)
            optimizer, hooks = build_arm(
                arm, model, artifact, 42, torch.device("cpu")
            )
            loss, accuracy = training_step(
                model=model,
                optimizer=optimizer,
                hooks=hooks,
                inputs=torch.randn(128, 4),
                targets=torch.arange(128) % 2,
            )
            self.assertTrue(torch.isfinite(loss), arm)
            self.assertTrue(0.0 <= accuracy <= 1.0, arm)

    def test_replay_reservoir_checkpoint_roundtrip(self) -> None:
        artifact = tuned_artifact()
        model1 = nn.Linear(4, 2)
        optimizer1, hooks1 = build_arm(
            "adam_replay", model1, artifact, 42, torch.device("cpu")
        )
        batch_count = 80
        for batch in range(batch_count):
            inputs = torch.arange(128 * 4, dtype=torch.float32).view(128, 4)
            inputs = inputs + batch * 1_000
            targets = torch.arange(128, dtype=torch.int64) % 2
            hooks1["batch_mixer"](inputs, targets)
        replay1 = optimizer1.phase1_state["replay"]
        self.assertEqual(replay1["size"], 5_000)
        fresh_after_first = 128 - int(replay1["replay_count"])
        self.assertEqual(
            replay1["seen"],
            128 + (batch_count - 1) * fresh_after_first,
        )

        model2 = nn.Linear(4, 2)
        optimizer2, hooks2 = build_arm(
            "adam_replay", model2, artifact, 42, torch.device("cpu")
        )
        checkpoint_buffer = io.BytesIO()
        torch.save(optimizer1.state_dict(), checkpoint_buffer)
        checkpoint_buffer.seek(0)
        optimizer2.load_state_dict(
            torch.load(checkpoint_buffer, map_location="cpu", weights_only=False)
        )
        replay2 = optimizer2.phase1_state["replay"]
        self.assertEqual(replay2["size"], replay1["size"])
        self.assertTrue(torch.equal(replay2["inputs"], replay1["inputs"]))
        self.assertTrue(
            torch.equal(
                replay2["generator"].get_state(),
                replay1["generator"].get_state(),
            )
        )
        fresh = torch.randn(128, 4)
        labels = torch.arange(128) % 2
        mixed1 = hooks1["batch_mixer"](fresh, labels)
        mixed2 = hooks2["batch_mixer"](fresh, labels)
        self.assertTrue(torch.equal(mixed1[0], mixed2[0]))
        self.assertTrue(torch.equal(mixed1[1], mixed2[1]))

    def test_ewc_raw_fisher_then_penalty_and_boundary(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        optimizer, hooks = build_arm(
            "ewc_oracle", model, tuned_artifact(), 42, torch.device("cpu")
        )
        parameter = model.weight
        parameter.grad = torch.tensor([[2.0, 3.0]])
        hooks["grad_hook"](model)
        state = optimizer.phase1_state["ewc"]
        name = "weight"
        self.assertTrue(
            torch.allclose(
                state["F_ema"][name], torch.tensor([[0.008, 0.018]])
            )
        )
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([[2.0, 3.0]])))
        hooks["on_boundary"](model)
        with torch.no_grad():
            parameter.add_(1.0)
        parameter.grad = torch.ones_like(parameter)
        hooks["grad_hook"](model)
        expected = torch.ones_like(parameter) + 1e3 * torch.tensor(
            [[0.008, 0.018]]
        )
        self.assertTrue(torch.allclose(parameter.grad, expected))

    def test_stream_b_schedule_is_pinned(self) -> None:
        schedule = stream_b_schedule()
        self.assertEqual(len(schedule), 200)
        self.assertTrue(all(0 <= item < 10 for item in schedule))
        self.assertEqual(schedule, stream_b_schedule())
        gaps = revisit_gap_distribution(schedule)
        self.assertEqual(gaps["expected_gap_tasks"], 10)
        self.assertEqual(gaps["count"], 190)
        self.assertEqual(
            sum(gaps["gap_task_counts"].values()), gaps["count"]
        )
        self.assertEqual(
            experiment_stream_b.stream_b_arms(),
            (
                "adamw",
                "adam_replay",
                "upgd",
                "ewc_oracle",
                "cap_release",
                "cap_ratchet",
                "cap_leak",
            ),
        )
        self.assertEqual(
            experiment_stream_b.stream_b_arms(True)[-1],
            CONDITIONAL_RETUNE_ARM,
        )

    def test_recurrent_validation_schedule_is_balanced(self) -> None:
        schedule = validation_schedule()
        self.assertEqual(len(schedule), 30)
        for start in range(0, 30, 5):
            self.assertEqual(set(schedule[start : start + 5]), set(range(5)))

    def test_v7_artifact_has_frozen_primary_and_dual_endpoints(self) -> None:
        grids = registered_grids()
        methods = {}
        validation = []
        for method, grid in grids.items():
            tried = []
            for index, config in enumerate(grid):
                tried.append(
                    {
                        "config_index": index,
                        "config": config,
                        "selection_metric": 0.90 - index * 0.001,
                        "tie_break_metric": 0.80,
                        "runtime_seconds": 1.0,
                        "status": "OK",
                    }
                )
                validation.append(
                    {
                        "method": method,
                        "config_index": index,
                        "validation_retention_metric": 0.95 if index == 1 else 0.50,
                        "runtime_seconds": 1.0,
                        "status": "OK",
                    }
                )
            methods[method] = {
                "config": grid[0],
                "selection_metric": 0.90,
                "tie_break_metric": 0.80,
                "winner_config_index": 0,
                "tried_configs": tried,
            }
        v4 = {
            "schema_version": 2,
            "amendment": "P1-A3",
            "methods": methods,
            "upgd": upgd_provenance(),
            "pre_amendment_upgd_smoke_gate": PRE_AMENDMENT_SMOKE,
            "amended_upgd_smoke_gate": {"status": "PASS"},
        }
        artifact = build_v7_artifact(
            v4=v4,
            validation_cells=validation,
            v4_hashes={"report.md": "abc"},
            v4_dir=Path("results/phase1/tuning_v4"),
        )
        self.assertEqual(artifact["schema_version"], 7)
        self.assertEqual(artifact["methods"]["adamw"]["primary"]["config"], grids["adamw"][0])
        self.assertEqual(
            artifact["methods"]["adamw"]["retention_focused"]["config"],
            grids["adamw"][1],
        )
        self.assertEqual(
            artifact["sensitivity_arms"]["cap_release_retention_focused"]["seeds"],
            [42, 123, 2026, 451, 789],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "phase1_tuned_configs.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(load_tuned_configs(path)["amendment"], "P1-A5")

    def test_stream_a_append_arm_matrix_uses_matching_rule(self) -> None:
        artifact = tuned_artifact()
        self.assertEqual(
            experiment_e4b.phase1_run_arms(artifact),
            (
                "adamw",
                "cap_leak",
                "cap_ratchet",
                "cap_release_10000",
                "adam_replay",
                "upgd",
                "ewc_oracle",
                "cap_release_tuned",
            ),
        )
        self.assertEqual(
            experiment_e4b.phase1_run_arms(artifact, True)[-1],
            CONDITIONAL_RETUNE_ARM,
        )

    def test_retention_focused_arm_uses_secondary_config(self) -> None:
        artifact = tuned_artifact()
        secondary = {"lr": 3e-4}
        entry = artifact["methods"]["adamw"]
        entry["retention_focused"]["config"] = secondary
        entry["endpoints_coincide"] = False
        artifact["sensitivity_arms"]["adamw_retention_focused"] = {
            "method": "adamw",
            "endpoint": "retention_focused",
            "label": RETENTION_FOCUSED_LABEL,
            "seeds": [42, 123, 2026],
            "config": secondary,
        }
        model = nn.Linear(4, 2)
        optimizer, _hooks = build_arm(
            "adamw_retention_focused",
            model,
            artifact,
            42,
            torch.device("cpu"),
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-4)

    def test_grading_uses_the_pinned_logged_stream_a_window(self) -> None:
        scalar_rows, pool_rows = grading_rows()
        artifact = tuned_artifact()
        grade = grade_phase1(
            stream_a_rows=scalar_rows,
            stream_b_pool_rows=pool_rows,
            tuned_configs=artifact,
        )
        self.assertEqual(grade["kill_gate"]["verdict"], "PASS")
        self.assertAlmostEqual(
            grade["kill_gate"]["primary"]["plasticity"]["threshold"], 0.80
        )
        self.assertAlmostEqual(
            grade["kill_gate"]["primary"]["retention"]["threshold"], 0.27
        )
        self.assertTrue(grade["controls"]["l2_init_descope_guard"]["passed"])
        self.assertNotIn("l2_init", grade["stream_b_end_retention"])

    def test_validity_controls_block_formal_kill_gate_evaluation(self) -> None:
        scalar_rows, pool_rows = grading_rows()
        replay_absent_rows = [
            {
                **row,
                "accuracy": (
                    0.205
                    if row["arm"] == "adam_replay"
                    else row["accuracy"]
                ),
            }
            for row in pool_rows
        ]
        replay_grade = grade_phase1(
            stream_a_rows=scalar_rows,
            stream_b_pool_rows=replay_absent_rows,
            tuned_configs=tuned_artifact(),
        )
        self.assertEqual(replay_grade["kill_gate"]["verdict"], "NOT-GRADED")
        self.assertEqual(
            replay_grade["kill_gate"]["primary"]["verdict"],
            "NOT-EVALUATED",
        )
        self.assertEqual(
            replay_grade["kill_gate"]["primary"]["reason"],
            "REPLAY-CONTROL-ABSENT",
        )

        artifact = tuned_artifact()
        artifact["methods"]["l2_init"]["tried_configs"][0][
            "validation_retention_metric"
        ] = 0.99
        guard_grade = grade_phase1(
            stream_a_rows=scalar_rows,
            stream_b_pool_rows=pool_rows,
            tuned_configs=artifact,
        )
        self.assertEqual(
            guard_grade["kill_gate"]["verdict"],
            "L2INIT-DESCOPE-INVALID",
        )
        self.assertEqual(
            guard_grade["kill_gate"]["primary"]["verdict"],
            "NOT-EVALUATED",
        )

    def test_conditional_retune_requires_grading_authorization(self) -> None:
        artifact = tuned_artifact()
        primary_a, primary_b = grading_rows(primary_stream_b=0.26)
        primary_grade = grade_phase1(
            stream_a_rows=primary_a,
            stream_b_pool_rows=primary_b,
            tuned_configs=artifact,
        )
        self.assertEqual(
            primary_grade["kill_gate"]["verdict"],
            "CONDITIONAL-RETUNE-REQUIRED",
        )
        self.assertEqual(
            primary_grade["kill_gate"]["primary"]["verdict"], "FAIL"
        )

        with tempfile.TemporaryDirectory() as directory:
            tuned_path = Path(directory) / "phase1_tuned_configs.json"
            request_path = Path(directory) / "conditional_retune_request.json"
            tuned_path.write_text(json.dumps(artifact), encoding="utf-8")
            request = conditional_retune_request(
                grade=primary_grade,
                tuned_config_path=tuned_path,
                tuned_configs=artifact,
            )
            request_path.write_text(json.dumps(request), encoding="utf-8")
            loaded = load_conditional_retune_request(
                request_path,
                tuned_config_path=tuned_path,
                tuned_configs=artifact,
            )
            self.assertEqual(loaded["config"], P1_A7_RETUNE_CAP_CONFIG)

        retune_a, retune_b = grading_rows(
            primary_stream_b=0.26,
            include_conditional=True,
        )
        final_grade = grade_phase1(
            stream_a_rows=retune_a,
            stream_b_pool_rows=retune_b,
            tuned_configs=artifact,
        )
        self.assertEqual(
            final_grade["kill_gate"]["verdict"],
            "PASS-AFTER-PREDECLARED-RETUNE",
        )
        self.assertTrue(
            final_grade["kill_gate"]["conditional_retune"]["passes"]
        )

    def test_e5_requires_complete_seed_coverage(self) -> None:
        seeds = (42, 123, 2026, 451, 789)
        drain_rows = [
            {
                "decay_mode": mode,
                "seed": seed,
                "step": DRAIN_STEPS,
                "relative_drift": {
                    "zero": 0.30,
                    "anchor": 0.10,
                    "blend": 0.01,
                }[mode],
            }
            for mode in ("zero", "anchor", "blend")
            for seed in seeds
        ]
        self.assertEqual(grade_drain(drain_rows, seeds)["verdict"], "PASS")
        self.assertFalse(grade_drain(drain_rows[:-1], seeds)["complete"])
        sanity_rows = [
            {
                "arm": arm,
                "seed": seed,
                "k": 3e-5 if arm == "cap_blend" else "",
                "test_accuracy": 0.90 + 0.001 * index,
            }
            for index, seed in enumerate(seeds)
            for arm in ("adamw", "cap_blend")
        ]
        self.assertTrue(grade_sanity(sanity_rows, 3e-5, seeds)["complete"])
        self.assertFalse(
            grade_sanity(sanity_rows[:-1], 3e-5, seeds)["complete"]
        )

    def test_e6_grading_and_dependency_skip(self) -> None:
        rows = []
        for workload in E6_WORKLOADS:
            for arm in E6_ARMS:
                for repeat in range(E6_REPEATS):
                    full_step = 10.4 if arm == "cap_diag_off" else 10.0
                    rows.append(
                        {
                            "workload": workload,
                            "arm": arm,
                            "repeat": repeat,
                            "full_step_ms": full_step,
                            "opt_step_ms": 1.0,
                            "max_memory_bytes": 100,
                            "state_ratio": 1.0,
                            "status": "OK",
                        }
                    )
        self.assertEqual(grade_e6(summarize_e6(rows))["verdict"], "PASS")
        without_gpt = [row for row in rows if row["workload"] != "gpt2"]
        self.assertEqual(
            grade_e6(summarize_e6(without_gpt))["verdict"],
            "SKIPPED-DEPENDENCY",
        )


if __name__ == "__main__":
    unittest.main()
