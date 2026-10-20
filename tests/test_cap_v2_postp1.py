"""Pre-launch gate for the CAP-v2.5 conflict-momentum-restart experiment."""

from __future__ import annotations

import contextlib
import copy
import io
import math
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import analyze_cap_v2_postp1 as analysis
import experiment_e4b as stream_a
import experiment_cap_v2_postp1 as runner
import experiment_stream_b as stream_b
from experiment_e2 import read_csv, write_csv
from experiment_e3 import fixed_permutation, make_loader, repeat_batches, seed_everything
from experiment_e4a import fixed_test_subset
from cap_adam_experimental import CAPAdam, RawImportanceTracker, _update_importance_
from cap_v2_postp1_common import (
    DIAGNOSTIC_FIELDS,
    FULL_SEEDS,
    MODES,
    POST_P1_POOL_SIZE,
    CAP_CONFIG,
    arm_name,
    block_stream_b_schedule,
    build_run_config,
    build_cap_release_variant,
    fresh_model,
    lock_run_config,
    model_tensor_hash,
)


class TinyMLP(nn.Module):
    def __init__(self, input_units: int, output_units: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(input_units, output_units))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _tiny_runtime() -> SimpleNamespace:
    generator = torch.Generator().manual_seed(91)
    inputs = torch.randn(16, 1, 28, 28, generator=generator)
    targets = torch.arange(16) % 10
    dataset = TensorDataset(inputs, targets)
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_workers=0,
        mnist_train=dataset,
        test_loader_a=test_loader,
        test_loader_b=test_loader,
        permutations_a={1: torch.arange(784), 2: torch.arange(783, -1, -1)},
        permutations_b={
            index: torch.roll(torch.arange(784), shifts=index)
            for index in range(POST_P1_POOL_SIZE)
        },
        schedule_b=[0, 1],
        pool_size_b=POST_P1_POOL_SIZE,
    )


@contextmanager
def _tiny_protocol():
    with ExitStack() as stack:
        stack.enter_context(
            patch.multiple(
                stream_a,
                TASK_COUNT=2,
                STEPS_PER_TASK=2,
                TOTAL_STEPS=4,
                SCALAR_LOG_EVERY=1,
                CHECKPOINT_EVERY=2,
                MLP300=TinyMLP,
            )
        )
        stack.enter_context(
            patch.multiple(
                stream_b,
                TASK_COUNT=2,
                STEPS_PER_TASK=2,
                TOTAL_STEPS=4,
                SCALAR_LOG_EVERY=1,
                CHECKPOINT_EVERY=2,
                MLP300=TinyMLP,
            )
        )
        yield


def _paths(root: Path, stream: str) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return runner.stream_paths(root, stream)


def _copy_partial(
    full: dict[str, Path], partial: dict[str, Path], stream: str, variant: str
) -> None:
    arm = arm_name(variant)
    write_csv(
        partial["scalars"],
        [row for row in read_csv(full["scalars"]) if int(float(row["step"])) <= 2],
        stream_a.SCALAR_FIELDS,
    )
    write_csv(
        partial["tasks"],
        [row for row in read_csv(full["tasks"]) if int(float(row["task"])) <= 1],
        stream_a.TASK_FIELDS,
    )
    write_csv(
        partial["diagnostics"],
        [
            row
            for row in read_csv(full["diagnostics"])
            if int(float(row["global_step"])) <= 2
        ],
        DIAGNOSTIC_FIELDS,
    )
    if stream == "b":
        write_csv(
            partial["pool"],
            [row for row in read_csv(full["pool"]) if int(float(row["task"])) <= 1],
            stream_b.POOL_FIELDS,
        )
    partial["checkpoints"].mkdir(parents=True, exist_ok=True)
    filename = f"{arm}_seed42_step2.pt"
    shutil.copy2(full["checkpoints"] / filename, partial["checkpoints"] / filename)


def _prepared_optimizer(
    *,
    response: str,
    previous_bind: bool,
    old_exp_avg: float,
    gradient: float,
    maximize: bool = False,
    track_bind_stats: bool = True,
    amsgrad: bool = False,
) -> tuple[nn.Parameter, CAPAdam, dict]:
    parameter = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = CAPAdam(
        [parameter],
        lr=0.0,
        betas=(0.9, 0.999),
        k=1.0,
        T_a=math.inf,
        freeze_stats=True,
        momentum_response=response,
        track_bind_stats=track_bind_stats,
        maximize=maximize,
        amsgrad=amsgrad,
    )
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    state["exp_avg"].fill_(old_exp_avg)
    state["exp_avg_sq"].fill_(4.0)
    state["imp"].fill_(1.0)
    state["last_bind"].fill_(previous_bind)
    if amsgrad:
        state["max_exp_avg_sq"].fill_(5.0)
    parameter.grad = torch.full_like(parameter, gradient)
    return parameter, optimizer, state


class CapV2PostP1Tests(unittest.TestCase):
    def assert_nested_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_value, right_value in zip(left, right):
                self.assert_nested_equal(left_value, right_value)
        else:
            self.assertEqual(left, right)

    def _parse_fails(self, arguments: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(arguments)

    def test_cli_has_exact_v25_scientific_modes(self) -> None:
        base = ["--stream", "a", "--seeds", "42"]
        self._parse_fails(base)
        self._parse_fails(
            [*base, "--baseline", "--conflict-momentum-restart"]
        )
        self._parse_fails([*base, "--q4-slack-gate"])
        self._parse_fails([*base, "--baseline", "--momentum-diagnostic"])
        for flag, variant in (
            ("--baseline", "baseline"),
            ("--conflict-momentum-restart", "conflict_momentum_restart"),
        ):
            args = runner.parse_args(["--stream", "a", flag, "--seeds", "42"])
            self.assertEqual(args.variant, variant)

    def test_seed_validation_and_order(self) -> None:
        base = ["--stream", "a", "--baseline"]
        self._parse_fails([*base, "--seeds", "42", "42"])
        self._parse_fails([*base, "--seeds", "999"])
        self._parse_fails([*base, "--seeds"])
        self._parse_fails([*base, "--all-seeds", "--seeds", "42"])
        self.assertEqual(
            runner.parse_args([*base, "--seeds", "789", "42", "123"]).resolved_seeds,
            (789, 42, 123),
        )
        self.assertEqual(runner.parse_args([*base, "--all-seeds"]).resolved_seeds, FULL_SEEDS)

    def test_variant_mapping_and_frozen_configuration(self) -> None:
        self.assertEqual(
            MODES,
            {
                "baseline": {"momentum_response": "none"},
                "conflict_momentum_restart": {
                    "momentum_response": "conflict_restart"
                },
            },
        )
        settings = {}
        for variant in MODES:
            optimizer = build_cap_release_variant(fresh_model(42), variant=variant)
            settings[variant] = [
                {
                    key: value
                    for key, value in group.items()
                    if key not in {"params", "momentum_response"}
                }
                for group in optimizer.param_groups
            ]
            self.assertTrue(all(group["box_geometry"] == "coordinate" for group in optimizer.param_groups))
            self.assertTrue(all(group["release_update"] == "contest_blocked" for group in optimizer.param_groups))
            self.assertTrue(
                all(
                    group["momentum_response"] == MODES[variant]["momentum_response"]
                    for group in optimizer.param_groups
                )
            )
        self.assertEqual(settings["baseline"], settings["conflict_momentum_restart"])
        self.assertEqual(
            CAP_CONFIG,
            {
                "lr": 0.0033333333333333335,
                "weight_decay": 0.0,
                "k": 7e-5,
                "T_a": 2000.0,
                "T_s": 2000.0,
                "T_m": 1_000_000.0,
                "retention": "release",
            },
        )

    def test_baseline_none_is_exact_default_equivalent(self) -> None:
        parameters = [nn.Parameter(torch.tensor([[0.1, -0.2]], dtype=torch.float64)) for _ in range(2)]
        optimizers = [
            CAPAdam([parameters[0]], release_update="contest_blocked", T_m=10.0),
            CAPAdam([parameters[1]], release_update="contest_blocked", T_m=10.0, momentum_response="none"),
        ]
        for gradient in (
            torch.tensor([[1.0, -2.0]], dtype=torch.float64),
            torch.tensor([[3.0, 0.5]], dtype=torch.float64),
            torch.tensor([[-0.25, 4.0]], dtype=torch.float64),
        ):
            for parameter, optimizer in zip(parameters, optimizers):
                parameter.grad = gradient.clone()
                optimizer.step()
            self.assertTrue(torch.equal(parameters[0], parameters[1]))
            self.assert_nested_equal(optimizers[0].state[parameters[0]], optimizers[1].state[parameters[1]])

    def test_conflict_restart_formula_preserves_current_gradient(self) -> None:
        _, optimizer, state = _prepared_optimizer(
            response="conflict_restart", previous_bind=True,
            old_exp_avg=2.0, gradient=-3.0,
        )
        optimizer.step()
        expected = (1.0 - 0.9) * -3.0
        ordinary = 0.9 * 2.0 + (1.0 - 0.9) * -3.0
        self.assertAlmostEqual(float(state["exp_avg"]), expected)
        self.assertNotAlmostEqual(float(state["exp_avg"]), ordinary)
        group = optimizer.param_groups[0]
        self.assertEqual(group["last_previous_bind_count"], 1)
        self.assertEqual(group["last_momentum_conflict_count"], 1)
        self.assertEqual(group["last_momentum_restart_count"], 1)

    def test_no_restart_without_previous_bind_or_when_signs_do_not_conflict(self) -> None:
        cases = (
            (False, 2.0, -3.0),
            (True, 2.0, 3.0),
            (True, 0.0, -3.0),
            (True, 2.0, 0.0),
        )
        for previous_bind, old_momentum, gradient in cases:
            _, optimizer, state = _prepared_optimizer(
                response="conflict_restart",
                previous_bind=previous_bind,
                old_exp_avg=old_momentum,
                gradient=gradient,
            )
            optimizer.step()
            expected = 0.9 * old_momentum + 0.1 * gradient
            self.assertAlmostEqual(float(state["exp_avg"]), expected)
            self.assertEqual(optimizer.param_groups[0]["last_momentum_restart_count"], 0)

    def test_current_bind_cannot_trigger_same_step_restart(self) -> None:
        parameter, optimizer, state = _prepared_optimizer(
            response="conflict_restart", previous_bind=False,
            old_exp_avg=2.0, gradient=-3.0,
        )
        with torch.no_grad():
            parameter.fill_(2.0)
        optimizer.step()
        self.assertAlmostEqual(float(state["exp_avg"]), 1.5)
        self.assertEqual(optimizer.param_groups[0]["last_momentum_restart_count"], 0)
        self.assertTrue(state["last_bind"].item())

    def test_only_first_moment_history_is_directly_restarted(self) -> None:
        _, optimizer, state = _prepared_optimizer(
            response="conflict_restart", previous_bind=True,
            old_exp_avg=2.0, gradient=-3.0, amsgrad=True,
        )
        old_anchor = state["anchor"].clone()
        old_imp = state["imp"].clone()
        old_step = state["step"]
        optimizer.step()
        self.assertEqual(state["step"], old_step + 1)
        self.assertAlmostEqual(float(state["exp_avg_sq"]), 4.005)
        self.assertEqual(float(state["max_exp_avg_sq"]), 5.0)
        self.assertTrue(torch.equal(state["anchor"], old_anchor))
        self.assertTrue(torch.equal(state["imp"], old_imp))

    def test_restart_uses_effective_gradient_under_maximize(self) -> None:
        _, optimizer, state = _prepared_optimizer(
            response="conflict_restart", previous_bind=True,
            old_exp_avg=2.0, gradient=3.0, maximize=True,
        )
        optimizer.step()
        self.assertAlmostEqual(float(state["exp_avg"]), -0.3)
        self.assertEqual(optimizer.param_groups[0]["last_momentum_restart_count"], 1)

    def test_exempt_and_passive_groups_do_not_restart(self) -> None:
        for group_options in (
            {"protect": False},
            {"protect": False, "passive": True},
        ):
            parameter = nn.Parameter(torch.zeros(1, dtype=torch.float64))
            optimizer = CAPAdam(
                [{"params": [parameter], **group_options}],
                lr=0.0,
                momentum_response="conflict_restart",
            )
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            state = optimizer.state[parameter]
            state["exp_avg"].fill_(2.0)
            state["last_bind"].fill_(True)
            parameter.grad = torch.full_like(parameter, -3.0)
            optimizer.step()
            self.assertAlmostEqual(float(state["exp_avg"]), 1.5)
            self.assertEqual(optimizer.param_groups[0]["last_momentum_restart_count"], 0)

    def test_invalid_restart_geometry_is_rejected(self) -> None:
        for options in (
            {"protection": "spring"},
            {"box_geometry": "q4_slack"},
            {"passive": True},
        ):
            with self.assertRaisesRegex(ValueError, "active coordinate-box"):
                CAPAdam(
                    [nn.Parameter(torch.zeros(2, 2))],
                    momentum_response="conflict_restart",
                    **options,
                )

    def test_restart_maintains_bind_state_when_reporting_is_disabled(self) -> None:
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = CAPAdam(
            [parameter], momentum_response="conflict_restart",
            track_bind_stats=False, lr=0.0, k=1.0,
        )
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertIn("last_bind", optimizer.state[parameter])
        self.assertEqual(optimizer.state[parameter]["last_bind"].dtype, torch.bool)
        self.assertIsNone(optimizer.param_groups[0]["last_bind_rate"])

    def test_baseline_counters_observe_conflicts_without_restarting(self) -> None:
        _, optimizer, state = _prepared_optimizer(
            response="none", previous_bind=True, old_exp_avg=2.0, gradient=-3.0
        )
        optimizer.step()
        self.assertAlmostEqual(float(state["exp_avg"]), 1.5)
        group = optimizer.param_groups[0]
        self.assertEqual(group["last_previous_bind_count"], 1)
        self.assertEqual(group["last_momentum_conflict_count"], 1)
        self.assertEqual(group["last_momentum_restart_count"], 0)

    def test_resume_equivalence_for_restart_optimizer(self) -> None:
        gradients = [1.0, -2.0, 3.0, -4.0]
        continuous_parameter = nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
        split_parameter = nn.Parameter(continuous_parameter.detach().clone())
        continuous = CAPAdam([continuous_parameter], momentum_response="conflict_restart", k=1e-4)
        split = CAPAdam([split_parameter], momentum_response="conflict_restart", k=1e-4)
        for gradient in gradients[:2]:
            for parameter, optimizer in ((continuous_parameter, continuous), (split_parameter, split)):
                parameter.grad = torch.tensor([gradient], dtype=parameter.dtype)
                optimizer.step()
        checkpoint = copy.deepcopy(split.state_dict())
        resumed_parameter = nn.Parameter(split_parameter.detach().clone())
        resumed = CAPAdam([resumed_parameter], momentum_response="conflict_restart", k=1e-4)
        resumed.load_state_dict(checkpoint)
        for gradient in gradients[2:]:
            continuous_parameter.grad = torch.tensor([gradient], dtype=continuous_parameter.dtype)
            continuous.step()
            resumed_parameter.grad = torch.tensor([gradient], dtype=resumed_parameter.dtype)
            resumed.step()
        self.assertTrue(torch.equal(continuous_parameter, resumed_parameter))
        self.assert_nested_equal(
            continuous.state[continuous_parameter], resumed.state[resumed_parameter]
        )

    def test_checkpoint_mode_crossing_is_rejected_before_mutation(self) -> None:
        for source_response, target_response, allowed in (
            ("none", "none", True),
            ("conflict_restart", "conflict_restart", True),
            ("none", "conflict_restart", False),
            ("conflict_restart", "none", False),
        ):
            source_parameter = nn.Parameter(torch.tensor([0.0]))
            source = CAPAdam([source_parameter], momentum_response=source_response)
            source_parameter.grad = torch.tensor([1.0])
            source.step()
            target_parameter = nn.Parameter(torch.tensor([5.0]))
            target = CAPAdam([target_parameter], momentum_response=target_response, lr=0.25)
            old_lr = target.param_groups[0]["lr"]
            if allowed:
                target.load_state_dict(copy.deepcopy(source.state_dict()))
            else:
                with self.assertRaisesRegex(RuntimeError, "momentum_response mismatch"):
                    target.load_state_dict(copy.deepcopy(source.state_dict()))
                self.assertEqual(target.param_groups[0]["lr"], old_lr)
                self.assertFalse(target.state)

    def test_old_checkpoint_defaults_to_none_and_guards_remain(self) -> None:
        parameter = nn.Parameter(torch.tensor([0.0]))
        optimizer = CAPAdam([parameter])
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        old = copy.deepcopy(optimizer.state_dict())
        old["param_groups"][0].pop("momentum_response")
        optimizer.load_state_dict(old)
        self.assertEqual(optimizer.param_groups[0]["momentum_response"], "none")
        for key, value in (("leash_mode", "selective"), ("evidence_mode", "adam_v")):
            checkpoint = copy.deepcopy(optimizer.state_dict())
            checkpoint["param_groups"][0][key] = value
            with self.assertRaisesRegex(RuntimeError, "rejected CAP-v2.1"):
                optimizer.load_state_dict(checkpoint)
        q4_parameter = nn.Parameter(torch.tensor([[0.0]]))
        q4_optimizer = CAPAdam([q4_parameter], box_geometry="q4_slack")
        coordinate_checkpoint = copy.deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(RuntimeError, "box_geometry mismatch"):
            q4_optimizer.load_state_dict(coordinate_checkpoint)

    def test_restart_checkpoint_requires_valid_bind_state(self) -> None:
        parameter = nn.Parameter(torch.tensor([0.0]))
        optimizer = CAPAdam([parameter], momentum_response="conflict_restart")
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        checkpoint["state"][0].pop("last_bind")
        target_parameter = nn.Parameter(torch.tensor([0.0]))
        target = CAPAdam([target_parameter], momentum_response="conflict_restart")
        with self.assertRaisesRegex(RuntimeError, "lacks last_bind"):
            target.load_state_dict(checkpoint)

    def test_release_raw_importance_and_tracker_are_unchanged(self) -> None:
        imp = torch.tensor([1.0])
        _update_importance_(
            imp, torch.tensor([100.0]), torch.tensor([True]),
            retention="release", release_update="contest_blocked", T_m=10.0,
        )
        self.assertAlmostEqual(float(imp), 0.9)
        parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = CAPAdam([parameter], lr=0.0, T_m=math.inf)
        tracker = RawImportanceTracker(optimizer)
        parameter.grad = torch.tensor([1.0, 2.0])
        optimizer.step()
        tracker.update()
        self.assertTrue(torch.equal(tracker.state[parameter], optimizer.state[parameter]["imp"]))

    def test_run_config_locks_scientific_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_config.json"
            proposed = build_run_config(
                stream="a", variant="conflict_momentum_restart",
                device=torch.device("cpu"), runner_path=runner.RUNNER_PATH,
            )
            self.assertEqual(proposed["experiment_version"], "v2.5-conflict-momentum-restart")
            self.assertEqual(proposed["momentum_response"], "conflict_restart")
            self.assertEqual(proposed["box_geometry"], "coordinate")
            lock_run_config(path, proposed, requested_seeds=(42,), resume=False, device=torch.device("cpu"))
            changed = dict(proposed, momentum_response="none")
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                lock_run_config(path, changed, requested_seeds=(123,), resume=True, device=torch.device("cpu"))

    def test_stream_pairing(self) -> None:
        dataset = TensorDataset(
            torch.arange(32 * 784, dtype=torch.float32).reshape(32, 1, 28, 28),
            torch.arange(32) % 10,
        )
        schedule = block_stream_b_schedule(POST_P1_POOL_SIZE)
        fingerprints = []
        for variant in MODES:
            seed_everything(42)
            model = TinyMLP(784, 10)
            build_cap_release_variant(model, variant=variant)
            loader = make_loader(dataset, 8, torch.device("cpu"), 0, shuffle=True, seed=42)
            first_batch = next(repeat_batches(loader))
            fingerprints.append(
                (
                    model_tensor_hash(model), first_batch[0].clone(), first_batch[1].clone(),
                    fixed_permutation(784, stream_a.PERMUTATION_SEED_BASE + 1),
                    fixed_permutation(784, stream_b.POOL_SEED_BASE), tuple(schedule),
                    torch.tensor(fixed_test_subset(dataset, stream_a.TEST_SUBSET_SEED).indices),
                    torch.tensor(fixed_test_subset(dataset, stream_b.TEST_SUBSET_SEED).indices),
                )
            )
        self.assert_nested_equal(fingerprints[0], fingerprints[1])

    def test_runner_resume_equivalence_for_both_streams_and_modes(self) -> None:
        runtime = _tiny_runtime()
        with _tiny_protocol(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for stream in ("a", "b"):
                for variant in MODES:
                    full = _paths(root / "full" / stream / variant, stream)
                    resumed = _paths(root / "resumed" / stream / variant, stream)
                    runner.run_cell(stream=stream, variant=variant, seed=42, runtime=runtime, paths=full, resume=False)
                    _copy_partial(full, resumed, stream, variant)
                    runner.run_cell(stream=stream, variant=variant, seed=42, runtime=runtime, paths=resumed, resume=True)
                    for key in ("scalars", "tasks", "diagnostics"):
                        self.assertEqual(read_csv(full[key]), read_csv(resumed[key]))
                    if stream == "b":
                        self.assertEqual(read_csv(full["pool"]), read_csv(resumed["pool"]))
                    filename = f"{arm_name(variant)}_seed42_step4.pt"
                    full_checkpoint = stream_a.load_torch_checkpoint(full["checkpoints"] / filename)
                    resumed_checkpoint = stream_a.load_torch_checkpoint(resumed["checkpoints"] / filename)
                    self.assert_nested_equal(full_checkpoint["model_state_dict"], resumed_checkpoint["model_state_dict"])
                    self.assert_nested_equal(full_checkpoint["optimizer_state_dict"], resumed_checkpoint["optimizer_state_dict"])

    def test_final_task_csv_without_committed_pool_resumes_previous_checkpoint(self) -> None:
        runtime = _tiny_runtime()
        with _tiny_protocol(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, interrupted = _paths(root / "full", "b"), _paths(root / "interrupted", "b")
            runner.run_cell(stream="b", variant="baseline", seed=42,
                            runtime=runtime, paths=full, resume=False)
            _copy_partial(full, interrupted, "b", "baseline")
            # Simulate the final task CSV committed before the final pool CSV
            # and checkpoint: the last durable checkpoint is still step two.
            write_csv(interrupted["tasks"], read_csv(full["tasks"]), stream_b.TASK_FIELDS)
            runner.run_cell(stream="b", variant="baseline", seed=42,
                            runtime=runtime, paths=interrupted, resume=True)
            self.assertEqual(read_csv(full["pool"]), read_csv(interrupted["pool"]))
            self.assertEqual(read_csv(full["tasks"]), read_csv(interrupted["tasks"]))

    def test_runner_cross_mode_resume_is_rejected(self) -> None:
        runtime = _tiny_runtime()
        with _tiny_protocol(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _paths(root / "baseline", "a")
            runner.run_cell(stream="a", variant="baseline", seed=42, runtime=runtime, paths=baseline, resume=False)
            restart = _paths(root / "restart", "a")
            restart["checkpoints"].mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                baseline["checkpoints"] / f"{arm_name('baseline')}_seed42_step2.pt",
                restart["checkpoints"] / f"{arm_name('conflict_momentum_restart')}_seed42_step2.pt",
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                runner.run_cell(
                    stream="a", variant="conflict_momentum_restart", seed=42,
                    runtime=runtime, paths=restart, resume=True,
                )

    def test_v25_directories_and_diagnostic_counter_contract(self) -> None:
        root = Path("results")
        for variant in MODES:
            self.assertEqual(
                runner.stream_directory(root, variant, "a"),
                root / f"v25_{variant}" / "stream_a",
            )
            self.assertNotIn(
                "boundary_momentum",
                runner.stream_paths(root / f"v25_{variant}" / "stream_a", "a"),
            )
        runtime = _tiny_runtime()
        with _tiny_protocol(), tempfile.TemporaryDirectory() as temporary:
            for variant in MODES:
                paths = _paths(Path(temporary) / variant, "a")
                runner.run_cell(stream="a", variant=variant, seed=42, runtime=runtime, paths=paths, resume=False)
                rows = read_csv(paths["diagnostics"])
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(
                        int(float(row["momentum_restart_count"])),
                        int(float(row["momentum_conflict_count"]))
                        if variant == "conflict_momentum_restart"
                        else 0,
                    )
                checkpoint = stream_a.load_torch_checkpoint(
                    paths["checkpoints"] / f"{arm_name(variant)}_seed42_step4.pt"
                )
                for state in checkpoint["optimizer_state_dict"]["state"].values():
                    self.assertTrue(
                        all(
                            torch.isfinite(value).all()
                            for value in state.values()
                            if isinstance(value, torch.Tensor)
                        )
                    )

    def test_analysis_reports_v25_performance_mechanism_steps_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = block_stream_b_schedule(POST_P1_POOL_SIZE)
            prior_index = schedule[0]
            for variant, plasticity, retention, rebind in (
                ("baseline", 0.70, 0.60, 0.9),
                ("conflict_momentum_restart", 0.80, 0.65, 0.5),
            ):
                arm = arm_name(variant)
                root_a = root / f"v25_{variant}" / "stream_a"
                root_b = root / f"v25_{variant}" / "stream_b"
                scalar_rows = []
                for step in (500_100, 502_000):
                    scalar_rows.append(
                        {
                            "arm": arm, "seed": 42, "step": step,
                            "batch_loss": 0.1, "batch_accuracy": plasticity,
                            "median_radius": "", "frozen_fraction": "", "bind_rate": "",
                            "median_imp": "", "theta_norm": "",
                        }
                    )
                write_csv(root_a / "stream_scalars.csv", scalar_rows, stream_a.SCALAR_FIELDS)
                diagnostic = {field: "" for field in DIAGNOSTIC_FIELDS}
                diagnostic.update(
                    {
                        "variant": variant, "stream": "a", "seed": 42,
                        "global_step": 1_000_000, "task": 200, "local_step": 5_000,
                        "median_imp": 4.0, "median_radius": 0.5, "bind_rate": 0.2,
                        "theta_norm": 3.0, "previous_bind_count": 10,
                        "momentum_conflict_count": 4,
                        "momentum_restart_count": 4 if variant != "baseline" else 0,
                        "rebind_count": int(rebind * 10),
                        "momentum_conflict_given_previous_bind": 0.4,
                        "rebind_given_current_bind": rebind,
                        "rebind_given_previous_bind": rebind,
                    }
                )
                write_csv(root_a / "cap_v2_diagnostics.csv", [diagnostic], DIAGNOSTIC_FIELDS)
                write_csv(
                    root_b / "streamb_tasks.csv",
                    [{"arm": arm, "seed": 42, "task": 200, "task_end_accuracy": 0.75, "steps_to_085": ""}],
                    stream_b.TASK_FIELDS,
                )
                pool_rows = [
                    {
                        "arm": arm, "seed": 42, "task": 2,
                        "pool_index": prior_index, "accuracy": retention - 0.1,
                        "is_current": False,
                    }
                ]
                pool_rows.extend(
                    {
                        "arm": arm, "seed": 42, "task": 200, "pool_index": index,
                        "accuracy": retention if index != schedule[-1] else 0.9,
                        "is_current": index == schedule[-1],
                    }
                    for index in range(POST_P1_POOL_SIZE)
                )
                write_csv(root_b / "streamb_pool.csv", pool_rows, stream_b.POOL_FIELDS)

            summary, differences, report = analysis.analyze(root, (42,))
            delta = next(row for row in differences if row["metric"] == "stream_a_plasticity")
            self.assertAlmostEqual(delta["mean_difference"], 0.1)
            rebind_delta = next(row for row in differences if row["metric"] == "rebind_given_current_bind")
            self.assertAlmostEqual(rebind_delta["mean_difference"], -0.4)
            metrics = {row["metric"] for row in summary}
            self.assertIn("stream_a_accuracy_step_100", metrics)
            self.assertIn("late_stream_a_accuracy_step_2000", metrics)
            self.assertIn("stream_b_retention_gap_1_5", metrics)
            self.assertIn("momentum_restart_rate", metrics)
            self.assertIn("Status: **PRELIMINARY**", report)


if __name__ == "__main__":
    unittest.main()
