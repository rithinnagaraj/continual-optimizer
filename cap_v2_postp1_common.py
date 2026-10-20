"""Shared invariants and utilities for the frozen Post-P1 CAP-v2 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Sequence

import numpy as np
import torch
from torch import Tensor, nn

import experiment_e4b as stream_a
import experiment_stream_b as stream_b
from experiment_e2 import write_csv
from experiment_e3 import cap_parameter_groups
from experiment_e4a import (
    ADAM_EPS,
    BETAS,
    C_REFRESH_STEPS,
    DECAY_MODE,
    PROTECTION,
    TRACK_BIND_STATS,
)
from model import MLP300
from post_phase1_streams import EVALUATION_BATCH_SIZE as PAPER_EVALUATION_BATCH_SIZE
from phase1_stream_common import CHECKPOINT_EVERY as STREAM_CHECKPOINT_EVERY
from phase1_stream_common import SCALAR_LOG_EVERY as STREAM_SCALAR_LOG_EVERY
from post_phase1_common import block_stream_b_schedule
from cap_adam_experimental import CAPAdam, protection_radius


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "post_phase1_cap_v2"
EXPERIMENT_VERSION = "v2.5-conflict-momentum-restart"
PROTOCOL_IDENTIFIER = "post-phase1-final-streams-PP1-A1(C)-pool40"
FULL_SEEDS = (42, 123, 2026, 451, 789)
POST_P1_POOL_SIZE = 40
DIAGNOSTIC_EVERY = 1_000

CAP_CONFIG: dict[str, Any] = {
    "lr": 0.0033333333333333335,
    "weight_decay": 0.0,
    "k": 7e-5,
    "T_a": 2000.0,
    "T_s": 2000.0,
    "T_m": 1_000_000.0,
    "retention": "release",
}

MODES = {
    "baseline": {"momentum_response": "none"},
    "conflict_momentum_restart": {"momentum_response": "conflict_restart"},
}

MODE_FLAGS = {
    "baseline": "baseline",
    "conflict_momentum_restart": "conflict_momentum_restart",
}

DIAGNOSTIC_FIELDS = (
    "variant",
    "stream",
    "seed",
    "global_step",
    "task",
    "local_step",
    "median_imp",
    "p90_imp",
    "median_radius",
    "p10_radius",
    "bind_rate",
    "previous_bind_count",
    "momentum_conflict_count",
    "momentum_restart_count",
    "rebind_count",
    "momentum_conflict_given_previous_bind",
    "rebind_given_current_bind",
    "rebind_given_previous_bind",
    "median_anchor_distance",
    "p90_anchor_distance",
    "theta_norm",
    "anchor_norm",
    "median_anchor_delta_since_previous_diag",
    "p90_anchor_delta_since_previous_diag",
    "fraction_imp_decreased_since_previous_diag",
    "total_imp_mass",
)

PROTOCOL_SOURCE_FILES = (
    ROOT / "post_phase1_streams.py",
    ROOT / "post_phase1_common.py",
    ROOT / "experiment_e4b.py",
    ROOT / "experiment_stream_b.py",
    ROOT / "phase1_stream_common.py",
    ROOT / "phase1_arms.py",
    ROOT / "paper_run_integrity.py",
    ROOT / "experiment_cap_v2_postp1.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def protocol_source_hashes() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in PROTOCOL_SOURCE_FILES}


def protocol_hash() -> str:
    payload = {
        "identifier": PROTOCOL_IDENTIFIER,
        "plasticity_observation": "full incoming current-task batch before replay mixing",
        "evaluation_batch_size": PAPER_EVALUATION_BATCH_SIZE,
        "source_hashes": protocol_source_hashes(),
        "stream_a": {
            "tasks": stream_a.TASK_COUNT,
            "steps_per_task": stream_a.STEPS_PER_TASK,
            "scalar_log_every": stream_a.SCALAR_LOG_EVERY,
            "checkpoint_every": stream_a.CHECKPOINT_EVERY,
            "test_subset_seed": stream_a.TEST_SUBSET_SEED,
            "permutation_seed_base": stream_a.PERMUTATION_SEED_BASE,
        },
        "stream_b": {
            "tasks": stream_b.TASK_COUNT,
            "steps_per_task": stream_b.STEPS_PER_TASK,
            "pool_size": POST_P1_POOL_SIZE,
            "pool_seed_base": stream_b.POOL_SEED_BASE,
            "schedule_seed": stream_b.SCHEDULE_SEED,
            "schedule": block_stream_b_schedule(POST_P1_POOL_SIZE),
            "test_subset_seed": stream_b.TEST_SUBSET_SEED,
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def assert_authoritative_protocol() -> None:
    expected = (200, 5_000, 1_000_000)
    observed_a = (stream_a.TASK_COUNT, stream_a.STEPS_PER_TASK, stream_a.TOTAL_STEPS)
    observed_b = (stream_b.TASK_COUNT, stream_b.STEPS_PER_TASK, stream_b.TOTAL_STEPS)
    if observed_a != expected or observed_b != expected:
        raise RuntimeError(
            f"Post-P1 stream dimensions changed: A={observed_a}, B={observed_b}"
        )
    if (
        stream_a.SCALAR_LOG_EVERY != 100
        or STREAM_SCALAR_LOG_EVERY != 100
        or stream_a.CHECKPOINT_EVERY != 50_000
        or STREAM_CHECKPOINT_EVERY != 50_000
        or C_REFRESH_STEPS != 100
    ):
        raise RuntimeError("Post-P1 logging, checkpoint, or c-refresh cadence changed")
    if (
        stream_a.TEST_SUBSET_SEED != 999
        or stream_b.TEST_SUBSET_SEED != 999
        or stream_a.PERMUTATION_SEED_BASE != 100_000
        or stream_b.POOL_SEED_BASE != 200_001
        or stream_b.SCHEDULE_SEED != 555
    ):
        raise RuntimeError("Post-P1 permutation or evaluation seed changed")
    schedule = block_stream_b_schedule(POST_P1_POOL_SIZE)
    if (
        POST_P1_POOL_SIZE != 40
        or len(schedule) != 200
        or any(schedule.count(index) != 5 for index in range(POST_P1_POOL_SIZE))
    ):
        raise RuntimeError("Post-P1 Stream-B selected pool is not the expected 40")
    if CAP_CONFIG["retention"] != "release":
        raise RuntimeError("the frozen Post-P1 CAP retention is not release")


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline",
        action="store_true",
        help="Run revised-Release CAP with ordinary Adam momentum.",
    )
    mode.add_argument(
        "--conflict-momentum-restart",
        action="store_true",
        help=(
            "Restart stale Adam first-moment history when a previously bound "
            "coordinate receives a current gradient opposing that momentum."
        ),
    )


def resolve_variant(args: argparse.Namespace) -> str:
    for attribute, variant in MODE_FLAGS.items():
        if bool(getattr(args, attribute, False)):
            return variant
    raise ValueError("exactly one registered CAP-v2 mode is required")


def add_seed_arguments(parser: argparse.ArgumentParser) -> None:
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--all-seeds", action="store_true")
    seeds.add_argument("--seeds", nargs="+", type=int)


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if bool(getattr(args, "all_seeds", False)):
        return FULL_SEEDS
    values = tuple(int(seed) for seed in (getattr(args, "seeds", None) or ()))
    if not values:
        raise ValueError("--seeds must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError("--seeds may not contain duplicates")
    unknown = [seed for seed in values if seed not in FULL_SEEDS]
    if unknown:
        raise ValueError(f"unregistered seed(s): {unknown}; expected {FULL_SEEDS}")
    return values


def arm_name(variant: str) -> str:
    if variant not in MODES:
        raise ValueError(f"unknown CAP-v2 variant: {variant!r}")
    return f"cap_release_v25_{variant}"


def _resolved_cap_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(CAP_CONFIG)
    if overrides:
        unknown = set(overrides) - {"k", "T_m"}
        if unknown:
            raise ValueError(
                "final CAP retuning may override only k and T_m, got: "
                + ", ".join(sorted(unknown))
            )
        resolved.update(overrides)
    return resolved


def complete_cap_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **_resolved_cap_config(overrides),
        "betas": list(BETAS),
        "eps": ADAM_EPS,
        "c_initial": 1e-3,
        "c_refresh_every": C_REFRESH_STEPS,
        "c_refresh_rule": "global protected-coordinate median importance",
        "passive": False,
        "protection": PROTECTION,
        "decay_mode": DECAY_MODE,
        "track_bind_stats": TRACK_BIND_STATS,
        "track_proposal": False,
        "parameter_groups": "matrix weights protected; biases exempt; matrix-only decay",
    }


def build_cap_release_variant(
    model: nn.Module,
    *,
    variant: str,
    cap_overrides: Mapping[str, Any] | None = None,
) -> CAPAdam:
    if variant not in MODES:
        raise ValueError(f"unknown CAP-v2 variant: {variant!r}")
    momentum_response = MODES[variant]["momentum_response"]
    config = _resolved_cap_config(cap_overrides)
    optimizer = CAPAdam(
        cap_parameter_groups(model, float(config["weight_decay"])),
        lr=float(config["lr"]),
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=float(config["weight_decay"]),
        k=float(config["k"]),
        T_a=float(config["T_a"]),
        T_s=float(config["T_s"]),
        T_m=float(config["T_m"]),
        passive=False,
        protection=PROTECTION,
        retention="release",
        decay_mode=DECAY_MODE,
        release_update="contest_blocked",
        box_geometry="coordinate",
        momentum_response=momentum_response,
        track_bind_stats=TRACK_BIND_STATS,
    )
    return optimizer


def optimizer_factory(
    variant: str, cap_overrides: Mapping[str, Any] | None = None
):
    def factory(model: nn.Module) -> tuple[CAPAdam, dict[str, Any]]:
        return build_cap_release_variant(
            model, variant=variant, cap_overrides=cap_overrides
        ), {
            "batch_mixer": None,
            "grad_hook": None,
            "on_boundary": None,
            "is_cap": True,
        }

    return factory


def model_architecture_signature(model: nn.Module) -> str:
    payload = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in model.named_parameters()
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def model_tensor_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_aux_rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def restore_aux_rng_state(checkpoint: Mapping[str, Any]) -> None:
    payload = checkpoint.get("cap_v2_postp1")
    if not isinstance(payload, Mapping):
        raise RuntimeError("CAP-v2 checkpoint metadata is missing")
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])


def cell_rows(
    rows: Sequence[Mapping[str, Any]], arm: str, seed: int
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("arm")) == arm and int(float(row.get("seed", -1))) == seed
    ]


def trim_cell_rows(
    rows: MutableSequence[dict[str, Any]], arm: str, seed: int, keep: int
) -> None:
    kept_cell = 0
    retained: list[dict[str, Any]] = []
    for row in rows:
        matches = (
            str(row.get("arm")) == arm
            and int(float(row.get("seed", -1))) == seed
        )
        if not matches or kept_cell < keep:
            retained.append(row)
        if matches:
            kept_cell += 1
    if keep > kept_cell:
        raise RuntimeError(
            f"checkpoint expects {keep} rows for {arm}/{seed}, found {kept_cell}"
        )
    rows[:] = retained


def diagnostic_cell_rows(
    rows: Sequence[Mapping[str, Any]], variant: str, stream: str, seed: int
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("variant")) == variant
        and str(row.get("stream")) == stream
        and int(float(row.get("seed", -1))) == seed
    ]


def trim_diagnostic_rows(
    rows: MutableSequence[dict[str, Any]],
    variant: str,
    stream: str,
    seed: int,
    keep: int,
) -> None:
    kept_cell = 0
    retained: list[dict[str, Any]] = []
    for row in rows:
        matches = (
            str(row.get("variant")) == variant
            and str(row.get("stream")) == stream
            and int(float(row.get("seed", -1))) == seed
        )
        if not matches or kept_cell < keep:
            retained.append(row)
        if matches:
            kept_cell += 1
    if keep > kept_cell:
        raise RuntimeError(
            f"checkpoint expects {keep} diagnostic rows, found {kept_cell}"
        )
    rows[:] = retained


def _quantile(values: Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item())


class DiagnosticRecorder:
    """Detached CAP mechanism snapshots and their independent CSV stream."""

    def __init__(
        self,
        *,
        variant: str,
        stream: str,
        seed: int,
        arm: str,
        rows: list[dict[str, Any]],
        path: Path,
    ) -> None:
        self.variant = variant
        self.stream = stream
        self.seed = seed
        self.arm = arm
        self.rows = rows
        self.path = path
        self.previous_anchors: list[Tensor] | None = None
        self.previous_importances: list[Tensor] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous_anchors": self.previous_anchors,
            "previous_importances": self.previous_importances,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        anchors = state.get("previous_anchors")
        importances = state.get("previous_importances")
        self.previous_anchors = (
            None if anchors is None else [tensor.detach().cpu().clone() for tensor in anchors]
        )
        self.previous_importances = (
            None
            if importances is None
            else [tensor.detach().cpu().clone() for tensor in importances]
        )

    @torch.no_grad()
    def record(
        self,
        model: nn.Module,
        optimizer: CAPAdam,
        global_step: int,
        task: int,
        local_step: int,
    ) -> None:
        del model
        if global_step % DIAGNOSTIC_EVERY and local_step != stream_a.STEPS_PER_TASK:
            return
        if any(
            int(float(row["global_step"])) == global_step
            for row in diagnostic_cell_rows(
                self.rows, self.variant, self.stream, self.seed
            )
        ):
            return

        importances: list[Tensor] = []
        radii: list[Tensor] = []
        anchor_distances: list[Tensor] = []
        parameters: list[Tensor] = []
        anchors: list[Tensor] = []
        current_anchor_snapshots: list[Tensor] = []
        current_imp_snapshots: list[Tensor] = []
        bind_count = 0
        bind_numel = 0
        previous_bind_count = 0
        momentum_conflict_count = 0
        momentum_restart_count = 0
        rebind_count = 0
        for group in optimizer.param_groups:
            if not bool(group["protect"]):
                continue
            bind_count += int(group.get("last_bind_count", 0))
            bind_numel += int(group.get("last_bind_numel", 0))
            previous_bind_count += int(group.get("last_previous_bind_count", 0))
            momentum_conflict_count += int(
                group.get("last_momentum_conflict_count", 0)
            )
            momentum_restart_count += int(
                group.get("last_momentum_restart_count", 0)
            )
            rebind_count += int(group.get("last_rebind_count", 0))
            for parameter in group["params"]:
                state = optimizer.state.get(parameter)
                if not state or "anchor" not in state or "imp" not in state:
                    raise RuntimeError("CAP diagnostic state is not initialized")
                imp = state["imp"].detach().cpu()
                anchor = state["anchor"].detach().cpu()
                theta = parameter.detach().cpu()
                importances.append(imp.reshape(-1))
                radii.append(
                    (
                        protection_radius(imp, group)
                    ).reshape(-1)
                )
                anchor_distances.append((theta - anchor).abs().reshape(-1))
                parameters.append(theta.reshape(-1))
                anchors.append(anchor.reshape(-1))
                current_anchor_snapshots.append(anchor.clone())
                current_imp_snapshots.append(imp.clone())
        if not importances:
            raise RuntimeError("CAP diagnostic found no protected coordinates")

        imp_values = torch.cat(importances)
        radius_values = torch.cat(radii)
        distance_values = torch.cat(anchor_distances)
        theta_values = torch.cat(parameters)
        anchor_values = torch.cat(anchors)
        if self.previous_anchors is None:
            median_delta: float | str = ""
            p90_delta: float | str = ""
            fraction_decreased: float | str = ""
        else:
            if self.previous_importances is None or len(self.previous_anchors) != len(
                current_anchor_snapshots
            ):
                raise RuntimeError("diagnostic checkpoint snapshot shape changed")
            deltas = torch.cat(
                [
                    (current - previous).abs().reshape(-1)
                    for current, previous in zip(
                        current_anchor_snapshots, self.previous_anchors
                    )
                ]
            )
            decreased = torch.cat(
                [
                    (current < previous).reshape(-1)
                    for current, previous in zip(
                        current_imp_snapshots, self.previous_importances
                    )
                ]
            )
            median_delta = float(deltas.median().item())
            p90_delta = _quantile(deltas, 0.9)
            fraction_decreased = float(decreased.float().mean().item())

        self.rows.append(
            {
                "variant": self.variant,
                "stream": self.stream,
                "seed": self.seed,
                "global_step": global_step,
                "task": task,
                "local_step": local_step,
                "median_imp": float(imp_values.median().item()),
                "p90_imp": _quantile(imp_values, 0.9),
                "median_radius": float(radius_values.median().item()),
                "p10_radius": _quantile(radius_values, 0.1),
                "bind_rate": bind_count / max(bind_numel, 1),
                "previous_bind_count": previous_bind_count,
                "momentum_conflict_count": momentum_conflict_count,
                "momentum_restart_count": momentum_restart_count,
                "rebind_count": rebind_count,
                "momentum_conflict_given_previous_bind": (
                    momentum_conflict_count / max(previous_bind_count, 1)
                ),
                "rebind_given_current_bind": rebind_count / max(bind_count, 1),
                "rebind_given_previous_bind": (
                    rebind_count / max(previous_bind_count, 1)
                ),
                "median_anchor_distance": float(distance_values.median().item()),
                "p90_anchor_distance": _quantile(distance_values, 0.9),
                "theta_norm": float(torch.linalg.vector_norm(theta_values).item()),
                "anchor_norm": float(torch.linalg.vector_norm(anchor_values).item()),
                "median_anchor_delta_since_previous_diag": median_delta,
                "p90_anchor_delta_since_previous_diag": p90_delta,
                "fraction_imp_decreased_since_previous_diag": fraction_decreased,
                "total_imp_mass": float(imp_values.sum().item()),
            }
        )
        self.previous_anchors = current_anchor_snapshots
        self.previous_importances = current_imp_snapshots
        write_csv(self.path, self.rows, DIAGNOSTIC_FIELDS)


def checkpoint_identity(
    *,
    stream: str,
    variant: str,
    seed: int,
    model: nn.Module,
    cap_overrides: Mapping[str, Any] | None = None,
    experiment_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "stream": stream,
        "variant": variant,
        "seed": seed,
        "cap_config": complete_cap_config(cap_overrides),
        "release_update": "contest_blocked",
        "protection": "box",
        "box_geometry": "coordinate",
        "momentum_response": MODES[variant]["momentum_response"],
        "cap_adam_sha256": sha256_file(ROOT / "cap_adam_experimental.py"),
        "protocol_hash": protocol_hash(),
        "model_architecture": model_architecture_signature(model),
    }
    if experiment_context:
        identity["experiment_context"] = dict(experiment_context)
        if "method" in experiment_context:
            identity["optimizer_method"] = experiment_context["method"]
            identity["optimizer_config"] = dict(experiment_context["optimizer_settings"])
            if experiment_context["method"] != "cap":
                identity["cap_config"] = None
        for name in ("release_update", "box_geometry", "momentum_response"):
            if name in experiment_context.get("optimizer_settings", {}):
                identity[name] = experiment_context["optimizer_settings"][name]
    return identity


def validate_checkpoint_identity(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> Mapping[str, Any]:
    payload = checkpoint.get("cap_v2_postp1")
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint lacks CAP-v2 Post-P1 identity")
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "checkpoint identity mismatch for: " + ", ".join(sorted(mismatches))
        )
    for key in (
        "global_step",
        "task_index",
        "local_step",
        "cell_row_counts",
        "diagnostic_state",
        "python_rng_state",
        "numpy_rng_state",
    ):
        if key not in payload:
            raise RuntimeError(f"checkpoint lacks required CAP-v2 field: {key}")
    return payload


def build_run_config(
    *,
    stream: str,
    variant: str,
    device: torch.device,
    runner_path: Path,
    cap_overrides: Mapping[str, Any] | None = None,
    experiment_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "stream": stream,
        "variant": variant,
        "release_update": "contest_blocked",
        "protection": "box",
        "box_geometry": "coordinate",
        "momentum_response": MODES[variant]["momentum_response"],
        "cap_config": complete_cap_config(cap_overrides),
        "full_seeds": list(FULL_SEEDS),
        "post_phase1_protocol_identifier": PROTOCOL_IDENTIFIER,
        "post_phase1_source_hashes": protocol_source_hashes(),
        "post_phase1_protocol_hash": protocol_hash(),
        "evaluation_batch_size": PAPER_EVALUATION_BATCH_SIZE,
        "plasticity_observation": "full incoming current-task batch before replay mixing",
        "cap_adam_sha256": sha256_file(ROOT / "cap_adam_experimental.py"),
        "model_sha256": sha256_file(ROOT / "model.py"),
        "runner_sha256": sha256_file(runner_path),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "device": str(device),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "created_at_utc": utc_now(),
        "execution_history": [],
        "mandatory_prelaunch_test": "python -m unittest tests.test_cap_v2_postp1",
    }
    if experiment_context:
        config["experiment_context"] = dict(experiment_context)
        if "method" in experiment_context:
            config["optimizer_method"] = experiment_context["method"]
            config["optimizer_config"] = dict(experiment_context["optimizer_settings"])
            if experiment_context["method"] != "cap":
                config["cap_config"] = None
        for name in ("release_update", "box_geometry", "momentum_response"):
            if name in experiment_context.get("optimizer_settings", {}):
                config[name] = experiment_context["optimizer_settings"][name]
    return config


def lock_run_config(
    path: Path,
    proposed: Mapping[str, Any],
    *,
    requested_seeds: Sequence[int],
    resume: bool,
    device: torch.device,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable_existing = {
            key: value
            for key, value in existing.items()
            if key not in {"created_at_utc", "execution_history"}
        }
        immutable_proposed = {
            key: value
            for key, value in proposed.items()
            if key not in {"created_at_utc", "execution_history"}
        }
        if immutable_existing != immutable_proposed:
            raise RuntimeError(f"immutable run configuration mismatch: {path}")
        payload = existing
    else:
        payload = dict(proposed)
    history = list(payload.get("execution_history", []))
    history.append(
        {
            "timestamp": utc_now(),
            "command_line": list(sys.argv if argv is None else argv),
            "requested_seeds": [int(seed) for seed in requested_seeds],
            "resume": bool(resume),
            "device": str(device),
            "hostname": "anonymous",
        }
    )
    payload["execution_history"] = history
    atomic_write_json(path, payload)
    return payload


def fresh_model(seed: int, device: torch.device = torch.device("cpu")) -> MLP300:
    from experiment_e3 import seed_everything

    seed_everything(seed)
    return MLP300(input_units=784, output_units=10).to(device)


assert_authoritative_protocol()


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DIAGNOSTIC_EVERY",
    "DIAGNOSTIC_FIELDS",
    "DiagnosticRecorder",
    "EXPERIMENT_VERSION",
    "FULL_SEEDS",
    "MODES",
    "POST_P1_POOL_SIZE",
    "PROTOCOL_IDENTIFIER",
    "CAP_CONFIG",
    "add_mode_arguments",
    "add_seed_arguments",
    "arm_name",
    "assert_authoritative_protocol",
    "atomic_write_json",
    "build_run_config",
    "build_cap_release_variant",
    "block_stream_b_schedule",
    "capture_aux_rng_state",
    "cell_rows",
    "checkpoint_identity",
    "complete_cap_config",
    "diagnostic_cell_rows",
    "fresh_model",
    "lock_run_config",
    "model_architecture_signature",
    "model_tensor_hash",
    "optimizer_factory",
    "protocol_hash",
    "resolve_seeds",
    "resolve_variant",
    "restore_aux_rng_state",
    "sha256_file",
    "trim_cell_rows",
    "trim_diagnostic_rows",
    "validate_checkpoint_identity",
]
