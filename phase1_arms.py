"""Phase-1 optimizer arms and their boundary-free hook contract.

The vendored UPGD implementation is imported without modification.  All other
method-specific state is kept in a checkpointable optimizer subclass so the
stream engines can continue to save a single optimizer state dictionary.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from torch import Tensor, nn

from experiment_e3 import adamw_parameter_groups, cap_parameter_groups
from experiment_e4a import (
    ADAM_EPS,
    BETAS,
    DECAY_MODE,
    K,
    LEARNING_RATE,
    PROTECTION,
    T_A,
    T_S,
    TRACK_BIND_STATS,
    WEIGHT_DECAY,
)
from cap_adam_experimental import CAPAdam


REPOSITORY_ROOT = Path(__file__).resolve().parent
UPGD_ROOT = REPOSITORY_ROOT / "external" / "upgd"
UPGD_SOURCE = UPGD_ROOT / "core" / "optim" / "weight_upgd" / "first_order.py"
UPGD_DESIGNATION_FILE = UPGD_ROOT / "experiments" / "input_permuted_mnist.py"
UPGD_TASK_CONFIG_FILE = (
    UPGD_ROOT / "experiments" / "statistics_input_permuted_mnist.py"
)
UPGD_COMMIT = "b75e90ad4b09c28971ac9dbb902a8fd86709b28c"
UPGD_REPOSITORY = "https://github.com/mohmdelsayed/upgd"
UPGD_LICENSE = "MIT"
UPGD_CLASS = "FirstOrderGlobalUPGD"
UPGD_DESIGNATION = "experiments/input_permuted_mnist.py:54"
UPGD_TASK_CONFIG_SOURCE = (
    "experiments/statistics_input_permuted_mnist.py:33-40,135"
)
UPGD_AMENDMENT = "P1-A3"
# The completed tuned-config artifact remains the immutable P1-A5/v7 schema.
# P1-A7 changes only final-stream deployment and grading.
PHASE1_AMENDMENT = "P1-A5"
FINAL_STREAM_AMENDMENT = "P1-A7"
RETENTION_FOCUSED_SUFFIX = "_retention_focused"
CONDITIONAL_RETUNE_ARM = "cap_release_retention_focused"
P1_FULL_SEEDS = (42, 123, 2026, 451, 789)
P1_A7_PRIMARY_CAP_CONFIG = {
    "k": 7e-5,
    "T_m": 10_000,
    "T_a": 2_000,
    "retention": "release",
}
P1_A7_RETUNE_CAP_CONFIG = {
    "k": 7e-5,
    "T_m": 50_000,
    "T_a": 2_000,
    "retention": "release",
}
RETENTION_FOCUSED_LABEL = "sensitivity — retention-focused selection"
TUNED_METHODS = (
    "adamw",
    "adam_replay",
    "upgd",
    "ewc_oracle",
    "l2_init",
    "cap_release",
)
P1_A3_DISCLOSURE = (
    "P1-A3 is a post-smoke, pre-tuning/pre-Stream-B integration amendment: "
    "the numerically stable README-default UPGD smoke failed all three "
    "learning rates, so the official published input-permuted-MNIST task "
    "configuration now supplies the UPGD baseline; the original failure is "
    "preserved and no threshold, opponent, seed, stream, model, selection "
    "formula, or non-UPGD grid changed."
)
P1_A4_DISCLOSURE = (
    "P1-A4 is a post-v4-tuning, pre-Stream-B amendment: the completed "
    "plasticity-only v4 winners remain primary, while a fresh-model, "
    "disjoint recurrent validation campaign selects method-relative "
    "retention-focused sensitivity endpoints."
)
P1_A5_DISCLOSURE = (
    "P1-A5 evaluates each differing retention-focused endpoint on both "
    "final streams; primary grading remains on the v4 winners, and the "
    "endpoint-matched flexibility analysis is permanently descriptive."
)
P1_A6_DISCLOSURE = (
    "P1-A6 introduced a guarded compute descope before Stream B; P1-A7 "
    "supersedes its roster while retaining the measured l2_init descope guard."
)
P1_A7_DISCLOSURE = (
    "P1-A7 is a post-validation, pre-Stream-B roster amendment: seven "
    "five-seed primary or primary-derived arms run unconditionally, all "
    "retention-focused endpoints leave the unconditional final streams, and "
    "the frozen CAP retention-focused endpoint may run once on both streams "
    "only after a valid primary kill-gate failure."
)
P1_A7_FREEZE_STATEMENT = (
    "The primary selection rule and configuration were fixed before any "
    "Stream-B result was seen; tuning and validation results had been seen, "
    "but Stream-B results had not."
)
P1_A7_FAILURE_BRANCH_STATEMENT = (
    "No configuration other than cap_release_retention_focused with the "
    "artifact-frozen P1-A4 retention-aware CAP configuration can enter the "
    "single conditional failure branch."
)
UPGD_SOURCE_SHA256 = (
    "d976b8ca34cced48b79a2002a638f717b72ad00368f6eeb7d1d424fa5724cdef"
)
UPGD_DESIGNATION_SHA256 = (
    "2b4aa5bd992ed4e178947520531aa937727b42d6a1a2b5e578f87b7f86787299"
)
UPGD_TASK_CONFIG_SHA256 = (
    "4abadc732e88b873e6e5ecc6e8dff1de9ea5e2fee6a576b42263021e46fea728"
)

# P1-A3 replaces the generic standalone README values with the official
# repository's published task-specific input-permuted-MNIST configuration.
UPGD_DEFAULTS = {
    "lr": 0.01,
    "weight_decay": 0.01,
    "beta_utility": 0.9999,
    "sigma": 0.1,
}
UPGD_GENERIC_README_DEFAULTS = {
    "lr": 1e-5,
    "weight_decay": 0.001,
    "beta_utility": 0.999,
    "sigma": 0.001,
}

REPLAY_CAPACITY = 5_000
EWC_FISHER_TIMESCALE = 500.0

Hook = Optional[Callable[..., Any]]


def _verify_upgd_citations() -> None:
    pinned_files = (
        (UPGD_DESIGNATION_FILE, UPGD_DESIGNATION_SHA256),
        (UPGD_TASK_CONFIG_FILE, UPGD_TASK_CONFIG_SHA256),
    )
    for path, expected_sha256 in pinned_files:
        if not path.exists():
            raise FileNotFoundError(f"vendored UPGD citation source is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(
                f"vendored UPGD citation source changed from the pinned commit: {path}"
            )
    designation_lines = UPGD_DESIGNATION_FILE.read_text(encoding="utf-8").splitlines()
    task_lines = UPGD_TASK_CONFIG_FILE.read_text(encoding="utf-8").splitlines()
    if designation_lines[53].strip() != "FirstOrderGlobalUPGDLearner(),":
        raise RuntimeError("UPGD variant designation moved from the pinned line 54")
    expected_task_lines = {
        32: "upgd2_grid = GridSearch(",
        34: "lr=[0.01],",
        35: "beta_utility=[0.9999],",
        36: "sigma=[0.1],",
        37: "weight_decay=[0.01],",
        134: "FirstOrderGlobalUPGDLearner(),",
    }
    if any(task_lines[index].strip() != expected for index, expected in expected_task_lines.items()):
        raise RuntimeError(
            "UPGD published task configuration no longer matches pinned lines 33-40,135"
        )


def upgd_provenance() -> dict[str, Any]:
    """Return the immutable provenance recorded in every Phase-1 run."""
    _verify_upgd_citations()
    return {
        "repository": UPGD_REPOSITORY,
        "commit": UPGD_COMMIT,
        "license": UPGD_LICENSE,
        "class_name": UPGD_CLASS,
        "source": UPGD_SOURCE.relative_to(REPOSITORY_ROOT).as_posix() + ":74",
        "source_sha256": UPGD_SOURCE_SHA256,
        "designation": UPGD_DESIGNATION,
        "designation_source_sha256": UPGD_DESIGNATION_SHA256,
        "designation_paraphrase": (
            "the input-permuted-MNIST experiment lists "
            "FirstOrderGlobalUPGDLearner as the primary protecting UPGD arm"
        ),
        "task_config_source": UPGD_TASK_CONFIG_SOURCE,
        "task_config_source_sha256": UPGD_TASK_CONFIG_SHA256,
        "task_config_paraphrase": (
            "the published statistics configuration pairs "
            "FirstOrderGlobalUPGDLearner with lr=0.01, weight_decay=0.01, "
            "beta_utility=0.9999, and sigma=0.1"
        ),
        "generic_defaults_source": "external/upgd/README.md:9",
        "generic_readme_defaults": dict(UPGD_GENERIC_README_DEFAULTS),
        "defaults_precedence": (
            "P1-A3 task-specific published input-permuted-MNIST values "
            "supersede the generic standalone README defaults"
        ),
        "amendment": UPGD_AMENDMENT,
        "resolved_default_kwargs": dict(UPGD_DEFAULTS),
        "harness_adaptation": (
            "step executes under the model parameter's torch.device context "
            "so the upstream scalar factory is CUDA-cap; upstream step "
            "logic is otherwise unchanged"
        ),
    }


def _serialize_extra(value: Any) -> Any:
    if isinstance(value, torch.Generator):
        return {"__phase1_generator_state__": value.get_state()}
    if isinstance(value, dict):
        return {key: _serialize_extra(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_extra(item) for item in value]
    return value


def _deserialize_extra(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__phase1_generator_state__"}:
        generator = torch.Generator()
        generator.set_state(value["__phase1_generator_state__"].cpu())
        return generator
    if isinstance(value, dict):
        return {key: _deserialize_extra(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_extra(item) for item in value]
    return value


class Phase1AdamW(torch.optim.AdamW):
    """AdamW carrying hook state through ordinary optimizer checkpoints."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phase1_state: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["phase1_state"] = _serialize_extra(self.phase1_state)
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        base_state = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(base_state)
        self.phase1_state = _deserialize_extra(
            state_dict.get("phase1_state", {})
        )


def _adamw(
    model: nn.Module, lr: float, weight_decay: float = WEIGHT_DECAY
) -> Phase1AdamW:
    return Phase1AdamW(
        adamw_parameter_groups(model, float(weight_decay)),
        lr=float(lr),
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=float(weight_decay),
    )


def _load_upgd_class():
    if not UPGD_SOURCE.exists():
        raise FileNotFoundError(
            f"vendored official UPGD source is missing: {UPGD_SOURCE}"
        )
    observed_sha256 = hashlib.sha256(UPGD_SOURCE.read_bytes()).hexdigest()
    if observed_sha256 != UPGD_SOURCE_SHA256:
        raise RuntimeError(
            "vendored UPGD source does not match the pinned official commit"
        )
    module_name = "phase1_vendored_upgd_first_order"
    spec = importlib.util.spec_from_file_location(module_name, UPGD_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import vendored UPGD source: {UPGD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, UPGD_CLASS)


def _device_adapted_upgd_class():
    """Wrap the official step solely to place its scalar on the model device."""
    upstream_class = _load_upgd_class()

    class DeviceAdaptedFirstOrderGlobalUPGD(upstream_class):
        def step(self, closure=None):
            del closure
            first_parameter = next(
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            )
            # Upstream constructs ``torch.tensor(-inf)`` without a device.
            # The device context makes that one factory CUDA-cap while
            # delegating every algorithmic operation to the official source.
            with torch.device(first_parameter.device):
                return super().step()

    DeviceAdaptedFirstOrderGlobalUPGD.__name__ = UPGD_CLASS
    DeviceAdaptedFirstOrderGlobalUPGD.__qualname__ = UPGD_CLASS
    return DeviceAdaptedFirstOrderGlobalUPGD


def method_for_arm(name: str) -> str:
    if name == "upgd_corrected":
        return "upgd"
    if name.endswith("_retention"):
        name = name[: -len("_retention")]
    base = (
        name[: -len(RETENTION_FOCUSED_SUFFIX)]
        if name.endswith(RETENTION_FOCUSED_SUFFIX)
        else name
    )
    if base in {"cap_release", "cap_release_tuned", "cap_release_10000"}:
        return "cap_release"
    if base in {"cap_ratchet", "cap_leak"}:
        return "cap_release"
    return base


def endpoint_for_arm(name: str) -> str:
    return (
        "retention_focused"
        if name.endswith(RETENTION_FOCUSED_SUFFIX)
        else "primary"
    )


def _config_entry(
    tuned_configs: Mapping[str, Any],
    method: str,
    endpoint: str = "primary",
) -> Mapping[str, Any]:
    methods = tuned_configs.get("methods", tuned_configs)
    if method not in methods:
        raise KeyError(f"tuned-config artifact has no {method!r} entry")
    entry = methods[method]
    if not isinstance(entry, Mapping):
        raise TypeError(f"invalid tuned-config entry for {method!r}")
    endpoint_entry = entry.get(endpoint)
    if isinstance(endpoint_entry, Mapping):
        config = endpoint_entry.get("config", endpoint_entry)
    else:
        # Compatibility is intentionally limited to reading the preserved v4
        # artifact during the P1-A4 migration.  Downstream loaders require v7.
        config = entry.get("config", entry.get("winning_config", entry))
    if not isinstance(config, Mapping):
        raise TypeError(f"invalid winning config for {method!r}")
    return config


def load_tuned_configs(path: Path) -> dict[str, Any]:
    """Load and minimally validate the sole downstream config artifact."""
    if not path.exists():
        raise FileNotFoundError(
            "Phase 1 tuned configs are required before this run: " f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 7 or payload.get("amendment") != PHASE1_AMENDMENT:
        raise ValueError("tuned-config artifact is not the P1-A5/v7 artifact")
    amendments = payload.get("amendments", [])
    if not all(tag in amendments for tag in ("P1-A3", "P1-A4", "P1-A5")):
        raise ValueError("tuned-config artifact is missing a Phase-1 amendment tag")
    provenance = payload.get("upgd")
    if not isinstance(provenance, Mapping):
        raise KeyError("tuned-config artifact is missing UPGD provenance")
    if provenance.get("task_config_source") != UPGD_TASK_CONFIG_SOURCE:
        raise ValueError(
            "tuned-config artifact does not pin the P1-A3 task config source"
        )
    if provenance.get("resolved_default_kwargs") != UPGD_DEFAULTS:
        raise ValueError(
            "tuned-config artifact has incorrect P1-A3 UPGD defaults"
        )
    if not isinstance(payload.get("pre_amendment_upgd_smoke_gate"), Mapping):
        raise KeyError(
            "tuned-config artifact is missing the preserved pre-amendment "
            "UPGD smoke result"
        )
    sensitivity = payload.get("sensitivity_arms")
    if not isinstance(sensitivity, Mapping):
        raise KeyError("tuned-config artifact is missing sensitivity_arms")
    for method in TUNED_METHODS:
        primary = _config_entry(payload, method, "primary")
        retention = _config_entry(payload, method, "retention_focused")
        entry = payload["methods"][method]
        if not isinstance(entry.get("tried_configs"), list):
            raise KeyError(f"v7 artifact is missing tried_configs for {method!r}")
        coincide = dict(primary) == dict(retention)
        if bool(entry.get("endpoints_coincide")) != coincide:
            raise ValueError(f"inconsistent endpoint coincidence for {method!r}")
        arm = method + RETENTION_FOCUSED_SUFFIX
        if coincide and arm in sensitivity:
            raise ValueError(f"coincident endpoint unexpectedly registered: {arm}")
        if not coincide:
            record = sensitivity.get(arm)
            if not isinstance(record, Mapping) or record.get("method") != method:
                raise KeyError(f"missing resolved sensitivity arm {arm!r}")
            if record.get("label") != RETENTION_FOCUSED_LABEL:
                raise ValueError(f"missing sensitivity label for {arm!r}")
            if record.get("config") != dict(retention):
                raise ValueError(f"resolved sensitivity config mismatch for {arm!r}")
            expected_seeds = (
                [42, 123, 2026, 451, 789]
                if method == "cap_release"
                else [42, 123, 2026]
            )
            if list(record.get("seeds", [])) != expected_seeds:
                raise ValueError(f"incorrect P1-A5 seed set for {arm!r}")
    derived = payload.get("derived_configs")
    if not isinstance(derived, Mapping):
        raise KeyError("tuned-config artifact is missing derived_configs")
    for arm in ("cap_ratchet", "cap_leak"):
        if not isinstance(derived.get(arm), Mapping):
            raise KeyError(f"tuned-config artifact is missing {arm!r}")
    expected_release_arm = (
        "cap_release_10000"
        if cap_release_matches_core(payload)
        else "cap_release_tuned"
    )
    if payload.get("stream_a_release_arm") != expected_release_arm:
        raise ValueError(
            "tuned-config artifact has an inconsistent Stream-A matching rule"
        )
    return payload


def validation_metric_for_endpoint(
    tuned_configs: Mapping[str, Any],
    method: str,
    endpoint: str,
) -> float:
    """Return a frozen validation metric without reconstructing a config."""
    methods = tuned_configs.get("methods")
    if not isinstance(methods, Mapping) or method not in methods:
        raise KeyError(f"tuned-config artifact has no {method!r} entry")
    entry = methods[method]
    endpoint_entry = entry.get(endpoint)
    if not isinstance(endpoint_entry, Mapping):
        raise KeyError(f"{method!r} has no {endpoint!r} endpoint")
    index = int(endpoint_entry["config_index"])
    matches = [
        item
        for item in entry.get("tried_configs", [])
        if int(item["config_index"]) == index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{method!r}/{endpoint!r} does not resolve to exactly one tried config"
        )
    value = matches[0].get("validation_retention_metric")
    if value is None:
        raise ValueError(f"{method!r}/{endpoint!r} has no validation metric")
    return float(value)


def l2_descope_guard(tuned_configs: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen P1-A6(a)/P1-A7(b) validation-scale guard."""
    l2_value = validation_metric_for_endpoint(tuned_configs, "l2_init", "primary")
    opponents = {
        method: validation_metric_for_endpoint(tuned_configs, method, "primary")
        for method in ("adamw", "upgd")
    }
    binding = max(opponents, key=opponents.get)
    binding_value = opponents[binding]
    passed = l2_value <= binding_value
    return {
        "l2_init_primary_validation_retention": l2_value,
        "opponents": opponents,
        "binding_opponent": binding,
        "binding_value": binding_value,
        "passed": passed,
        "verdict": "PASS" if passed else "L2INIT-DESCOPE-INVALID",
    }


def validate_p1_a7_artifact(tuned_configs: Mapping[str, Any]) -> None:
    """Require the exact post-validation configurations frozen by P1-A7."""
    primary = dict(_config_entry(tuned_configs, "cap_release", "primary"))
    retention = dict(
        _config_entry(tuned_configs, "cap_release", "retention_focused")
    )
    if primary != P1_A7_PRIMARY_CAP_CONFIG:
        raise ValueError(
            "P1-A7 requires the observed v4-primary CAP configuration "
            f"{P1_A7_PRIMARY_CAP_CONFIG}, found {primary}"
        )
    if retention != P1_A7_RETUNE_CAP_CONFIG:
        raise ValueError(
            "P1-A7 requires the frozen P1-A4 CAP re-tune configuration "
            f"{P1_A7_RETUNE_CAP_CONFIG}, found {retention}"
        )
    if tuned_configs.get("stream_a_release_arm") != "cap_release_tuned":
        raise ValueError("P1-A7 requires cap_release_tuned on Stream A")
    if tuned_configs.get("kill_gate_retune_arm") != CONDITIONAL_RETUNE_ARM:
        raise ValueError("P1-A7 conditional re-tune arm is not frozen correctly")
    derived = tuned_configs.get("derived_configs")
    if not isinstance(derived, Mapping):
        raise ValueError("P1-A7 requires artifact-owned CAP derived configs")
    for arm, policy in (("cap_ratchet", "ratchet"), ("cap_leak", "leak")):
        expected = {**P1_A7_PRIMARY_CAP_CONFIG, "retention": policy}
        if derived.get(arm) != expected:
            raise ValueError(
                "P1-A7 CAP triad must differ only by retention policy: "
                f"{arm}={derived.get(arm)!r}, expected {expected!r}"
            )
    record = tuned_configs.get("sensitivity_arms", {}).get(
        CONDITIONAL_RETUNE_ARM
    )
    if (
        not isinstance(record, Mapping)
        or record.get("config") != P1_A7_RETUNE_CAP_CONFIG
        or tuple(int(seed) for seed in record.get("seeds", [])) != P1_FULL_SEEDS
    ):
        raise ValueError("P1-A7 CAP re-tune registry is inconsistent")
    guard = l2_descope_guard(tuned_configs)
    expected_guard_values = {
        "l2_init": 0.2773375,
        "adamw": 0.9130625,
        "upgd": 0.2261,
    }
    observed_guard_values = {
        "l2_init": guard["l2_init_primary_validation_retention"],
        **guard["opponents"],
    }
    if not all(
        math.isclose(
            observed_guard_values[name],
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, expected in expected_guard_values.items()
    ):
        raise ValueError(
            "P1-A7 l2_init descope guard values do not match the frozen "
            f"validation record: {observed_guard_values}"
        )
    if not guard["passed"]:
        raise ValueError("L2INIT-DESCOPE-INVALID")


def load_conditional_retune_request(
    path: Path,
    *,
    tuned_config_path: Path,
    tuned_configs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the grading-owned authorization for the conditional branch."""
    if not path.exists():
        raise FileNotFoundError(
            "conditional re-tune is not authorized; run phase1_grade.py after "
            f"both unconditional streams complete: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(tuned_config_path.read_bytes()).hexdigest()
    expected_config = dict(
        _config_entry(tuned_configs, "cap_release", "retention_focused")
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("amendment") != FINAL_STREAM_AMENDMENT
        or payload.get("status") != "CONDITIONAL-RETUNE-REQUIRED"
        or payload.get("arm") != CONDITIONAL_RETUNE_ARM
        or payload.get("config") != expected_config
        or tuple(int(seed) for seed in payload.get("seeds", [])) != P1_FULL_SEEDS
        or payload.get("tuned_config_sha256") != expected_hash
        or payload.get("primary_kill_gate_verdict") != "FAIL"
    ):
        raise ValueError("conditional re-tune request is invalid or stale")
    return payload


def _resolved_config(
    name: str, tuned_configs: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve either a legacy core config or an artifact-owned config."""
    if tuned_configs is not None:
        if name in {"cap_ratchet", "cap_leak"}:
            derived = tuned_configs.get("derived_configs")
            if not isinstance(derived, Mapping) or not isinstance(
                derived.get(name), Mapping
            ):
                raise KeyError(
                    f"tuned-config artifact has no derived config for {name!r}"
                )
            config = dict(derived[name])
        else:
            config = dict(
                _config_entry(
                    tuned_configs,
                    method_for_arm(name),
                    endpoint_for_arm(name),
                )
            )
        if method_for_arm(name) == "cap_release" and name not in {
            "cap_ratchet",
            "cap_leak",
        }:
            config.setdefault("retention", "release")
        return config

    if name == "adamw":
        return {"lr": LEARNING_RATE}
    if name in {"cap_leak", "cap_ratchet", "cap_release_10000"}:
        return {
            "k": K,
            "T_m": 10_000,
            "T_a": T_A,
            "retention": name.removeprefix("cap_").split("_", 1)[0],
        }
    raise ValueError(f"arm {name!r} requires the tuned-config artifact")


def _build_replay_mixer(
    optimizer: Phase1AdamW, replay_count: int, seed: int
) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    if replay_count < 1 or replay_count >= 128:
        raise ValueError(f"invalid replay_count: {replay_count}")
    optimizer.phase1_state["replay"] = {
        "capacity": REPLAY_CAPACITY,
        "replay_count": int(replay_count),
        "short_batch_adjustments": 0,
        "seen": 0,
        "size": 0,
        "inputs": None,
        "targets": None,
        "generator": torch.Generator().manual_seed(seed + 70_000),
    }

    def uniform_slots(example_numbers: Tensor, generator: torch.Generator) -> Tensor:
        """Draw exact independent integers in [0, n) for varying n."""
        bounds = example_numbers.to(dtype=torch.int64)
        slots = torch.empty_like(bounds)
        pending = torch.arange(bounds.numel(), dtype=torch.int64)
        random_high = torch.iinfo(torch.int64).max
        while pending.numel():
            pending_bounds = bounds.index_select(0, pending)
            random_values = torch.randint(
                0,
                random_high,
                (pending.numel(),),
                dtype=torch.int64,
                generator=generator,
            )
            limits = torch.div(
                random_high, pending_bounds, rounding_mode="floor"
            ) * pending_bounds
            accepted = random_values < limits
            accepted_positions = pending[accepted]
            slots[accepted_positions] = (
                random_values[accepted] % pending_bounds[accepted]
            )
            pending = pending[~accepted]
        return slots

    def batch_mixer(inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        state = optimizer.phase1_state["replay"]
        configured_count = int(state["replay_count"])
        # The MNIST loader preserves its final, short batch.  The registered
        # cap (batch size - 1) is therefore applied to the *actual* batch as
        # well as the nominal batch of 128, so every step retains at least one
        # fresh example.  Full batches still use the exact configured count.
        count = min(configured_count, max(int(inputs.shape[0]) - 1, 0))
        if count != configured_count:
            state["short_batch_adjustments"] = (
                int(state.get("short_batch_adjustments", 0)) + 1
            )
        size = int(state["size"])
        generator = state["generator"]
        if size >= count:
            indices = torch.randperm(size, generator=generator)[:count]
            replay_inputs = state["inputs"].index_select(0, indices).to(
                inputs.device, non_blocking=True
            )
            replay_targets = state["targets"].index_select(0, indices).to(
                targets.device, non_blocking=True
            )
            keep = inputs.shape[0] - count
            if keep <= 0:
                raise RuntimeError(
                    "internal replay invariant failed: no fresh example remains"
                )
            fresh_inputs = inputs[:keep]
            fresh_targets = targets[:keep]
            mixed_inputs = torch.cat((fresh_inputs, replay_inputs), dim=0)
            mixed_targets = torch.cat((fresh_targets, replay_targets), dim=0)
        else:
            fresh_inputs, fresh_targets = inputs, targets
            mixed_inputs, mixed_targets = inputs, targets

        cpu_inputs = fresh_inputs.detach().to("cpu")
        cpu_targets = fresh_targets.detach().to("cpu")
        if state["inputs"] is None:
            state["inputs"] = torch.empty(
                (REPLAY_CAPACITY, *cpu_inputs.shape[1:]), dtype=cpu_inputs.dtype
            )
            state["targets"] = torch.empty(
                (REPLAY_CAPACITY,), dtype=cpu_targets.dtype
            )

        offset = 0
        fill = min(REPLAY_CAPACITY - size, cpu_inputs.shape[0])
        if fill:
            state["inputs"][size : size + fill].copy_(cpu_inputs[:fill])
            state["targets"][size : size + fill].copy_(cpu_targets[:fill])
            size += fill
            offset = fill

        seen_before = int(state["seen"])
        state["seen"] = seen_before + cpu_inputs.shape[0]
        if offset < cpu_inputs.shape[0]:
            example_numbers = torch.arange(
                seen_before + offset + 1,
                seen_before + cpu_inputs.shape[0] + 1,
                dtype=torch.int64,
            )
            slots = uniform_slots(example_numbers, generator)
            candidates = torch.nonzero(slots < REPLAY_CAPACITY).flatten()
            # Sequential assignment preserves the specified last-write result
            # when two examples in one fresh batch select the same slot.
            for relative in candidates.tolist():
                source = offset + relative
                destination = int(slots[relative].item())
                state["inputs"][destination].copy_(cpu_inputs[source])
                state["targets"][destination].copy_(cpu_targets[source])
        state["size"] = size
        return mixed_inputs, mixed_targets

    return batch_mixer


def _build_ewc_hooks(
    optimizer: Phase1AdamW, model: nn.Module, lam: float, gamma: float
) -> tuple[Callable[[nn.Module], None], Callable[[nn.Module], None]]:
    if not (0.0 < lam <= 1e6) or not (0.0 <= gamma <= 1.0):
        raise ValueError(f"invalid EWC config lambda={lam}, gamma={gamma}")
    optimizer.phase1_state["ewc"] = {
        "lambda": float(lam),
        "gamma": float(gamma),
        "F_ema": {name: torch.zeros_like(p) for name, p in model.named_parameters()},
        "F_cons": {name: torch.zeros_like(p) for name, p in model.named_parameters()},
        "theta_star": None,
    }

    @torch.no_grad()
    def grad_hook(hook_model: nn.Module) -> None:
        state = optimizer.phase1_state["ewc"]
        theta_star = state["theta_star"]
        for name, parameter in hook_model.named_parameters():
            if parameter.grad is None:
                continue
            fisher = state["F_ema"][name]
            if fisher.device != parameter.device:
                fisher = fisher.to(parameter.device)
                state["F_ema"][name] = fisher
                state["F_cons"][name] = state["F_cons"][name].to(parameter.device)
                if theta_star is not None:
                    theta_star[name] = theta_star[name].to(parameter.device)
            raw_gradient = parameter.grad.detach()
            fisher.mul_(1.0 - 1.0 / EWC_FISHER_TIMESCALE).addcmul_(
                raw_gradient,
                raw_gradient,
                value=1.0 / EWC_FISHER_TIMESCALE,
            )
            if theta_star is not None:
                parameter.grad.add_(
                    state["F_cons"][name]
                    * (parameter.detach() - theta_star[name]),
                    alpha=float(state["lambda"]),
                )

    @torch.no_grad()
    def on_boundary(hook_model: nn.Module) -> None:
        state = optimizer.phase1_state["ewc"]
        state["theta_star"] = {
            name: parameter.detach().clone()
            for name, parameter in hook_model.named_parameters()
        }
        for name in state["F_ema"]:
            state["F_cons"][name].mul_(float(state["gamma"])).add_(
                state["F_ema"][name]
            )
            state["F_ema"][name].zero_()

    return grad_hook, on_boundary


def _build_l2_hook(
    optimizer: Phase1AdamW, model: nn.Module, lam: float
) -> Callable[[nn.Module], None]:
    if lam not in (1e-4, 1e-3, 1e-2):
        raise ValueError(f"invalid l2_init lambda: {lam}")
    optimizer.phase1_state["l2_init"] = {
        "lambda": float(lam),
        "theta0": {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        },
    }

    @torch.no_grad()
    def grad_hook(hook_model: nn.Module) -> None:
        state = optimizer.phase1_state["l2_init"]
        for name, parameter in hook_model.named_parameters():
            if parameter.grad is None:
                continue
            theta0 = state["theta0"][name]
            if theta0.device != parameter.device:
                theta0 = theta0.to(parameter.device)
                state["theta0"][name] = theta0
            parameter.grad.add_(
                parameter.detach() - theta0, alpha=float(state["lambda"])
            )

    return grad_hook


def build_arm(
    name: str,
    model: nn.Module,
    tuned_configs: Optional[Mapping[str, Any]],
    seed: int,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Build an optimizer and the exact four-entry Phase-1 hook mapping."""
    del device  # construction follows the model parameters' existing devices
    config = _resolved_config(name, tuned_configs)
    method = method_for_arm(name)
    batch_mixer: Hook = None
    grad_hook: Hook = None
    on_boundary: Hook = None
    is_cap = False

    allowed_keys = {
        "adamw": {"lr", "weight_decay"},
        "adam_replay": {
            "lr",
            "weight_decay",
            "replay_count",
            "buffer_capacity",
        },
        "upgd": set(UPGD_DEFAULTS),
        "ewc_oracle": {"lr", "weight_decay", "lambda", "gamma"},
        "l2_init": {"lambda", "lr", "weight_decay"},
        "cap_release": {
            "lr",
            "weight_decay",
            "k",
            "T_s",
            "T_a",
            "T_m",
            "retention",
        },
    }
    unknown = set(config) - allowed_keys.get(method, set())
    if unknown:
        raise ValueError(
            f"unconsumed configuration key(s) for {name}: "
            + ", ".join(sorted(unknown))
        )

    if method == "adamw":
        optimizer: torch.optim.Optimizer = _adamw(
            model,
            float(config["lr"]),
            float(config.get("weight_decay", WEIGHT_DECAY)),
        )
    elif method == "adam_replay":
        capacity = int(config.get("buffer_capacity", REPLAY_CAPACITY))
        if capacity != REPLAY_CAPACITY:
            raise ValueError(
                "adam_replay buffer_capacity is a fixed protocol constant "
                f"at {REPLAY_CAPACITY}, got {capacity}"
            )
        optimizer = _adamw(
            model,
            float(config["lr"]),
            float(config.get("weight_decay", WEIGHT_DECAY)),
        )
        batch_mixer = _build_replay_mixer(
            optimizer, int(config["replay_count"]), seed
        )
    elif method == "upgd":
        kwargs = dict(UPGD_DEFAULTS)
        kwargs.update(
            {
                key: float(config[key])
                for key in UPGD_DEFAULTS
                if key in config
            }
        )
        optimizer = _device_adapted_upgd_class()(
            list(model.named_parameters()), **kwargs
        )
    elif method == "ewc_oracle":
        optimizer = _adamw(
            model,
            float(config.get("lr", LEARNING_RATE)),
            float(config.get("weight_decay", WEIGHT_DECAY)),
        )
        grad_hook, on_boundary = _build_ewc_hooks(
            optimizer,
            model,
            float(config["lambda"]),
            float(config["gamma"]),
        )
    elif method == "l2_init":
        optimizer = _adamw(
            model,
            float(config.get("lr", LEARNING_RATE)),
            float(config.get("weight_decay", WEIGHT_DECAY)),
        )
        grad_hook = _build_l2_hook(
            optimizer, model, float(config["lambda"])
        )
    elif method == "cap_release":
        retention = str(config["retention"])
        if retention not in {"release", "ratchet", "leak"}:
            raise ValueError(f"invalid CAP retention policy: {retention}")
        weight_decay = float(config.get("weight_decay", WEIGHT_DECAY))
        optimizer = CAPAdam(
            cap_parameter_groups(model, weight_decay),
            lr=float(config.get("lr", LEARNING_RATE)),
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=weight_decay,
            k=float(config["k"]),
            T_s=float(config.get("T_s", T_S)),
            T_a=float(config["T_a"]),
            T_m=float(config.get("T_m", 10_000)),
            passive=False,
            protection=PROTECTION,
            retention=retention,
            decay_mode=DECAY_MODE,
            track_bind_stats=TRACK_BIND_STATS,
        )
        is_cap = True
    else:
        raise ValueError(f"unknown Phase-1 arm: {name}")

    hooks = {
        "batch_mixer": batch_mixer,
        "grad_hook": grad_hook,
        "on_boundary": on_boundary,
        "is_cap": is_cap,
    }
    return optimizer, hooks


def cap_release_matches_core(tuned_configs: Mapping[str, Any]) -> bool:
    config = _config_entry(tuned_configs, "cap_release", "primary")
    return (
        math.isclose(float(config["k"]), 3e-5)
        and int(float(config["T_m"])) == 10_000
        and int(float(config["T_a"])) == 2_000
    )


def resolved_config_for_arm(
    name: str, tuned_configs: Mapping[str, Any]
) -> dict[str, Any]:
    """Public, read-only config resolver for run manifests and reports."""
    return _resolved_config(name, tuned_configs)


def retention_focused_arms(tuned_configs: Mapping[str, Any]) -> tuple[str, ...]:
    records = tuned_configs.get("sensitivity_arms", {})
    if not isinstance(records, Mapping):
        raise TypeError("sensitivity_arms must be a mapping")
    expected = [
        method + RETENTION_FOCUSED_SUFFIX
        for method in TUNED_METHODS
        if not bool(tuned_configs["methods"][method]["endpoints_coincide"])
    ]
    if set(records) != set(expected):
        raise ValueError("sensitivity-arm registry does not match endpoint selections")
    return tuple(expected)


__all__ = [
    "CONDITIONAL_RETUNE_ARM",
    "FINAL_STREAM_AMENDMENT",
    "Phase1AdamW",
    "PHASE1_AMENDMENT",
    "P1_A6_DISCLOSURE",
    "P1_A7_DISCLOSURE",
    "P1_A7_FAILURE_BRANCH_STATEMENT",
    "P1_A7_FREEZE_STATEMENT",
    "P1_A7_PRIMARY_CAP_CONFIG",
    "P1_A7_RETUNE_CAP_CONFIG",
    "P1_FULL_SEEDS",
    "P1_A3_DISCLOSURE",
    "P1_A4_DISCLOSURE",
    "P1_A5_DISCLOSURE",
    "RETENTION_FOCUSED_LABEL",
    "RETENTION_FOCUSED_SUFFIX",
    "TUNED_METHODS",
    "UPGD_AMENDMENT",
    "UPGD_DEFAULTS",
    "UPGD_GENERIC_README_DEFAULTS",
    "UPGD_TASK_CONFIG_SOURCE",
    "build_arm",
    "endpoint_for_arm",
    "l2_descope_guard",
    "load_conditional_retune_request",
    "load_tuned_configs",
    "method_for_arm",
    "resolved_config_for_arm",
    "retention_focused_arms",
    "cap_release_matches_core",
    "upgd_provenance",
    "validate_p1_a7_artifact",
    "validation_metric_for_endpoint",
]
