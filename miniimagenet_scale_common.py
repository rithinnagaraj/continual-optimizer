"""Shared protocol mechanics for the frozen-ResNet-50 mini-ImageNet study.

This module intentionally contains no dataset download logic.  It provides the
frozen stream, model, optimizer, metric, hashing, and atomic-I/O contracts used
by preparation, training, analysis, and tests.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from cap_adam_experimental import CAPAdam


EXPERIMENT_VERSION = "miniimagenet-frozen-resnet50-head-v1"
MODEL_NAME = "frozen_resnet50_imagenet1k_v2_mlp_head"
BACKBONE_WEIGHTS = "ResNet50_Weights.IMAGENET1K_V2"
DEFAULT_HEAD_HIDDEN_DIM = 300
DATA_SPLIT_SEED = 2026
DOMAIN_SEVERITY = 3
BATCH_SIZE = 128
UPDATES_PER_SEGMENT = 200
SAMPLES_PER_CLASS_PER_SEGMENT = 256
NUM_CLASSES = 100
TRAIN_PER_CLASS = 500
EVAL_PER_CLASS = 100
FAST_EVAL_PER_CLASS = 20
CYCLES = 5
FINAL_SEEDS = (42, 123, 2026, 451, 789)
DEVELOPMENT_SEEDS = (31415, 27182)
METHODS = ("adamw", "cap", "upgd", "ewc")
DOMAINS = (
    "clean",
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "fog",
    "frost",
    "contrast",
    "jpeg_compression",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed_seed(*parts: object) -> int:
    """Return a stable unsigned 63-bit seed from arbitrary protocol keys."""

    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FrozenResNet50Head(nn.Module):
    """ImageNet-pretrained frozen ResNet-50 plus a continual two-linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        weights_name: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.weights_name = weights_name
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenResNet50Head":
        super().train(mode)
        self.backbone.eval()
        self.head.train(mode)
        return self

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        return self.backbone(images).detach()

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs if inputs.ndim == 2 else self.extract_features(inputs)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected [batch,{self.feature_dim}] features, got {tuple(features.shape)}"
            )
        return self.head(features)


def build_frozen_resnet50_head(
    seed: int,
    *,
    hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> FrozenResNet50Head:
    """Build the frozen ImageNet ResNet-50 and freshly seeded continual head."""

    if hidden_dim < 1:
        raise ValueError("head hidden dimension must be positive")
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("torchvision is required for the frozen backbone") from error
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        model = FrozenResNet50Head(
            backbone,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            weights_name=BACKBONE_WEIGHTS if pretrained else "none-test-only",
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    model._cap_scale_architecture = MODEL_NAME  # type: ignore[attr-defined]
    return model


def model_parameter_counts(model: FrozenResNet50Head) -> dict[str, int]:
    """Return explicit frozen-backbone and trainable-head parameter counts."""

    return {
        "backbone_total_params": sum(p.numel() for p in model.backbone.parameters()),
        "backbone_trainable_params": sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        ),
        "head_total_params": sum(p.numel() for p in model.head.parameters()),
        "head_trainable_params": sum(
            p.numel() for p in model.head.parameters() if p.requires_grad
        ),
    }


def protected_parameter_names(model: nn.Module) -> set[str]:
    """Protect only trainable Linear weights; head biases remain exempt."""

    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        prefix = f"{module_name}." if module_name else ""
        if module.weight is not None:
            names.add(prefix + "weight")
    return names


def cap_parameter_groups(model: nn.Module, weight_decay: float = 0.0) -> list[dict]:
    protected_names = protected_parameter_names(model)
    protected: list[nn.Parameter] = []
    exempt: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (protected if name in protected_names else exempt).append(parameter)
    if not protected or not exempt:
        raise RuntimeError("head CAP parameter grouping is unexpectedly empty")
    return [
        {"params": protected, "protect": True, "weight_decay": weight_decay},
        {"params": exempt, "protect": False, "weight_decay": 0.0},
    ]


def adamw_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Apply decay to head weight matrices and not to head biases."""

    decayed_names = protected_parameter_names(model)
    decayed: list[nn.Parameter] = []
    exempt: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (decayed if name in decayed_names else exempt).append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": exempt, "weight_decay": 0.0},
    ]


class CheckpointableAdamW(torch.optim.AdamW):
    """AdamW with a namespace for method-controller checkpoint state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale_state: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["scale_state"] = self.scale_state
        return result

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(
            {"state": state_dict["state"], "param_groups": state_dict["param_groups"]}
        )
        self.scale_state = dict(state_dict.get("scale_state", {}))


class EWCOracle:
    """Boundary-aware online EWC controller around a normal AdamW optimizer.

    Raw squared gradients are averaged within a segment.  At the known segment
    boundary they are consolidated and the current weights become the new EWC
    reference.  The controller never receives the domain identity.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: CheckpointableAdamW,
        *,
        penalty_lambda: float,
        fisher_decay: float,
    ) -> None:
        if penalty_lambda < 0 or not 0 <= fisher_decay <= 1:
            raise ValueError("invalid EWC lambda or Fisher decay")
        self.model = model
        self.optimizer = optimizer
        optimizer.scale_state["ewc"] = {
            "lambda": float(penalty_lambda),
            "fisher_decay": float(fisher_decay),
            "segment_steps": 0,
            "fisher_segment": {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            },
            "fisher_consolidated": {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            },
            "theta_star": None,
        }

    @property
    def state(self) -> dict[str, Any]:
        return self.optimizer.scale_state["ewc"]

    @torch.no_grad()
    def after_backward(self) -> None:
        state = self.state
        theta_star = state["theta_star"]
        count = int(state["segment_steps"])
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            fisher = state["fisher_segment"][name]
            if fisher.device != parameter.device:
                for key in ("fisher_segment", "fisher_consolidated"):
                    state[key][name] = state[key][name].to(parameter.device)
                if theta_star is not None:
                    theta_star[name] = theta_star[name].to(parameter.device)
                fisher = state["fisher_segment"][name]
            raw_gradient = parameter.grad.detach()
            fisher.mul_(count / (count + 1.0)).addcmul_(
                raw_gradient, raw_gradient, value=1.0 / (count + 1.0)
            )
            if theta_star is not None:
                parameter.grad.add_(
                    state["fisher_consolidated"][name]
                    * (parameter.detach() - theta_star[name]),
                    alpha=float(state["lambda"]),
                )
        state["segment_steps"] = count + 1

    @torch.no_grad()
    def consolidate(self) -> None:
        state = self.state
        decay = float(state["fisher_decay"])
        for name, parameter in self.model.named_parameters():
            state["fisher_consolidated"][name].mul_(decay).add_(
                state["fisher_segment"][name]
            )
            state["fisher_segment"][name].zero_()
        state["theta_star"] = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        state["segment_steps"] = 0


@dataclass
class MethodBundle:
    optimizer: torch.optim.Optimizer
    ewc: EWCOracle | None
    metadata: dict[str, Any]

    def after_backward(self) -> None:
        if self.ewc is not None:
            self.ewc.after_backward()

    def on_boundary(self) -> None:
        if self.ewc is not None:
            self.ewc.consolidate()


def _load_validated_upgd_class() -> type[torch.optim.Optimizer]:
    root = Path(__file__).resolve().parent
    source = root / "external" / "upgd" / "core" / "optim" / "weight_upgd" / "first_order.py"
    if not source.exists():
        raise FileNotFoundError(f"validated vendored UPGD source is missing: {source}")
    spec = importlib.util.spec_from_file_location("scale_vendored_upgd", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import vendored UPGD source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = getattr(module, "FirstOrderGlobalUPGD")

    class DeviceAdaptedUPGD(upstream):
        def step(self, closure=None):
            del closure
            first = next(p for group in self.param_groups for p in group["params"])
            with torch.device(first.device):
                return super().step()

    DeviceAdaptedUPGD.__name__ = "FirstOrderGlobalUPGD"
    return DeviceAdaptedUPGD


def build_method(
    method: str, model: nn.Module, config: Mapping[str, Any]
) -> MethodBundle:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    if method == "adamw":
        optimizer = CheckpointableAdamW(
            adamw_parameter_groups(model, weight_decay), lr=lr, betas=(0.9, 0.999)
        )
        return MethodBundle(optimizer, None, {"variant": "AdamW"})
    if method == "cap":
        required = {"k", "T_a", "T_s", "T_m"}
        missing = required - set(config)
        if missing:
            raise KeyError(f"CAP config lacks {sorted(missing)}")
        optimizer = CAPAdam(
            cap_parameter_groups(model, weight_decay),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            k=float(config["k"]),
            T_a=float(config["T_a"]),
            T_s=float(config["T_s"]),
            T_m=float(config["T_m"]),
            passive=False,
            protection="box",
            retention="release",
            release_update="contest_blocked",
            box_geometry="coordinate",
            momentum_response="none",
            decay_mode="blend",
            track_bind_stats=True,
        )
        return MethodBundle(
            optimizer,
            None,
            {
                "variant": "raw-g2-moving-anchor-coordinate-box-contest-blocked-release",
                "protected_scope": "trainable head Linear.weight only; head biases exempt",
            },
        )
    if method == "upgd":
        upgd_class = _load_validated_upgd_class()
        optimizer = upgd_class(
            list(model.named_parameters()),
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=float(config["beta_utility"]),
            sigma=float(config["sigma"]),
        )
        return MethodBundle(
            optimizer,
            None,
            {
                "variant": "FirstOrderGlobalUPGD",
                "utility": "weight-wise first-order global",
                "source": "external/upgd/core/optim/weight_upgd/first_order.py",
            },
        )
    optimizer = CheckpointableAdamW(
        adamw_parameter_groups(model, weight_decay), lr=lr, betas=(0.9, 0.999)
    )
    ewc = EWCOracle(
        model,
        optimizer,
        penalty_lambda=float(config["lambda"]),
        fisher_decay=float(config.get("fisher_decay", 0.95)),
    )
    return MethodBundle(
        optimizer,
        ewc,
        {"variant": "EWC-oracle", "boundary_access": True},
    )


def build_domain_schedule(seed: int, cycles: int = CYCLES, *, domains: Sequence[str] = DOMAINS) -> list[dict[str, Any]]:
    rng = random.Random(keyed_seed("domain_schedule", seed))
    order: list[int] = []
    for cycle in range(cycles):
        current = list(range(len(domains)))
        rng.shuffle(current)
        if order and current[0] == order[-1]:
            current = current[1:] + current[:1]
        order.extend(current)
    previous: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for segment, domain_id in enumerate(order):
        prior = previous.get(domain_id)
        rows.append(
            {
                "seed": int(seed),
                "segment_index": segment,
                "cycle_index": segment // len(domains),
                "domain_id": domain_id,
                "domain_name": domains[domain_id],
                "severity": DOMAIN_SEVERITY,
                "previous_visit_segment": prior,
                "revisit_gap": None if prior is None else segment - prior - 1,
            }
        )
        previous[domain_id] = segment
    return rows


def validate_schedule(schedule: Sequence[Mapping[str, Any]], cycles: int = CYCLES, *, domains: Sequence[str] = DOMAINS) -> None:
    expected = len(domains) * cycles
    if len(schedule) != expected:
        raise ValueError(f"schedule has {len(schedule)} segments, expected {expected}")
    ids = [int(row["domain_id"]) for row in schedule]
    for cycle in range(cycles):
        block = ids[cycle * len(domains) : (cycle + 1) * len(domains)]
        if sorted(block) != list(range(len(domains))):
            raise ValueError(f"cycle {cycle} is not a domain permutation")
    for boundary in range(len(domains), expected, len(domains)):
        if ids[boundary - 1] == ids[boundary]:
            raise ValueError("immediate duplicate at a cycle boundary")


def schedule_with_hash(seed: int, cycles: int = CYCLES, *, domains: Sequence[str] = DOMAINS,
                       experiment_version: str = EXPERIMENT_VERSION) -> dict[str, Any]:
    rows = build_domain_schedule(seed, cycles, domains=domains)
    validate_schedule(rows, cycles, domains=domains)
    return {
        "experiment_version": experiment_version,
        "seed": seed,
        "domains": list(domains),
        "cycles": cycles,
        "segments": rows,
        "schedule_hash": payload_hash(rows),
    }


def select_segment_records(
    records: Sequence[Mapping[str, Any]], seed: int, segment_index: int
) -> list[dict[str, Any]]:
    """Select 256/class without replacement and deterministically shuffle."""

    by_class: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_class.setdefault(int(record["class_index"]), []).append(record)
    if sorted(by_class) != list(range(NUM_CLASSES)):
        raise ValueError("training manifest must contain class indices 0..99")
    selected: list[dict[str, Any]] = []
    for class_index in range(NUM_CLASSES):
        class_records = sorted(by_class[class_index], key=lambda item: str(item["source_id"]))
        if len(class_records) != TRAIN_PER_CLASS:
            raise ValueError(
                f"class {class_index} has {len(class_records)} train records, expected 500"
            )
        rng = np.random.default_rng(keyed_seed("segment_class", seed, segment_index, class_index))
        indices = rng.choice(
            len(class_records), size=SAMPLES_PER_CLASS_PER_SEGMENT, replace=False
        )
        selected.extend(dict(class_records[int(index)]) for index in indices)
    rng = np.random.default_rng(keyed_seed("segment_shuffle", seed, segment_index))
    permutation = rng.permutation(len(selected))
    return [selected[int(index)] for index in permutation]


def select_evaluation_bank(
    records: Sequence[Mapping[str, Any]], seed: int, per_class: int
) -> list[dict[str, Any]]:
    by_class: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_class.setdefault(int(record["class_index"]), []).append(record)
    result: list[dict[str, Any]] = []
    for class_index in range(NUM_CLASSES):
        class_records = sorted(by_class.get(class_index, []), key=lambda item: str(item["source_id"]))
        if len(class_records) != EVAL_PER_CLASS:
            raise ValueError(
                f"class {class_index} has {len(class_records)} eval records, expected 100"
            )
        rng = np.random.default_rng(keyed_seed("evaluation_bank", seed, class_index))
        indices = rng.permutation(len(class_records))[:per_class]
        result.extend(dict(class_records[int(index)]) for index in indices)
    return result


def records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return payload_hash([str(record["source_id"]) for record in records])


def sample_order_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return records_hash(records)


def corruption_seed_hash(
    seed: int, segment_index: int, domain_id: int, records: Sequence[Mapping[str, Any]]
) -> str:
    # ``seed`` and ``segment_index`` remain in the public signature so old
    # callers do not accidentally perturb stream pairing.  A frozen feature is
    # a property of a source/domain pair, so its corruption cannot vary by
    # visit, method, or position if it is to be cached exactly.
    del seed, segment_index
    values = [
        keyed_seed(
            "frozen_feature_corruption",
            record["source_id"],
            domain_id,
            DOMAIN_SEVERITY,
        )
        for record in records
    ]
    return payload_hash(values)


def scale_metrics(segment_rows: Sequence[Mapping[str, Any]], *, total_segments: int = 45) -> dict[str, float]:
    """Equal-length segment online accuracy and revisit retention in the second half."""

    second_half = [row for row in segment_rows if total_segments // 2 < int(row["segment"]) <= total_segments]
    plasticity = [float(row["mean_online_accuracy"]) for row in second_half]
    revisits = [
        float(row["pre_visit_accuracy"])
        for row in second_half
        if row.get("revisit_gap") not in (None, "")
    ]
    all_revisits = [
        float(row["pre_visit_accuracy"])
        for row in segment_rows
        if row.get("revisit_gap") not in (None, "")
    ]
    if not plasticity or not revisits:
        raise ValueError("a complete-enough stream is required for P_scale/R_scale")
    p_scale = fmean(plasticity)
    r_scale = fmean(revisits)
    return {
        "P_scale": p_scale,
        "R_scale": r_scale,
        "B_scale": min(p_scale, r_scale),
        "R_full_stream": fmean(all_revisits),
    }


def gradient_norm(model: nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def parameter_norm(model: nn.Module) -> float:
    squares = [parameter.detach().float().square().sum() for parameter in model.parameters()]
    return float(torch.stack(squares).sum().sqrt().item())


@torch.no_grad()
def cap_diagnostics(optimizer: CAPAdam) -> dict[str, float]:
    importance: list[Tensor] = []
    radii: list[Tensor] = []
    displacement: list[Tensor] = []
    bind_count = 0
    bind_numel = 0
    for group in optimizer.param_groups:
        if not bool(group.get("protect", True)):
            continue
        bind_count += int(group.get("last_bind_count", 0))
        bind_numel += int(group.get("last_bind_numel", 0))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "imp" not in state or "anchor" not in state:
                continue
            imp = state["imp"].detach().reshape(-1)
            importance.append(imp)
            radii.append(
                (
                    float(group["k"])
                    / (state["imp"].detach().sqrt() + float(group["eps"]))
                ).reshape(-1)
            )
            displacement.append(
                (parameter.detach() - state["anchor"].detach()).abs().reshape(-1)
            )
    if not importance:
        return {
            "bind_rate": 0.0,
            "median_importance": 0.0,
            "median_radius": math.inf,
            "frozen_fraction": 0.0,
            "anchor_displacement": 0.0,
        }
    imp = torch.cat(importance)
    radius = torch.cat(radii)
    anchor_distance = torch.cat(displacement)
    return {
        "bind_rate": bind_count / max(bind_numel, 1),
        "median_importance": float(imp.median().item()),
        "median_radius": float(radius.median().item()),
        "frozen_fraction": float((radius < 1e-6).float().mean().item()),
        "anchor_displacement": float(anchor_distance.median().item()),
    }


@torch.no_grad()
def upgd_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    utilities: list[Tensor] = []
    for state in optimizer.state.values():
        if "avg_utility" in state:
            utilities.append(state["avg_utility"].detach().reshape(-1).float())
    if not utilities:
        return {"median_utility": 0.0, "p90_utility": 0.0}
    values = torch.cat(utilities)
    return {
        "median_utility": float(values.median().item()),
        "p90_utility": float(torch.quantile(values, 0.9).item()),
    }


def default_config(method: str, *, cap_k: float = 7e-5) -> dict[str, Any]:
    return {
        "adamw": {"lr": 3e-4, "weight_decay": 1e-4},
        "cap": {
            "lr": 3e-4,
            "weight_decay": 0.0,
            "k": cap_k,
            "T_a": 2_000.0,
            "T_s": 2_000.0,
            "T_m": 10_000.0,
        },
        "upgd": {
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "beta_utility": 0.9999,
            "sigma": 1e-3,
        },
        "ewc": {
            "lr": 3e-4,
            "weight_decay": 0.0,
            "lambda": 100.0,
            "fisher_decay": 0.95,
        },
    }[method]


def tuning_grid(method: str, *, cap_k: float = 7e-5) -> list[dict[str, Any]]:
    learning_rates = (1e-4, 3e-4, 1e-3)
    if method == "adamw":
        return [
            {"lr": lr, "weight_decay": decay}
            for lr in learning_rates
            for decay in (0.0, 1e-4, 1e-3)
        ]
    if method == "cap":
        return [
            {
                "lr": lr,
                "weight_decay": 0.0,
                "k": cap_k * multiplier,
                "T_a": 2_000.0,
                "T_s": 2_000.0,
                "T_m": 10_000.0,
            }
            for lr in learning_rates
            for multiplier in (0.5, 1.0, 2.0)
        ]
    if method == "upgd":
        return [
            {
                "lr": lr,
                "weight_decay": 1e-3,
                "beta_utility": 0.9999,
                "sigma": sigma,
            }
            for lr in learning_rates
            for sigma in (1e-4, 1e-3, 1e-2, 1e-1)
        ]
    if method == "ewc":
        return [
            {
                "lr": lr,
                "weight_decay": 0.0,
                "lambda": penalty,
                "fisher_decay": 0.95,
            }
            for lr in learning_rates
            for penalty in (1.0, 10.0, 100.0, 1000.0)
        ]
    raise ValueError(f"unknown method {method!r}")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        # Checkpoints are loaded with ``map_location=device`` by the runner.
        # On a CUDA run that also moves these serialized RNG ByteTensors to the
        # GPU, while PyTorch's RNG restoration API requires CPU ByteTensors.
        torch.cuda.set_rng_state_all(
            [rng_state.detach().cpu() for rng_state in state["cuda"]]
        )


__all__ = [
    "BACKBONE_WEIGHTS",
    "BATCH_SIZE",
    "CYCLES",
    "DATA_SPLIT_SEED",
    "DEVELOPMENT_SEEDS",
    "DEFAULT_HEAD_HIDDEN_DIM",
    "DOMAINS",
    "DOMAIN_SEVERITY",
    "EVAL_PER_CLASS",
    "EXPERIMENT_VERSION",
    "FAST_EVAL_PER_CLASS",
    "FINAL_SEEDS",
    "METHODS",
    "MODEL_NAME",
    "NUM_CLASSES",
    "SAMPLES_PER_CLASS_PER_SEGMENT",
    "TRAIN_PER_CLASS",
    "UPDATES_PER_SEGMENT",
    "MethodBundle",
    "FrozenResNet50Head",
    "adamw_parameter_groups",
    "atomic_write_csv",
    "atomic_write_json",
    "build_domain_schedule",
    "build_method",
    "build_frozen_resnet50_head",
    "canonical_json",
    "capture_rng_state",
    "corruption_seed_hash",
    "default_config",
    "file_hash",
    "gradient_norm",
    "keyed_seed",
    "model_parameter_counts",
    "parameter_norm",
    "payload_hash",
    "protected_parameter_names",
    "read_csv",
    "records_hash",
    "restore_rng_state",
    "cap_diagnostics",
    "cap_parameter_groups",
    "sample_order_hash",
    "scale_metrics",
    "schedule_with_hash",
    "select_evaluation_bank",
    "select_segment_records",
    "tuning_grid",
    "upgd_diagnostics",
    "validate_schedule",
]
