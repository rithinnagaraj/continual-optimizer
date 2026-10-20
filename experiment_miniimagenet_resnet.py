"""Run the frozen-ResNet-50-head recurrent-domain mini-ImageNet experiment.

The supported comparison is AdamW, final revised-Release CAP, validated UPGD,
and EWC-oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from miniimagenet_domains import (
    ManifestImageDataset,
    PREPROCESSING_NAME,
    validate_corruptions,
    write_corruption_grid,
)
from miniimagenet_scale_common import (
    BATCH_SIZE,
    BACKBONE_WEIGHTS,
    CYCLES,
    DEVELOPMENT_SEEDS,
    DEFAULT_HEAD_HIDDEN_DIM,
    DOMAINS,
    DOMAIN_SEVERITY,
    EXPERIMENT_VERSION,
    FAST_EVAL_PER_CLASS,
    FINAL_SEEDS,
    METHODS,
    MODEL_NAME,
    NUM_CLASSES,
    SAMPLES_PER_CLASS_PER_SEGMENT,
    TRAIN_PER_CLASS,
    UPDATES_PER_SEGMENT,
    atomic_write_csv,
    atomic_write_json,
    build_frozen_resnet50_head,
    build_method,
    capture_rng_state,
    corruption_seed_hash,
    default_config,
    gradient_norm,
    keyed_seed,
    model_parameter_counts,
    parameter_norm,
    payload_hash,
    read_csv,
    records_hash,
    restore_rng_state,
    cap_diagnostics,
    sample_order_hash,
    scale_metrics,
    schedule_with_hash,
    select_evaluation_bank,
    select_segment_records,
    tuning_grid,
    upgd_diagnostics,
)
from cap_adam_experimental import CAPAdam


DEFAULT_OUTPUT_DIR = Path("results/miniimagenet_frozen_resnet50_head")
ONLINE_FIELDS = (
    "method",
    "seed",
    "global_update",
    "segment",
    "cycle",
    "domain_id",
    "domain_name",
    "within_segment_update",
    "loss",
    "pre_update_accuracy",
    "gradient_norm",
    "parameter_norm",
    "update_seconds",
    "examples_per_second",
    "peak_gpu_memory_bytes",
)
SEGMENT_FIELDS = (
    "domain",
    "mean_all_domain_accuracy",
    "method",
    "seed",
    "segment",
    "cycle",
    "domain_id",
    "domain_name",
    "previous_visit_segment",
    "revisit_gap",
    "pre_visit_accuracy",
    "first_visit_pre_accuracy",
    "is_first_visit",
    "post_visit_accuracy",
    "learning_gain",
    "mean_online_accuracy",
    "non_current_domain_accuracy",
    "all_domain_accuracy",
    "mean_seen_domain_accuracy",
    "worst_seen_domain_accuracy",
    "sample_order_hash",
    "corruption_seed_hash",
)
EVAL_FIELDS = (
    "segment",
    "training_domain",
    "seen_before",
    "method",
    "seed",
    "checkpoint_segment",
    "cycle",
    "timing",
    "evaluated_domain_id",
    "evaluated_domain",
    "domain_seen_before",
    "segments_since_last_visit",
    "accuracy",
    "best_historical_accuracy",
    "forgetting_from_best",
    "bank_size",
)
DIAGNOSTIC_FIELDS = (
    "upgd_mean_gate", "upgd_protected_fraction", "upgd_update_norm", "upgd_perturbation_norm",
    "hidden_median_importance", "hidden_median_radius", "hidden_frozen_fraction", "hidden_anchor_displacement",
    "output_median_importance", "output_median_radius", "output_frozen_fraction", "output_anchor_displacement",
    "method",
    "seed",
    "segment",
    "domain_name",
    "bind_rate",
    "median_importance",
    "median_radius",
    "frozen_fraction",
    "anchor_displacement",
    "median_utility",
    "p90_utility",
    "gradient_norm",
    "parameter_norm",
    "mean_loss",
    "mean_update_seconds",
    "peak_gpu_memory_bytes",
    "optimizer_state_bytes",
    "backbone_trainable_params",
    "head_trainable_params",
    "backbone_mode",
)


class MiniImageNetAdapter:
    domains = DOMAINS
    experiment_version = EXPERIMENT_VERSION
    total_segments = 45
    full_evaluation = False

    def schedule(self, seed):
        return schedule_with_hash(seed)

    def select(self, records, seed, segment_index, domain_id, count):
        selected = select_segment_records(records, seed, segment_index)
        if count > len(selected):
            raise ValueError("requested updates exceed the 25,600-example balanced segment")
        return selected[:count]

    def filter_records(self, records, domain_id):
        return list(records)

    def images(self, records, **kwargs):
        return ManifestImageDataset(records, **kwargs)

    def cache_metadata(self, split, domain_id):
        return {}


class FeatureCacheStore:
    """Optional manifest-wide source/domain frozen-feature cache."""

    def __init__(
        self,
        cache_dir: Path | None,
        *,
        rebuild: bool,
        data_root: Path,
        manifest_hash: str,
        train_records: Sequence[Mapping[str, Any]],
        eval_records: Sequence[Mapping[str, Any]],
        workers: int,
        batch_size: int,
        adapter: Any = None,
    ) -> None:
        self.adapter = adapter or MiniImageNetAdapter()
        self.cache_dir = cache_dir
        self.rebuild = bool(rebuild)
        self.data_root = data_root
        self.manifest_hash = manifest_hash
        self.records = {
            "train": [dict(row) for row in train_records],
            "eval": [dict(row) for row in eval_records],
        }
        self.workers = workers
        self.batch_size = batch_size
        self._payloads: dict[tuple[str, int], dict[str, Any]] = {}
        self._rebuilt: set[tuple[str, int]] = set()

    def _records(self, split, domain_id):
        return self.adapter.filter_records(self.records[split], domain_id)

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def identity(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir.resolve()) if self.cache_dir else None,
            "manifest_hash": self.manifest_hash,
            "preprocessing": PREPROCESSING_NAME,
            "backbone_weights": BACKBONE_WEIGHTS,
            "cache_format": "source-domain-frozen-features-v1",
            "experiment_version": self.adapter.experiment_version,
            "dataset_cache_metadata": {
                f"{domain}_{split}": self.adapter.cache_metadata(split, domain_id)
                for domain_id, domain in enumerate(self.adapter.domains)
                for split in ("train", "eval")
            } if self.adapter.full_evaluation else {},
        }

    def _path(self, split: str, domain_id: int) -> Path:
        assert self.cache_dir is not None
        if self.adapter.full_evaluation:
            return self.cache_dir / f"{self.adapter.domains[domain_id]}_{'test' if split == 'eval' else split}.pt"
        return self.cache_dir / f"{split}_domain_{domain_id:02d}_{DOMAINS[domain_id]}.pt"

    def _expected_metadata(self, split: str, domain_id: int, feature_dim: int) -> dict[str, Any]:
        return {
            "experiment_version": self.adapter.experiment_version,
            "cache_format": "source-domain-frozen-features-v1",
            "manifest_hash": self.manifest_hash,
            "split": "test" if self.adapter.full_evaluation and split == "eval" else split,
            "domain_id": domain_id,
            "domain_name": self.adapter.domains[domain_id],
            "severity": None if self.adapter.full_evaluation else DOMAIN_SEVERITY,
            "backbone_weights": BACKBONE_WEIGHTS,
            "preprocessing": PREPROCESSING_NAME,
            "feature_dim": feature_dim,
            "records_hash": records_hash(self._records(split, domain_id)),
            **self.adapter.cache_metadata(split, domain_id),
        }

    @torch.inference_mode()
    def _build(
        self,
        split: str,
        domain_id: int,
        model: Any,
        device: torch.device,
    ) -> dict[str, Any]:
        build_started = time.perf_counter()
        records = self._records(split, domain_id)
        dataset = self.adapter.images(
            records,
            data_root=self.data_root,
            domain_id=domain_id,
            benchmark_seed=0,
            segment_index=0 if split == "train" else None,
            training=split == "train",
        )
        loader = _loader(
            dataset,
            batch_size=self.batch_size,
            workers=self.workers,
            seed=keyed_seed("feature_cache_loader", split, domain_id),
        )
        was_training = model.training
        model.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for images, targets in loader:
            features.append(model.extract_features(images.to(device, non_blocking=True)).cpu())
            labels.append(targets.to(dtype=torch.long).cpu())
        model.train(was_training)
        payload = {
            "metadata": self._expected_metadata(split, domain_id, model.feature_dim),
            "source_ids": [str(row["source_id"]) for row in records],
            "source_paths": [str(row["source_path"]) for row in records],
            "labels": torch.cat(labels),
            "features": torch.cat(features).to(dtype=torch.float32),
            "cache_construction_seconds": time.perf_counter() - build_started,
        }
        _atomic_torch_save(self._path(split, domain_id), payload)
        return payload

    def _load(self, split: str, domain_id: int, model: Any, device: torch.device) -> dict[str, Any]:
        key = (split, domain_id)
        if key in self._payloads:
            return self._payloads[key]
        path = self._path(split, domain_id)
        should_rebuild = self.rebuild and key not in self._rebuilt
        payload = self._build(split, domain_id, model, device) if should_rebuild or not path.exists() else _torch_load(path)
        self._rebuilt.add(key)
        expected = self._expected_metadata(split, domain_id, model.feature_dim)
        if payload.get("metadata") != expected:
            raise RuntimeError(
                f"incompatible feature cache {path}; use --rebuild-feature-cache"
            )
        source_ids = payload.get("source_ids")
        labels = payload.get("labels")
        features = payload.get("features")
        if not isinstance(source_ids, list) or not isinstance(labels, torch.Tensor) or not isinstance(features, torch.Tensor):
            raise RuntimeError(f"malformed feature cache: {path}")
        if len(source_ids) != len(labels) or len(labels) != len(features):
            raise RuntimeError(f"feature-cache lengths disagree: {path}")
        if labels.dtype != torch.long or features.dtype != torch.float32:
            raise RuntimeError(f"feature-cache tensor dtypes are invalid: {path}")
        expected_source_ids = [str(row["source_id"]) for row in self._records(split, domain_id)]
        expected_labels = torch.tensor(
            [int(row["class_index"]) for row in self._records(split, domain_id)], dtype=torch.long
        )
        if source_ids != expected_source_ids or not torch.equal(labels.cpu(), expected_labels):
            raise RuntimeError(f"feature-cache source IDs/labels disagree: {path}")
        if features.ndim != 2 or features.shape[1] != model.feature_dim:
            raise RuntimeError(f"feature-cache tensor shape is invalid: {path}")
        if not torch.isfinite(features).all():
            raise RuntimeError(f"feature cache contains non-finite values: {path}")
        self._payloads[key] = payload
        return payload

    def dataset(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        domain_id: int,
        model: Any,
        device: torch.device,
    ) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
        records = self.adapter.filter_records(records, domain_id)
        if not self.enabled:
            return self.adapter.images(
                records,
                data_root=self.data_root,
                domain_id=domain_id,
                benchmark_seed=0,
                segment_index=0 if split == "train" else None,
                training=split == "train",
            )
        payload = self._load(split, domain_id, model, device)
        positions = {source_id: index for index, source_id in enumerate(payload["source_ids"])}
        try:
            indices = torch.tensor(
                [positions[str(row["source_id"])] for row in records], dtype=torch.long
            )
        except KeyError as error:
            raise RuntimeError(f"source ID is absent from {split} feature cache") from error
        labels = payload["labels"].index_select(0, indices)
        expected_labels = torch.tensor(
            [int(row["class_index"]) for row in records], dtype=torch.long
        )
        if not torch.equal(labels, expected_labels):
            raise RuntimeError("feature-cache labels/order disagree with the manifest")
        return TensorDataset(payload["features"].index_select(0, indices), labels)


def _typed_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        row: dict[str, Any] = dict(raw)
        row["class_index"] = int(row["class_index"])
        row["within_class_index"] = int(row["within_class_index"])
        rows.append(row)
    return rows


def load_manifest_bundle(
    manifest_dir: Path, data_root: Path | None
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata_path = manifest_dir / "miniimagenet_manifest.json"
    train_path = manifest_dir / "miniimagenet_train_manifest.csv"
    eval_path = manifest_dir / "miniimagenet_eval_manifest.csv"
    if not metadata_path.exists() or not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"prepared manifests are missing under {manifest_dir}; run "
            "prepare_miniimagenet_scale.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = data_root or Path(metadata["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"mini-ImageNet data root is unavailable: {root}")
    train_rows = _typed_manifest_rows(train_path)
    eval_rows = _typed_manifest_rows(eval_path)
    expected_hash = str(metadata["manifest_hash"])
    all_rows = train_rows + eval_rows
    if any(row.get("manifest_hash") != expected_hash for row in all_rows):
        raise ValueError("manifest CSV hash does not match manifest metadata")
    canonical_rows = []
    for row in sorted(
        all_rows, key=lambda item: (int(item["class_index"]), int(item["within_class_index"]))
    ):
        canonical_rows.append({key: value for key, value in row.items() if key != "manifest_hash"})
    # CSV parsing produces strings, whereas preparation hashes numeric indices.
    for row in canonical_rows:
        row["class_index"] = int(row["class_index"])
        row["within_class_index"] = int(row["within_class_index"])
    if payload_hash(canonical_rows) != expected_hash:
        raise ValueError("manifest contents do not match the recorded manifest hash")
    if len(train_rows) != NUM_CLASSES * TRAIN_PER_CLASS or len(eval_rows) != 10_000:
        raise ValueError("prepared manifest does not have the required 50k/10k split")
    if {row["source_id"] for row in train_rows} & {row["source_id"] for row in eval_rows}:
        raise ValueError("prepared manifest contains train/evaluation leakage")
    for class_index in range(NUM_CLASSES):
        train_count = sum(int(row["class_index"]) == class_index for row in train_rows)
        eval_count = sum(int(row["class_index"]) == class_index for row in eval_rows)
        if (train_count, eval_count) != (500, 100):
            raise ValueError(
                f"class {class_index} has {train_count}/{eval_count} train/eval rows"
            )
    return root, train_rows, eval_rows, metadata


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    if hasattr(optimizer, "scale_state"):
        stack = [getattr(optimizer, "scale_state")]
        seen: set[int] = set()
        while stack:
            value = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
            elif isinstance(value, Mapping):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
    return total


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    domain_id: int,
    seed: int,
    device: torch.device,
    workers: int,
    batch_size: int,
    feature_cache: FeatureCacheStore,
) -> float:
    dataset = feature_cache.dataset(
        records, split="eval", domain_id=domain_id, model=model, device=device
    )
    loader = _loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        seed=keyed_seed("eval_loader", seed, domain_id, records_hash(records)),
    )
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    model.train(was_training)
    return correct / max(total, 1)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location=map_location)


def _write_run_rows(run_dir: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    atomic_write_csv(run_dir / "online_steps.csv", rows["online"], ONLINE_FIELDS)
    atomic_write_csv(run_dir / "stream_segments.csv", rows["segments"], SEGMENT_FIELDS)
    atomic_write_csv(run_dir / "domain_eval_fast.csv", rows["fast"], EVAL_FIELDS)
    atomic_write_csv(run_dir / "domain_eval_full.csv", rows["full"], EVAL_FIELDS)
    atomic_write_csv(run_dir / "domain_eval.csv", rows["fast"], EVAL_FIELDS)
    atomic_write_csv(
        run_dir / "optimizer_diagnostics.csv", rows["diagnostics"], DIAGNOSTIC_FIELDS
    )


def _run_identity(
    *,
    method: str,
    seed: int,
    config: Mapping[str, Any],
    manifest_hash: str,
    schedule_hash: str,
    fast_bank_hash: str,
    full_bank_hash: str,
    batch_size: int,
    updates_per_segment: int,
    head_hidden_dim: int,
    feature_cache_identity: Mapping[str, Any],
    adapter: Any = None,
    max_segments: int = 45,
) -> dict[str, Any]:
    adapter = adapter or MiniImageNetAdapter()
    identity = {
        "experiment_version": adapter.experiment_version,
        "method": method,
        "seed": seed,
        "model": MODEL_NAME,
        "backbone_weights": BACKBONE_WEIGHTS,
        "preprocessing": PREPROCESSING_NAME,
        "head_hidden_dim": head_hidden_dim,
        "feature_cache": dict(feature_cache_identity),
        "manifest_hash": manifest_hash,
        "domain_set_hash": payload_hash({"domains": adapter.domains, "severity": None if adapter.full_evaluation else DOMAIN_SEVERITY}),
        "schedule_hash": schedule_hash,
        "fast_evaluation_bank_hash": fast_bank_hash,
        "full_evaluation_bank_hash": full_bank_hash,
        "batch_size": batch_size,
        "updates_per_segment": updates_per_segment,
        "max_segments": max_segments,
        "optimizer_config_hash": payload_hash(config),
    }
    identity["run_identity_hash"] = payload_hash(identity)
    return identity


def _write_schedule(
    output_dir: Path,
    seed: int,
    train_records: Sequence[Mapping[str, Any]],
    *, adapter: Any = None, batch_size: int = BATCH_SIZE,
    updates_per_segment: int = UPDATES_PER_SEGMENT,
) -> dict[str, Any]:
    adapter = adapter or MiniImageNetAdapter()
    name = f"seed_{seed}.json" if adapter.full_evaluation else f"domain_schedule_seed_{seed}.json"
    if adapter.full_evaluation and (batch_size, updates_per_segment) != (BATCH_SIZE, UPDATES_PER_SEGMENT):
        name = f"seed_{seed}_smoke_b{batch_size}_u{updates_per_segment}.json"
    path = output_dir / "schedules" / name
    train_manifest_hash = records_hash(train_records)
    payload = adapter.schedule(seed)
    payload["train_manifest_records_hash"] = train_manifest_hash
    for row in payload["segments"]:
        selected = adapter.select(train_records, seed, int(row["segment_index"]), int(row["domain_id"]), batch_size * updates_per_segment)
        row["sample_order_hash"] = sample_order_hash(selected)
        row["corruption_seed_hash"] = "" if adapter.full_evaluation else corruption_seed_hash(
            seed, int(row["segment_index"]), int(row["domain_id"]), selected
        )
        row["segment"] = int(row["segment_index"]) + 1
        row["cycle"] = int(row["cycle_index"]) + 1
        row["domain"] = row["domain_name"]
        if adapter.full_evaluation:
            row.pop("severity", None)
            row["previous_visit_segment_index"] = row["previous_visit_segment"]
            if row["previous_visit_segment"] is not None:
                row["previous_visit_segment"] += 1
    payload["batch_size"] = batch_size
    payload["updates_per_segment"] = updates_per_segment
    payload["gap_reporting"] = "integer intervening-segment gap; fixed before training"
    payload["gap_counts"] = {str(gap): sum(row["revisit_gap"] == gap for row in payload["segments"])
                             for gap in sorted({row["revisit_gap"] for row in payload["segments"] if row["revisit_gap"] is not None})}
    payload["schedule_hash"] = payload_hash(payload["segments"])
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing domain schedule is incompatible: {path}")
        return existing
    atomic_write_json(path, payload)
    return payload


def _save_checkpoint(
    run_dir: Path,
    *,
    identity: Mapping[str, Any],
    model: nn.Module,
    bundle: Any,
    next_segment: int,
    next_update: int,
    segment_context: Mapping[str, Any] | None,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    complete: bool,
    schedule: Mapping[str, Any],
) -> None:
    payload = {
        "identity": dict(identity),
        "model": model.state_dict(),
        "optimizer": bundle.optimizer.state_dict(),
        "next_segment": next_segment,
        "next_update": next_update,
        "segment_context": dict(segment_context) if segment_context else None,
        "rows": {key: list(value) for key, value in rows.items()},
        "rng_states": capture_rng_state(),
        "complete": complete,
        "domain_schedule": dict(schedule),
    }
    _write_run_rows(run_dir, rows)
    _atomic_torch_save(run_dir / "checkpoint_latest.pt", payload)
    if complete:
        _atomic_torch_save(run_dir / "checkpoint_final.pt", payload)


def _append_eval_rows(
    target: list[dict[str, Any]],
    *,
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    data_root: Path,
    method: str,
    seed: int,
    segment_index: int,
    timing: str,
    domain_ids: Iterable[int],
    schedule_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    workers: int,
    batch_size: int,
    best_by_domain: dict[int, float],
    feature_cache: FeatureCacheStore,
) -> dict[int, float]:
    domains = feature_cache.adapter.domains
    accuracies: dict[int, float] = {}
    for domain_id in domain_ids:
        accuracy = evaluate(
            model,
            records,
            data_root=data_root,
            domain_id=domain_id,
            seed=seed,
            device=device,
            workers=workers,
            batch_size=batch_size,
            feature_cache=feature_cache,
        )
        prior_visits = [
            int(row["segment_index"])
            for row in schedule_rows[: segment_index + 1]
            if int(row["domain_id"]) == domain_id
        ]
        previous_best = best_by_domain.get(domain_id, accuracy)
        best = max(previous_best, accuracy)
        target.append(
            {
                "method": method,
                "seed": seed,
                "checkpoint_segment": segment_index + 1,
                "cycle": segment_index // len(domains) + 1,
                "segment": segment_index + 1,
                "training_domain": domains[int(schedule_rows[segment_index]["domain_id"])],
                "seen_before": bool(prior_visits),
                "timing": timing,
                "evaluated_domain_id": domain_id,
                "evaluated_domain": domains[domain_id],
                "domain_seen_before": bool(prior_visits),
                "segments_since_last_visit": "" if not prior_visits else segment_index - max(prior_visits),
                "accuracy": accuracy,
                "best_historical_accuracy": previous_best,
                "forgetting_from_best": previous_best - accuracy,
                "bank_size": len(feature_cache.adapter.filter_records(records, domain_id)),
            }
        )
        if prior_visits:
            best_by_domain[domain_id] = best
        accuracies[domain_id] = accuracy
    return accuracies


def run_single(
    *,
    method: str,
    seed: int,
    config: Mapping[str, Any],
    run_dir: Path,
    output_dir: Path,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any],
    device: torch.device,
    workers: int,
    batch_size: int,
    updates_per_segment: int,
    max_segments: int,
    resume: bool,
    checkpoint_every_updates: int,
    feature_cache: FeatureCacheStore,
    evaluation_per_class: int = FAST_EVAL_PER_CLASS,
    head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
    method_factory: Any = None,
    study_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = feature_cache.adapter
    domains = adapter.domains
    total_segments = adapter.total_segments
    if not 1 <= max_segments <= total_segments or min(batch_size, updates_per_segment, checkpoint_every_updates) < 1:
        raise ValueError("invalid segment, batch, update, or checkpoint count")
    schedule_payload = _write_schedule(output_dir, seed, train_records, adapter=adapter,
                                      batch_size=batch_size, updates_per_segment=updates_per_segment)
    schedule_rows = schedule_payload["segments"][:max_segments]
    fast_records = list(eval_records) if adapter.full_evaluation else select_evaluation_bank(eval_records, seed, evaluation_per_class)
    full_records = fast_records if adapter.full_evaluation else (
        select_evaluation_bank(eval_records, seed, 100)
        if max_segments == total_segments and updates_per_segment == UPDATES_PER_SEGMENT
        else fast_records
    )
    identity = _run_identity(
        method=method,
        seed=seed,
        config=config,
        manifest_hash=str(manifest_metadata["manifest_hash"]),
        schedule_hash=str(schedule_payload["schedule_hash"]),
        fast_bank_hash=records_hash(fast_records),
        full_bank_hash=records_hash(full_records),
        batch_size=batch_size,
        updates_per_segment=updates_per_segment,
        head_hidden_dim=head_hidden_dim,
        feature_cache_identity=feature_cache.identity(),
        adapter=adapter, max_segments=max_segments,
    )
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_frozen_resnet50_head(seed, hidden_dim=head_hidden_dim).to(device)
    identity["feature_dim"] = model.feature_dim
    identity["head_initialization_hash"] = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.head.parameters()
    )).hexdigest()
    if study_protocol is not None:
        identity["experiment_version"] = study_protocol["experiment_version"]
        identity["study_protocol"] = dict(study_protocol)
    identity["run_identity_hash"] = payload_hash({k: v for k, v in identity.items() if k != "run_identity_hash"})
    method_label = "EWC-oracle" if adapter.full_evaluation and method == "ewc" else method
    if study_protocol is not None and method == "upgd":
        method_label = "UPGD (corrected)"
    counts = model_parameter_counts(model)
    if counts["backbone_trainable_params"] != 0:
        raise RuntimeError("the frozen ResNet-50 backbone has trainable parameters")
    if any(not parameter.requires_grad for parameter in model.head.parameters()):
        raise RuntimeError("every continual-head parameter must be trainable")
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    head_ids = {id(parameter) for parameter in model.head.parameters()}
    if trainable_ids != head_ids:
        raise RuntimeError("all and only head parameters must be trainable")
    bundle = (method_factory or build_method)(method, model.head, config)
    optimizer_ids = {
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    }
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    if optimizer_ids != head_ids or optimizer_ids & backbone_ids:
        raise RuntimeError("optimizer parameter groups must contain exactly the head")
    run_config = {
        **identity,
        "config": dict(config),
        "architecture": {
            "backbone": "torchvision.models.resnet50",
            "backbone_weights": BACKBONE_WEIGHTS,
            "backbone_classifier": "Identity",
            "feature_dim": model.feature_dim,
            "head": f"Linear({model.feature_dim},{model.hidden_dim}) -> ReLU -> Linear({model.hidden_dim},{model.num_classes})",
            **counts,
            "backbone_mode": "eval",
        },
        "cap_timescales_in_segments": (
            {
                key: float(config[key]) / updates_per_segment
                for key in ("T_a", "T_s", "T_m")
            }
            if method == "cap"
            else None
        ),
        "domains": list(domains),
        "severity": None if adapter.full_evaluation else DOMAIN_SEVERITY,
        "invalidated_resnet18_cap_config": {
            "T_a": 80,
            "T_s": 80,
            "T_m": 40000,
            "k_center": 2e-6,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise RuntimeError(f"run configuration changed for {run_dir}")
    else:
        atomic_write_json(config_path, run_config)

    rows: dict[str, list[dict[str, Any]]] = {
        "online": [],
        "segments": [],
        "fast": [],
        "full": [],
        "diagnostics": [],
    }
    next_segment = 0
    next_update = 0
    segment_context: dict[str, Any] | None = None
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    if resume and checkpoint_path.exists():
        checkpoint = _torch_load(checkpoint_path, map_location=device)
        if checkpoint.get("identity") != identity:
            raise RuntimeError(f"incompatible checkpoint identity: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        rows = {key: list(value) for key, value in checkpoint["rows"].items()}
        next_segment = int(checkpoint["next_segment"])
        next_update = int(checkpoint["next_update"])
        segment_context = checkpoint.get("segment_context")
        restore_rng_state(checkpoint["rng_states"])
        if bool(checkpoint.get("complete")):
            return {
                "rows": rows,
                "metrics": scale_metrics(rows["segments"], total_segments=total_segments) if max_segments == total_segments else None,
                "run_dir": run_dir,
            }
    elif checkpoint_path.exists():
        raise FileExistsError(f"run already contains a checkpoint; pass --resume: {run_dir}")

    best_fast: dict[int, float] = {}
    for row in rows["fast"]:
        if str(row.get("timing")) == "post" and row.get("domain_seen_before"):
            domain_id = int(row["evaluated_domain_id"])
            best_fast[domain_id] = max(best_fast.get(domain_id, 0.0), float(row["accuracy"]))

    for segment_index in range(next_segment, len(schedule_rows)):
        schedule = schedule_rows[segment_index]
        domain_id = int(schedule["domain_id"])
        selected = adapter.select(train_records, seed, segment_index, domain_id, batch_size * updates_per_segment)
        actual_order_hash = sample_order_hash(selected)
        if actual_order_hash != schedule["sample_order_hash"]:
            raise RuntimeError("training sample order disagrees with the saved paired schedule")
        if segment_index != next_segment or next_update == 0:
            pre_accuracy = evaluate(
                model,
                fast_records,
                data_root=data_root,
                domain_id=domain_id,
                seed=seed,
                device=device,
                workers=workers,
                batch_size=batch_size,
                feature_cache=feature_cache,
            )
            segment_context = {
                "pre_visit_accuracy": pre_accuracy,
                "online_accuracy_sum": 0.0,
                "loss_sum": 0.0,
                "update_seconds_sum": 0.0,
                "last_gradient_norm": 0.0,
            }
        elif segment_context is None:
            raise RuntimeError("checkpoint is missing within-segment context")

        dataset = feature_cache.dataset(
            selected,
            split="train",
            domain_id=domain_id,
            model=model,
            device=device,
        )
        loader = _loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            seed=keyed_seed("train_loader", seed, segment_index),
        )
        model.train()
        for update_index, (inputs, targets) in enumerate(loader):
            if update_index < (next_update if segment_index == next_segment else 0):
                continue
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bundle.optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if study_protocol is not None and not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss in retune candidate")
            online_accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
            loss.backward()
            if any(parameter.grad is not None for parameter in model.backbone.parameters()):
                raise RuntimeError("frozen backbone unexpectedly received gradients")
            if model.backbone.training:
                raise RuntimeError("frozen backbone left eval mode during training")
            grad = gradient_norm(model.head)
            if study_protocol is not None and not np.isfinite(grad):
                raise FloatingPointError("nonfinite gradient in retune candidate")
            bundle.after_backward()
            bundle.optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            head_norm = parameter_norm(model.head)
            if study_protocol is not None and not np.isfinite(head_norm):
                raise FloatingPointError("nonfinite parameters in retune candidate")
            global_update = segment_index * updates_per_segment + update_index + 1
            context = segment_context
            assert context is not None
            context["online_accuracy_sum"] += online_accuracy
            context["loss_sum"] += float(loss.item())
            context["update_seconds_sum"] += elapsed
            context["last_gradient_norm"] = grad
            rows["online"].append(
                {
                    "method": method_label,
                    "seed": seed,
                    "global_update": global_update,
                    "segment": segment_index + 1,
                    "cycle": segment_index // len(domains) + 1,
                    "domain_id": domain_id,
                    "domain_name": domains[domain_id],
                    "within_segment_update": update_index + 1,
                    "loss": float(loss.item()),
                    "pre_update_accuracy": online_accuracy,
                    "gradient_norm": grad,
                    "parameter_norm": head_norm,
                    "update_seconds": elapsed,
                    "examples_per_second": int(targets.numel()) / max(elapsed, 1e-12),
                    "peak_gpu_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                }
            )
            completed_update = update_index + 1
            if (
                completed_update < updates_per_segment
                and completed_update % checkpoint_every_updates == 0
            ):
                _save_checkpoint(
                    run_dir,
                    identity=identity,
                    model=model,
                    bundle=bundle,
                    next_segment=segment_index,
                    next_update=completed_update,
                    segment_context=context,
                    rows=rows,
                    complete=False,
                    schedule=schedule_payload,
                )

        bundle.on_boundary()
        context = segment_context
        assert context is not None
        seen_domains = sorted(
            {int(row["domain_id"]) for row in schedule_rows[: segment_index + 1]}
        )
        post_accuracies = _append_eval_rows(
            rows["fast"],
            model=model,
            records=fast_records,
            data_root=data_root,
            method=method_label,
            seed=seed,
            segment_index=segment_index,
            timing="post",
            domain_ids=range(len(domains)) if adapter.full_evaluation else seen_domains,
            schedule_rows=schedule_rows,
            device=device,
            workers=workers,
            batch_size=batch_size,
            best_by_domain=best_fast,
            feature_cache=feature_cache,
        )
        post_accuracy = post_accuracies[domain_id]
        non_current = [value for key, value in post_accuracies.items() if key != domain_id]
        rows["segments"].append(
            {
                "method": method_label,
                "seed": seed,
                "segment": segment_index + 1,
                "cycle": segment_index // len(domains) + 1,
                "domain_id": domain_id,
                "domain_name": domains[domain_id],
                "previous_visit_segment": (
                    ""
                    if schedule["previous_visit_segment"] is None
                    else int(schedule["previous_visit_segment"]) + (0 if adapter.full_evaluation else 1)
                ),
                "revisit_gap": "" if schedule["revisit_gap"] is None else schedule["revisit_gap"],
                "pre_visit_accuracy": context["pre_visit_accuracy"],
                "first_visit_pre_accuracy": (context["pre_visit_accuracy"] if schedule["previous_visit_segment"] is None else "") if adapter.full_evaluation else schedule["previous_visit_segment"] is None,
                "is_first_visit": schedule["previous_visit_segment"] is None,
                "post_visit_accuracy": post_accuracy,
                "learning_gain": post_accuracy - float(context["pre_visit_accuracy"]),
                "mean_online_accuracy": context["online_accuracy_sum"] / updates_per_segment,
                "non_current_domain_accuracy": fmean(non_current) if non_current else "",
                "all_domain_accuracy": fmean(post_accuracies.values()),
                "mean_all_domain_accuracy": fmean(post_accuracies.values()),
                "domain": domains[domain_id],
                "mean_seen_domain_accuracy": fmean(post_accuracies[d] for d in seen_domains),
                "worst_seen_domain_accuracy": min(post_accuracies[d] for d in seen_domains),
                "sample_order_hash": actual_order_hash,
                "corruption_seed_hash": schedule["corruption_seed_hash"],
            }
        )
        diagnostic = {
            "method": method_label,
            "seed": seed,
            "segment": segment_index + 1,
            "domain_name": domains[domain_id],
            "gradient_norm": context["last_gradient_norm"],
            "parameter_norm": parameter_norm(model.head),
            "mean_loss": context["loss_sum"] / updates_per_segment,
            "mean_update_seconds": context["update_seconds_sum"] / updates_per_segment,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "optimizer_state_bytes": _optimizer_state_bytes(bundle.optimizer),
            "backbone_trainable_params": counts["backbone_trainable_params"],
            "head_trainable_params": counts["head_trainable_params"],
            "backbone_mode": "eval" if not model.backbone.training else "train",
        }
        if isinstance(bundle.optimizer, CAPAdam):
            diagnostic.update(cap_diagnostics(bundle.optimizer))
            for layer_name, layer in (("hidden", model.head[0]), ("output", model.head[2])):
                state = bundle.optimizer.state.get(layer.weight, {})
                if "imp" in state and "anchor" in state:
                    group = next(g for g in bundle.optimizer.param_groups if any(p is layer.weight for p in g["params"]))
                    radius = float(group["k"]) / (state["imp"].sqrt() + float(group["eps"]))
                    diagnostic.update({
                        f"{layer_name}_median_importance": float(state["imp"].median()),
                        f"{layer_name}_median_radius": float(radius.median()),
                        f"{layer_name}_frozen_fraction": float((radius < 1e-6).float().mean()),
                        f"{layer_name}_anchor_displacement": float((layer.weight.detach() - state["anchor"]).abs().median()),
                    })
        if method == "upgd":
            diagnostic.update(upgd_diagnostics(bundle.optimizer))
            if hasattr(bundle.optimizer, "diagnostics"):
                diagnostic.update(bundle.optimizer.diagnostics())
        rows["diagnostics"].append(diagnostic)

        cycle_complete = (segment_index + 1) % len(domains) == 0
        final_segment = segment_index + 1 == len(schedule_rows)
        if adapter.full_evaluation:
            rows["full"].extend(dict(row) for row in rows["fast"][-len(domains):])
        elif cycle_complete or final_segment:
            full_best: dict[int, float] = {}
            for old in rows["full"]:
                old_domain = int(old["evaluated_domain_id"])
                full_best[old_domain] = max(
                    full_best.get(old_domain, 0.0), float(old["accuracy"])
                )
            _append_eval_rows(
                rows["full"],
                model=model,
                records=full_records,
                data_root=data_root,
                method=method,
                seed=seed,
                segment_index=segment_index,
                timing="cycle_or_final",
                domain_ids=range(len(domains)),
                schedule_rows=schedule_rows,
                device=device,
                workers=workers,
                batch_size=batch_size,
                best_by_domain=full_best,
                feature_cache=feature_cache,
            )
        next_update = 0
        segment_context = None
        complete = segment_index + 1 == len(schedule_rows)
        _save_checkpoint(
            run_dir,
            identity=identity,
            model=model,
            bundle=bundle,
            next_segment=segment_index + 1,
            next_update=0,
            segment_context=None,
            rows=rows,
            complete=complete,
            schedule=schedule_payload,
        )

    metrics = scale_metrics(rows["segments"], total_segments=total_segments) if len(schedule_rows) == total_segments else None
    if metrics is not None:
        atomic_write_json(run_dir / "scale_metrics.json", metrics)
    return {"rows": rows, "metrics": metrics, "run_dir": run_dir}


def _config_id(config: Mapping[str, Any]) -> str:
    return payload_hash(config)[:12]


def _config_distance(config, method, cap_k):
    """Deterministic tie preference for the transferred central configuration."""
    center = default_config(method, cap_k=cap_k)
    return sum(abs(float(value) - float(center[key])) / max(abs(float(center[key])), 1e-12)
               for key, value in config.items())


def run_tuning(
    args: argparse.Namespace,
    *,
    method: str,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any],
    device: torch.device,
    feature_cache: FeatureCacheStore,
) -> None:
    if method == "cap" and args.cap_k_center is None:
        raise ValueError(
            "CAP tuning requires --cap-k-center chosen from the smoke pilot's "
            "CAP diagnostics; the MLP value is not silently transplanted"
        )
    grid = tuning_grid(method, cap_k=args.cap_k_center or 7e-5)
    stage1: list[dict[str, Any]] = []
    for config in grid:
        result = run_single(
            method=method,
            seed=DEVELOPMENT_SEEDS[0],
            config=config,
            run_dir=args.output_dir / "tune" / method / _config_id(config) / f"seed_{DEVELOPMENT_SEEDS[0]}",
            output_dir=args.output_dir,
            data_root=data_root,
            train_records=train_records,
            eval_records=eval_records,
            manifest_metadata=manifest_metadata,
            device=device,
            workers=args.num_workers,
            batch_size=BATCH_SIZE,
            updates_per_segment=UPDATES_PER_SEGMENT,
            max_segments=feature_cache.adapter.total_segments,
            resume=args.resume,
            checkpoint_every_updates=args.checkpoint_every_updates,
            evaluation_per_class=FAST_EVAL_PER_CLASS,
            head_hidden_dim=args.head_hidden_dim,
            feature_cache=feature_cache,
        )
        stage1.append({"config": config, "metrics": result["metrics"]})
    stage1.sort(
        key=lambda row: (
            float(row["metrics"]["B_scale"]),
            (float(row["metrics"]["P_scale"]) + float(row["metrics"]["R_scale"])) / 2,
            -_config_distance(row["config"], method, args.cap_k_center or 7e-5),
            -grid.index(row["config"]),
        ),
        reverse=True,
    )
    top_three = stage1[:3]
    combined: list[dict[str, Any]] = []
    for candidate in top_three:
        config = candidate["config"]
        second = run_single(
            method=method,
            seed=DEVELOPMENT_SEEDS[1],
            config=config,
            run_dir=args.output_dir / "tune" / method / _config_id(config) / f"seed_{DEVELOPMENT_SEEDS[1]}",
            output_dir=args.output_dir,
            data_root=data_root,
            train_records=train_records,
            eval_records=eval_records,
            manifest_metadata=manifest_metadata,
            device=device,
            workers=args.num_workers,
            batch_size=BATCH_SIZE,
            updates_per_segment=UPDATES_PER_SEGMENT,
            max_segments=feature_cache.adapter.total_segments,
            resume=args.resume,
            checkpoint_every_updates=args.checkpoint_every_updates,
            evaluation_per_class=FAST_EVAL_PER_CLASS,
            head_hidden_dim=args.head_hidden_dim,
            feature_cache=feature_cache,
        )
        metrics_by_seed = [candidate["metrics"], second["metrics"]]
        means = {
            key: fmean(float(metrics[key]) for metrics in metrics_by_seed)
            for key in ("P_scale", "R_scale", "B_scale")
        }
        combined.append(
            {"config": config, "metrics_by_seed": metrics_by_seed, "mean_metrics": means}
        )
    combined.sort(
        key=lambda row: (
            float(row["mean_metrics"]["B_scale"]),
            min(float(row["mean_metrics"]["P_scale"]), float(row["mean_metrics"]["R_scale"])),
            (
                float(row["mean_metrics"]["P_scale"])
                + float(row["mean_metrics"]["R_scale"])
            )
            / 2,
            -pstdev(float(metrics["B_scale"]) for metrics in row["metrics_by_seed"]),
            -_config_distance(row["config"], method, args.cap_k_center or 7e-5),
            -grid.index(row["config"]),
        ),
        reverse=True,
    )
    selected = {
        "experiment_version": feature_cache.adapter.experiment_version,
        "model": MODEL_NAME,
        "backbone_weights": BACKBONE_WEIGHTS,
        "head_hidden_dim": args.head_hidden_dim,
        "method": method,
        "manifest_hash": manifest_metadata["manifest_hash"],
        "selection_metric": "mean development B_scale",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "config": combined[0]["config"],
        "mean_metrics": combined[0]["mean_metrics"],
        "stage1": stage1,
        "stage2": combined,
        "cap_k_calibration": (
            {
                "k_center": args.cap_k_center,
                "selection_basis": "user-supplied or provisional scale calibration center; not accuracy",
            }
            if method == "cap"
            else None
        ),
    }
    selected["selected_config_hash"] = payload_hash(selected["config"])
    atomic_write_json(args.output_dir / "selected_configs" / f"{method}.json", selected)


def run_smoke(
    args: argparse.Namespace,
    methods: Sequence[str],
    seeds: Sequence[int],
    *,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any],
    device: torch.device,
    feature_cache: FeatureCacheStore,
) -> None:
    from PIL import Image

    first_path_value = Path(str(eval_records[0]["source_path"]))
    first_path = first_path_value if first_path_value.is_absolute() else data_root / first_path_value
    with Image.open(first_path) as source:
        corruption_report = validate_corruptions(source)
    smoke_dir = args.output_dir / "smoke"
    atomic_write_json(smoke_dir / "corruption_sanity.json", corruption_report)
    write_corruption_grid(eval_records, data_root, smoke_dir / "corruption_grid.png")
    _verify_frozen_model_and_cache(
        args,
        data_root=data_root,
        train_records=train_records,
        eval_records=eval_records,
        manifest_hash=str(manifest_metadata["manifest_hash"]),
        device=device,
        smoke_dir=smoke_dir,
    )
    _run_clean_sanity(
        args,
        seed=seeds[0],
        data_root=data_root,
        train_records=train_records,
        eval_records=eval_records,
        device=device,
        feature_cache=feature_cache,
        output_path=smoke_dir / "clean_trained_domain_sanity.csv",
    )
    fingerprints: dict[int, dict[str, list[str]]] = {}
    for seed in seeds:
        fingerprints[seed] = {}
        for method in methods:
            run_kwargs = dict(
                method=method,
                seed=seed,
                config=default_config(method, cap_k=args.cap_k_center or 7e-5),
                run_dir=smoke_dir / method / f"seed_{seed}",
                output_dir=args.output_dir,
                data_root=data_root,
                train_records=train_records,
                eval_records=eval_records,
                manifest_metadata=manifest_metadata,
                device=device,
                workers=args.num_workers,
                batch_size=args.smoke_batch_size,
                updates_per_segment=args.smoke_updates_per_segment,
                max_segments=args.smoke_segments,
                resume=args.resume,
                checkpoint_every_updates=max(1, args.smoke_updates_per_segment // 2),
                evaluation_per_class=args.smoke_eval_per_class,
                head_hidden_dim=args.head_hidden_dim,
                feature_cache=feature_cache,
            )
            result = run_single(**run_kwargs)
            resumed_kwargs = {**run_kwargs, "resume": True}
            resumed = run_single(**resumed_kwargs)
            if resumed["rows"] != result["rows"]:
                raise RuntimeError(f"checkpoint/resume changed completed {method} smoke rows")
            fingerprints[seed][method] = [
                f"{row['sample_order_hash']}:{row['corruption_seed_hash']}"
                for row in result["rows"]["segments"]
            ]
        values = list(fingerprints[seed].values())
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("method isolation failed: sample order differs across methods")
    atomic_write_json(smoke_dir / "paired_stream_fingerprints.json", fingerprints)
    if "cap" in methods:
        _run_cap_calibration_pilot(
            args,
            seed=seeds[0],
            data_root=data_root,
            train_records=train_records,
            device=device,
            feature_cache=feature_cache,
            output_path=smoke_dir / "cap_k_calibration_pilot.json",
        )


def _verify_frozen_model_and_cache(
    args: argparse.Namespace,
    *,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    device: torch.device,
    smoke_dir: Path,
) -> None:
    """Exercise pretrained loading, feature shape, learning, and cache equivalence."""

    model = build_frozen_resnet50_head(
        DEVELOPMENT_SEEDS[0], hidden_dim=args.head_hidden_dim
    ).to(device)
    counts = model_parameter_counts(model)
    subset = list(eval_records[: min(4, len(eval_records))])
    online = ManifestImageDataset(
        subset,
        data_root=data_root,
        domain_id=1,
        benchmark_seed=0,
        segment_index=None,
        training=False,
    )
    images, online_labels = next(
        iter(_loader(online, batch_size=len(subset), workers=0, seed=0))
    )
    with torch.inference_mode():
        online_features = model.extract_features(images.to(device)).cpu()
    if online_features.shape != (len(subset), model.feature_dim):
        raise RuntimeError("pretrained ResNet-50 returned an unexpected feature shape")
    micro_cache = FeatureCacheStore(
        smoke_dir / "feature_cache_equivalence",
        rebuild=True,
        data_root=data_root,
        manifest_hash=f"{manifest_hash}:smoke-subset",
        train_records=train_records[:1],
        eval_records=subset,
        workers=0,
        batch_size=len(subset),
    )
    cached = micro_cache.dataset(
        subset, split="eval", domain_id=1, model=model, device=device
    )
    cached_features, cached_labels = next(
        iter(_loader(cached, batch_size=len(subset), workers=0, seed=0))
    )
    torch.testing.assert_close(cached_features, online_features, rtol=1e-5, atol=1e-6)
    if not torch.equal(cached_labels, online_labels):
        raise RuntimeError("cached and online labels/order differ")
    before = [parameter.detach().clone() for parameter in model.head.parameters()]
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(
        model(cached_features.to(device)), cached_labels.to(device)
    )
    loss.backward()
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("backbone received gradients in smoke verification")
    optimizer.step()
    if not any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.head.parameters())
    ):
        raise RuntimeError("continual head did not learn in smoke verification")
    atomic_write_json(
        smoke_dir / "frozen_backbone_verification.json",
        {
            "backbone": "torchvision.models.resnet50",
            "weights": BACKBONE_WEIGHTS,
            "feature_dim": model.feature_dim,
            "head_hidden_dim": model.hidden_dim,
            **counts,
            "backbone_mode": "eval" if not model.backbone.training else "train",
            "backbone_gradients_none": True,
            "head_changed_after_step": True,
            "online_cache_allclose": True,
            "cache_rtol": 1e-5,
            "cache_atol": 1e-6,
        },
    )


def _run_clean_sanity(
    args: argparse.Namespace,
    *,
    seed: int,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    output_path: Path,
    feature_cache: FeatureCacheStore,
) -> None:
    """Train only on clean images, then sanity-evaluate all fixed domains."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    model = build_frozen_resnet50_head(seed, hidden_dim=args.head_hidden_dim).to(device)
    bundle = build_method("adamw", model.head, default_config("adamw"))
    selected = select_segment_records(train_records, seed, 0)[
        : args.smoke_batch_size * args.smoke_updates_per_segment
    ]
    dataset = feature_cache.dataset(
        selected, split="train", domain_id=0, model=model, device=device
    )
    loader = _loader(
        dataset,
        batch_size=args.smoke_batch_size,
        workers=args.num_workers,
        seed=keyed_seed("clean_sanity_loader", seed),
    )
    model.train()
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bundle.optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        bundle.optimizer.step()
    bank = select_evaluation_bank(eval_records, seed, args.smoke_eval_per_class)
    rows = [
        {
            "training_domain": "clean",
            "training_updates": args.smoke_updates_per_segment,
            "evaluated_domain": domain,
            "severity": DOMAIN_SEVERITY,
            "accuracy": evaluate(
                model,
                bank,
                data_root=data_root,
                domain_id=domain_id,
                seed=seed,
                device=device,
                workers=args.num_workers,
                batch_size=args.smoke_batch_size,
                feature_cache=feature_cache,
            ),
        }
        for domain_id, domain in enumerate(DOMAINS)
    ]
    atomic_write_csv(output_path, rows, tuple(rows[0]))


def _run_cap_calibration_pilot(
    args: argparse.Namespace,
    *,
    seed: int,
    data_root: Path,
    train_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    output_path: Path,
    feature_cache: FeatureCacheStore,
) -> None:
    """Accuracy-blind CAP pilot; natural datasets use their first two visits."""

    provisional_k = args.cap_k_center or 7e-5
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    model = build_frozen_resnet50_head(seed, hidden_dim=args.head_hidden_dim).to(device)
    bundle = build_method("cap", model.head, default_config("cap", cap_k=provisional_k))
    diagnostic_rows: list[dict[str, Any]] = []
    adapter = feature_cache.adapter
    pilot_domains = ([row["domain_id"] for row in adapter.schedule(seed)["segments"][:2]]
                     if adapter.full_evaluation else [0, 1])
    for segment_index, domain_id in enumerate(pilot_domains):
        selected = adapter.select(train_records, seed, segment_index, domain_id,
                                  args.smoke_batch_size * args.smoke_updates_per_segment)
        dataset = feature_cache.dataset(
            selected,
            split="train",
            domain_id=domain_id,
            model=model,
            device=device,
        )
        loader = _loader(
            dataset,
            batch_size=args.smoke_batch_size,
            workers=args.num_workers,
            seed=keyed_seed("cap_calibration_loader", seed, segment_index),
        )
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bundle.optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            bundle.optimizer.step()
        diagnostic_rows.append(
            {
                "segment": segment_index + 1,
                "domain": adapter.domains[domain_id],
                **cap_diagnostics(bundle.optimizer),
                "parameter_norm": parameter_norm(model.head),
            }
        )
    atomic_write_json(
        output_path,
        {
            "experiment_version": adapter.experiment_version,
            "model": MODEL_NAME,
            "backbone_weights": BACKBONE_WEIGHTS,
            "head_hidden_dim": args.head_hidden_dim,
            "pilot_k": provisional_k,
            "pilot_domains": [adapter.domains[d] for d in pilot_domains],
            "seed": seed,
            "pilot_is_accuracy_blind": True,
            "updates_per_domain": args.smoke_updates_per_segment,
            "diagnostics": diagnostic_rows,
            "selection_rule": (
                "Choose one --cap-k-center with finite radii, nonzero but not "
                "near-100% binding, and no near-total freezing; do not use accuracy."
            ),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--train", action="store_true")
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument("--method", choices=METHODS)
    method.add_argument("--all-methods", action="store_true")
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument("--seeds", type=int, nargs="+")
    seed.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/miniimagenet_resnet/data_manifest"),
        help="Prepared data manifest (read-only and reusable across experiment versions).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=DEFAULT_HEAD_HIDDEN_DIM,
        help="Width of the continually trained two-linear MLP head (default: 300).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Optional source/domain frozen-feature cache; built lazily when missing.",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Recompute cache entries used in this invocation (requires --feature-cache-dir).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=UPDATES_PER_SEGMENT,
        help=(
            "Mid-segment checkpoint cadence. The default checkpoints at segment "
            "boundaries; choose a smaller value for finer interruption recovery."
        ),
    )
    parser.add_argument(
        "--cap-k-center",
        type=float,
        help=(
            "Head-scale CAP k center chosen accuracy-blind from the smoke pilot. "
            "Required for CAP tuning; smoke uses 7e-5 only as a provisional probe."
        ),
    )
    parser.add_argument("--smoke-segments", type=int, default=2)
    parser.add_argument("--smoke-updates-per-segment", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    parser.add_argument("--smoke-eval-per-class", type=int, default=2)
    return parser.parse_args(argv)


def _resolve_methods_and_seeds(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...]]:
    methods = METHODS if args.all_methods else (str(args.method),)
    if args.tune and len(methods) != 1:
        raise ValueError("--tune requires exactly one --method")
    if args.all_seeds:
        seeds = FINAL_SEEDS if args.train else DEVELOPMENT_SEEDS
    elif args.seeds:
        seeds = tuple(int(seed) for seed in args.seeds)
    elif args.train:
        seeds = FINAL_SEEDS
    else:
        seeds = (DEVELOPMENT_SEEDS[0],)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list contains duplicates")
    if args.train and any(seed not in FINAL_SEEDS for seed in seeds):
        raise ValueError(f"final training seeds must be drawn from {FINAL_SEEDS}")
    if args.tune and args.seeds:
        raise ValueError("tuning uses the frozen disjoint development seeds")
    return methods, seeds


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_workers < 0 or args.checkpoint_every_updates < 1:
        raise ValueError("worker and checkpoint settings must be non-negative/positive")
    if args.cap_k_center is not None and args.cap_k_center <= 0:
        raise ValueError("--cap-k-center must be positive")
    if args.head_hidden_dim < 1:
        raise ValueError("--head-hidden-dim must be positive")
    if args.rebuild_feature_cache and args.feature_cache_dir is None:
        raise ValueError("--rebuild-feature-cache requires --feature-cache-dir")
    if not 1 <= args.smoke_eval_per_class <= 100:
        raise ValueError("--smoke-eval-per-class must be in 1..100")
    methods, seeds = _resolve_methods_and_seeds(args)
    data_root, train_records, eval_records, metadata = load_manifest_bundle(
        args.manifest_dir, args.data_root
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    feature_cache = FeatureCacheStore(
        args.feature_cache_dir,
        rebuild=args.rebuild_feature_cache,
        data_root=data_root,
        manifest_hash=str(metadata["manifest_hash"]),
        train_records=train_records,
        eval_records=eval_records,
        workers=args.num_workers,
        batch_size=BATCH_SIZE,
    )

    if args.smoke:
        run_smoke(
            args,
            methods,
            seeds,
            data_root=data_root,
            train_records=train_records,
            eval_records=eval_records,
            manifest_metadata=metadata,
            device=device,
            feature_cache=feature_cache,
        )
    elif args.tune:
        run_tuning(
            args,
            method=methods[0],
            data_root=data_root,
            train_records=train_records,
            eval_records=eval_records,
            manifest_metadata=metadata,
            device=device,
            feature_cache=feature_cache,
        )
    else:
        for method in methods:
            selected_path = args.output_dir / "selected_configs" / f"{method}.json"
            if not selected_path.exists():
                raise FileNotFoundError(
                    f"final training requires frozen selected config: {selected_path}"
                )
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            if selected.get("experiment_version") != EXPERIMENT_VERSION:
                raise ValueError(f"selected config has wrong experiment version: {selected_path}")
            if selected.get("model") != MODEL_NAME or int(
                selected.get("head_hidden_dim", -1)
            ) != args.head_hidden_dim:
                raise ValueError(
                    f"selected config targets a different model/head width: {selected_path}"
                )
            config = selected["config"]
            if selected.get("selected_config_hash") != payload_hash(config):
                raise ValueError(f"selected config hash mismatch: {selected_path}")
            for seed in seeds:
                run_single(
                    method=method,
                    seed=seed,
                    config=config,
                    run_dir=args.output_dir / "train" / method / f"seed_{seed}",
                    output_dir=args.output_dir,
                    data_root=data_root,
                    train_records=train_records,
                    eval_records=eval_records,
                    manifest_metadata=metadata,
                    device=device,
                    workers=args.num_workers,
                    batch_size=BATCH_SIZE,
                    updates_per_segment=UPDATES_PER_SEGMENT,
                    max_segments=45,
                    resume=args.resume,
                    checkpoint_every_updates=args.checkpoint_every_updates,
                    evaluation_per_class=FAST_EVAL_PER_CLASS,
                    head_hidden_dim=args.head_hidden_dim,
                    feature_cache=feature_cache,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
