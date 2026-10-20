from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import prepare_miniimagenet_scale as preparation
from experiment_miniimagenet_resnet import FeatureCacheStore, parse_args
from miniimagenet_domains import (
    ManifestImageDataset,
    apply_domain,
    mild_train_augmentation,
    to_pretrained_resnet50_tensor,
    to_normalized_tensor,
    validate_corruptions,
)
from miniimagenet_scale_common import (
    DOMAINS,
    METHODS,
    NUM_CLASSES,
    BACKBONE_WEIGHTS,
    DEFAULT_HEAD_HIDDEN_DIM,
    EXPERIMENT_VERSION,
    FINAL_SEEDS,
    build_domain_schedule,
    build_frozen_resnet50_head,
    build_method,
    capture_rng_state,
    default_config,
    payload_hash,
    protected_parameter_names,
    restore_rng_state,
    cap_parameter_groups,
    sample_order_hash,
    scale_metrics,
    select_evaluation_bank,
    select_segment_records,
    tuning_grid,
    validate_schedule,
)


def _train_records() -> list[dict[str, object]]:
    return [
        {
            "class_index": class_index,
            "source_id": f"class_{class_index:03d}/train_{index:03d}.jpg",
            "source_path": f"class_{class_index:03d}/train_{index:03d}.jpg",
        }
        for class_index in range(100)
        for index in range(500)
    ]


def _eval_records() -> list[dict[str, object]]:
    return [
        {
            "class_index": class_index,
            "source_id": f"class_{class_index:03d}/eval_{index:03d}.jpg",
            "source_path": f"class_{class_index:03d}/eval_{index:03d}.jpg",
        }
        for class_index in range(100)
        for index in range(100)
    ]


def test_schedule_is_balanced_recurrent_and_deterministic() -> None:
    first = build_domain_schedule(42)
    second = build_domain_schedule(42)
    assert first == second
    validate_schedule(first)
    ids = [row["domain_id"] for row in first]
    assert len(ids) == 45
    assert Counter(ids) == Counter({index: 5 for index in range(9)})
    assert all(ids[boundary - 1] != ids[boundary] for boundary in (9, 18, 27, 36))
    for row in first:
        prior = row["previous_visit_segment"]
        if prior is not None:
            assert row["revisit_gap"] == row["segment_index"] - prior - 1


def test_segment_sampling_is_exact_balanced_paired_and_without_replacement() -> None:
    records = _train_records()
    first = select_segment_records(records, 123, 7)
    second = select_segment_records(records, 123, 7)
    assert sample_order_hash(first) == sample_order_hash(second)
    assert len(first) == 25_600
    assert Counter(int(row["class_index"]) for row in first) == Counter(
        {index: 256 for index in range(100)}
    )
    assert len({row["source_id"] for row in first}) == len(first)
    assert sample_order_hash(first) != sample_order_hash(
        select_segment_records(records, 123, 8)
    )


def test_evaluation_bank_is_fixed_and_class_balanced() -> None:
    records = _eval_records()
    bank = select_evaluation_bank(records, 42, 20)
    assert len(bank) == 2_000
    assert Counter(int(row["class_index"]) for row in bank) == Counter(
        {index: 20 for index in range(100)}
    )
    assert bank == select_evaluation_bank(records, 42, 20)


def test_metric_definitions_use_one_indexed_segments_23_through_45() -> None:
    rows = [
        {
            "segment": segment,
            "mean_online_accuracy": segment / 100,
            "pre_visit_accuracy": (100 - segment) / 100,
            "revisit_gap": "" if segment <= 9 else 4,
        }
        for segment in range(1, 46)
    ]
    metrics = scale_metrics(rows)
    assert metrics["P_scale"] == pytest.approx(np.mean(np.arange(23, 46) / 100))
    assert metrics["R_scale"] == pytest.approx(
        np.mean((100 - np.arange(23, 46)) / 100)
    )
    assert metrics["B_scale"] == min(metrics["P_scale"], metrics["R_scale"])


def test_all_corruptions_are_valid_and_seeded() -> None:
    gradient = np.linspace(0, 255, 84 * 84 * 3, dtype=np.uint8).reshape(84, 84, 3)
    image = Image.fromarray(gradient, mode="RGB")
    report = validate_corruptions(image)
    assert set(report) == set(DOMAINS)
    for domain in DOMAINS:
        first = np.asarray(apply_domain(image, domain, seed=77))
        second = np.asarray(apply_domain(image, domain, seed=77))
        assert np.array_equal(first, second)
        assert first.shape == (84, 84, 3)
    augmented = mild_train_augmentation(image, seed=91)
    assert np.array_equal(
        np.asarray(augmented), np.asarray(mild_train_augmentation(image, seed=91))
    )
    tensor = to_normalized_tensor(image)
    assert tensor.shape == (3, 84, 84)
    assert torch.isfinite(tensor).all()
    pretrained_tensor = to_pretrained_resnet50_tensor(image)
    assert pretrained_tensor.shape == (3, 224, 224)
    assert torch.isfinite(pretrained_tensor).all()


def test_manifest_dataset_pairs_corruption_across_methods(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (84, 84), (70, 120, 180)).save(path)
    records = [{"source_id": "sample.jpg", "source_path": "sample.jpg", "class_index": 4}]
    first = ManifestImageDataset(
        records,
        data_root=tmp_path,
        domain_id=1,
        benchmark_seed=42,
        segment_index=3,
        training=True,
    )
    second = ManifestImageDataset(
        records,
        data_root=tmp_path,
        domain_id=1,
        benchmark_seed=42,
        segment_index=3,
        training=True,
    )
    assert torch.equal(first[0][0], second[0][0])
    assert first[0][1] == second[0][1] == 4
    assert first[0][0].shape == (3, 224, 224)


def test_resnet50_backbone_is_frozen_eval_and_head_only_is_trainable() -> None:
    pytest.importorskip("torchvision")
    model = build_frozen_resnet50_head(42, pretrained=False)
    assert model.feature_dim == 2048
    assert model.hidden_dim == DEFAULT_HEAD_HIDDEN_DIM == 300
    assert model(torch.zeros(2, 2048)).shape == (2, 100)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    model.train()
    assert not model.backbone.training
    assert model.head.training
    assert BACKBONE_WEIGHTS == "ResNet50_Weights.IMAGENET1K_V2"
    assert EXPERIMENT_VERSION == "miniimagenet-frozen-resnet50-head-v1"


def test_all_optimizers_are_scoped_to_head_and_backbone_grads_stay_none() -> None:
    pytest.importorskip("torchvision")
    model = build_frozen_resnet50_head(42, hidden_dim=8, pretrained=False)
    for method in METHODS:
        bundle = build_method(method, model.head, {
            "adamw": {"lr": 1e-3, "weight_decay": 0.0},
            "cap": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "k": 1e-4,
                "T_a": 2000,
                "T_s": 2000,
                "T_m": 10000,
            },
            "upgd": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "beta_utility": 0.9999,
                "sigma": 1e-3,
            },
            "ewc": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "lambda": 10.0,
                "fisher_decay": 0.95,
            },
        }[method])
        optimizer_ids = {
            id(parameter)
            for group in bundle.optimizer.param_groups
            for parameter in group["params"]
        }
        assert optimizer_ids == {id(parameter) for parameter in model.head.parameters()}
    loss = model(torch.randn(2, 2048)).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_feature_cache_matches_online_features_labels_and_order(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    records: list[dict[str, object]] = []
    for index, color in enumerate(((10, 20, 30), (80, 90, 100), (150, 160, 170))):
        path = tmp_path / f"sample_{index}.jpg"
        Image.new("RGB", (84, 84), color).save(path)
        records.append(
            {"source_id": path.name, "source_path": path.name, "class_index": index}
        )

    class TinyFrozenModel(nn.Module):
        feature_dim = 3

        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

        @torch.inference_mode()
        def extract_features(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    model = TinyFrozenModel()
    online_dataset = ManifestImageDataset(
        records,
        data_root=tmp_path,
        domain_id=1,
        benchmark_seed=9,
        segment_index=None,
        training=False,
    )
    online_images = torch.stack([online_dataset[index][0] for index in range(3)])
    online_features = model.extract_features(online_images)
    store = FeatureCacheStore(
        tmp_path / "cache",
        rebuild=False,
        data_root=tmp_path,
        manifest_hash="test-manifest",
        train_records=records,
        eval_records=records,
        workers=0,
        batch_size=2,
    )
    reordered = [records[2], records[0]]
    cached = store.dataset(
        reordered,
        split="eval",
        domain_id=1,
        model=model,
        device=torch.device("cpu"),
    )
    assert torch.allclose(cached.tensors[0], online_features[[2, 0]], atol=1e-6)
    assert torch.equal(cached.tensors[1], torch.tensor([2, 0]))


def test_cap_scope_is_only_head_linear_weights() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, bias=True),
        nn.GroupNorm(4, 8),
        nn.Flatten(),
        nn.Linear(8 * 82 * 82, 3, bias=True),
    )
    assert protected_parameter_names(model) == {"3.weight"}
    groups = cap_parameter_groups(model)
    assert groups[0]["protect"] is True
    assert groups[1]["protect"] is False
    protected_ids = {id(parameter) for parameter in groups[0]["params"]}
    assert protected_ids == {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name == "3.weight"
    }


def test_method_registry_and_cli_exclude_experience_replay() -> None:
    assert METHODS == ("adamw", "cap", "upgd", "ewc")
    assert FINAL_SEEDS == (42, 123, 2026, 451, 789)
    cap = default_config("cap")
    assert (cap["T_a"], cap["T_s"], cap["T_m"]) == (2000.0, 2000.0, 10000.0)
    assert cap["k"] != 2e-6
    for method in METHODS:
        assert tuning_grid(method)
    with pytest.raises(SystemExit):
        parse_args(["--smoke", "--method", "replay"])


def test_cap_and_ewc_method_mechanics() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.GroupNorm(1, 4), nn.Linear(4, 2))
    cap = build_method(
        "cap",
        model,
        {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "k": 1e-4,
            "T_a": 2000,
            "T_s": 2000,
            "T_m": 10_000,
        },
    )
    assert cap.optimizer.param_groups[0]["release_update"] == "contest_blocked"
    assert cap.optimizer.param_groups[0]["box_geometry"] == "coordinate"
    assert cap.optimizer.param_groups[0]["momentum_response"] == "none"

    model = nn.Linear(4, 2)
    ewc = build_method(
        "ewc",
        model,
        {"lr": 1e-3, "weight_decay": 0.0, "lambda": 10.0, "fisher_decay": 0.95},
    )
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    ewc.after_backward()
    ewc.optimizer.step()
    ewc.on_boundary()
    assert ewc.ewc is not None
    assert ewc.ewc.state["theta_star"] is not None
    assert ewc.ewc.state["segment_steps"] == 0


def test_rng_checkpoint_resume_matches_uninterrupted() -> None:
    def initialize() -> tuple[nn.Module, torch.optim.Optimizer]:
        torch.manual_seed(123)
        model = nn.Linear(3, 2)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

    def steps(model: nn.Module, optimizer: torch.optim.Optimizer, count: int) -> None:
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(5, 3)).square().mean()
            loss.backward()
            optimizer.step()

    uninterrupted, uninterrupted_optimizer = initialize()
    steps(uninterrupted, uninterrupted_optimizer, 4)

    interrupted, interrupted_optimizer = initialize()
    steps(interrupted, interrupted_optimizer, 2)
    checkpoint = {
        "model": interrupted.state_dict(),
        "optimizer": interrupted_optimizer.state_dict(),
        "rng": capture_rng_state(),
    }
    resumed, resumed_optimizer = initialize()
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng"])
    steps(resumed, resumed_optimizer, 2)
    assert all(
        torch.equal(left, right)
        for left, right in zip(uninterrupted.parameters(), resumed.parameters())
    )


def test_cuda_rng_checkpoint_state_is_restored_from_cpu_byte_tensors() -> None:
    state = capture_rng_state()
    state["cuda"] = [torch.arange(16, dtype=torch.uint8)]
    with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
        torch.cuda, "set_rng_state_all"
    ) as set_rng_state_all:
        restore_rng_state(state)
    restored = set_rng_state_all.call_args.args[0]
    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_manifest_split_is_deterministic_and_leak_free(tmp_path: Path) -> None:
    examples = [
        (f"class_{class_index:03d}", tmp_path / f"class_{class_index:03d}" / f"{index:03d}.jpg")
        for class_index in range(100)
        for index in range(600)
    ]
    with mock.patch.object(preparation, "discover_examples", return_value=examples), mock.patch.object(
        preparation,
        "_relative",
        side_effect=lambda path, root: path.relative_to(root).as_posix(),
    ):
        first, first_hash = preparation.build_manifests(tmp_path)
        second, second_hash = preparation.build_manifests(tmp_path)
    assert first_hash == second_hash
    assert first == second
    assert len(first) == 60_000
    train_ids = {row["source_id"] for row in first if row["split"] == "train"}
    eval_ids = {row["source_id"] for row in first if row["split"] == "eval"}
    assert len(train_ids) == 50_000
    assert len(eval_ids) == 10_000
    assert train_ids.isdisjoint(eval_ids)
