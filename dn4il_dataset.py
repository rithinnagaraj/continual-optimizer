"""Official DN4IL annotations and the natural-domain frozen-feature adapter.

No downloads, relabeling, custom splits, class balancing, or corruptions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from miniimagenet_domains import to_pretrained_resnet50_tensor
from miniimagenet_scale_common import (
    BATCH_SIZE, UPDATES_PER_SEGMENT, atomic_write_json, file_hash, keyed_seed,
    payload_hash, schedule_with_hash,
)

DOMAINS = ("real", "clipart", "infograph", "painting", "quickdraw", "sketch")
EXPERIMENT_VERSION = "dn4il-frozen-resnet50-head-recurrent-v1"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "DomainNet"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data" / "DN4IL-dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dn4il_frozen_resnet50_head"


class DN4ILDataset(Dataset):
    def __init__(self, data_root: Path, split_file: Path, domain: str, split: str):
        if domain not in DOMAINS or split not in ("train", "test"):
            raise ValueError("invalid DN4IL domain or split")
        self.data_root = Path(data_root).resolve()
        self.split_file = Path(split_file).resolve()
        self.domain, self.split = domain, split
        self.records: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for source_index, line in enumerate(self.split_file.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            try:
                raw_path, raw_label = line.rsplit(maxsplit=1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(f"invalid annotation {self.split_file}:{source_index + 1}") from error
            if not 0 <= label < 100:
                raise ValueError(f"label outside 0..99 at {self.split_file}:{source_index + 1}")
            relative = PurePosixPath(raw_path.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3 or relative.parts[0] != domain:
                raise ValueError(f"invalid domain-relative image path: {raw_path}")
            absolute = (self.data_root / str(relative)).resolve()
            if not absolute.is_relative_to(self.data_root):
                raise ValueError(f"image escapes data root: {raw_path}")
            if not absolute.is_file():
                raise FileNotFoundError(f"referenced image is missing: {absolute}")
            if absolute in seen:
                raise ValueError(f"duplicate image in split: {absolute}")
            seen.add(absolute)
            self.records.append({
                "absolute_path": str(absolute), "relative_path": str(relative),
                "label": label, "domain": domain, "split": split,
                "source_index": source_index, "class_name": relative.parts[1],
                "source_id": str(relative), "source_path": str(relative),
                "class_index": label,
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["absolute_path"]) as image:
            return to_pretrained_resnet50_tensor(image), row["label"]


class DN4ILImages(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["absolute_path"]) as image:
            return to_pretrained_resnet50_tensor(image), int(row["class_index"])


def validate_dn4il(data_root: Path, splits_root: Path, manifest_dir: Path | None = None):
    data_root, splits_root = Path(data_root).resolve(), Path(splits_root).resolve()
    train, test, domain_counts, split_hashes = [], [], {}, {}
    label_classes: dict[int, str] = {}
    class_labels: dict[str, int] = {}
    for domain in DOMAINS:
        if not (data_root / domain).is_dir():
            raise FileNotFoundError(f"missing DN4IL image domain: {data_root / domain}")
        domain_rows = {}
        domain_counts[domain] = {}
        for split in ("train", "test"):
            split_file = splits_root / f"{domain}_{split}.txt"
            dataset = DN4ILDataset(data_root, split_file, domain, split)
            rows = dataset.records
            counts = Counter(row["label"] for row in rows)
            if set(counts) != set(range(100)):
                raise ValueError(f"{split_file} must contain all 100 labels 0..99")
            for row in rows:
                label, name = row["label"], row["class_name"]
                if label_classes.setdefault(label, name) != name or class_labels.setdefault(name, label) != label:
                    raise ValueError(f"inconsistent semantic class mapping: {split_file}, label {label}, class {name}")
            domain_rows[split] = rows
            domain_counts[domain][split] = {"count": len(rows), "class_counts": dict(sorted(counts.items()))}
            split_hashes[split_file.name] = file_hash(split_file)
            (train if split == "train" else test).extend(rows)
        if {row["absolute_path"] for row in domain_rows["train"]} & {row["absolute_path"] for row in domain_rows["test"]}:
            raise ValueError(f"train/test leakage in {domain}")
    metadata = {
        "experiment_version": EXPERIMENT_VERSION, "dataset_version": "official-DN4IL-100",
        "domains": list(DOMAINS), "data_root": str(data_root), "splits_root": str(splits_root),
        "domain_counts": domain_counts, "split_file_hashes": split_hashes,
        "label_to_class": label_classes, "num_classes": 100, "train_test_leakage": False,
    }
    metadata["manifest_hash"] = payload_hash(metadata)
    if manifest_dir is not None:
        path = Path(manifest_dir) / "dn4il_manifest.json"
        if path.exists():
            import json
            if json.loads(path.read_text(encoding="utf-8")).get("manifest_hash") != metadata["manifest_hash"]:
                raise RuntimeError(f"dataset identity changed; use a new output directory: {path}")
        atomic_write_json(path, metadata)
    return data_root, train, test, metadata


def sample_segment(records: Sequence[Mapping[str, Any]], seed: int, segment_index: int,
                   domain: str, count: int = BATCH_SIZE * UPDATES_PER_SEGMENT):
    """Consume seeded permutations of the finite pool, reshuffling on exhaustion."""
    pool = [row for row in records if row["domain"] == domain]
    if not pool or count < 1:
        raise ValueError("segment needs a nonempty domain pool and positive sample count")
    rng = np.random.default_rng(keyed_seed("dn4il_segment", seed, segment_index, domain))
    selected = []
    while len(selected) < count:
        indices = rng.permutation(len(pool))[:count - len(selected)]
        selected.extend(dict(pool[int(index)]) for index in indices)
    return selected


class DN4ILAdapter:
    domains = DOMAINS
    experiment_version = EXPERIMENT_VERSION
    full_evaluation = True
    total_segments = 30

    def __init__(self, metadata):
        self.metadata = metadata

    def schedule(self, seed):
        return schedule_with_hash(seed, domains=self.domains, experiment_version=self.experiment_version)

    def select(self, records, seed, segment_index, domain_id, count):
        return sample_segment(records, seed, segment_index, self.domains[domain_id], count)

    def filter_records(self, records, domain_id):
        return [row for row in records if row["domain"] == self.domains[domain_id]]

    def images(self, records, **kwargs):
        return DN4ILImages(records)

    def cache_metadata(self, split, domain_id):
        from torchvision.models import ResNet50_Weights
        import torchvision
        domain = self.domains[domain_id]
        split = "test" if split == "eval" else split
        return {
            "dataset_version": self.metadata["dataset_version"],
            "split_file_hash": self.metadata["split_file_hashes"][f"{domain}_{split}.txt"],
            "preprocessing_hash": payload_hash({
                "transform": repr(ResNet50_Weights.IMAGENET1K_V2.transforms()),
                "torchvision": torchvision.__version__, "input": "original RGB image",
            }),
        }
