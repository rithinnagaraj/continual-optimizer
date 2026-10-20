"""Build the deterministic 500/100-per-class mini-ImageNet scale manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from miniimagenet_scale_common import (
    DATA_SPLIT_SEED,
    EVAL_PER_CLASS,
    EXPERIMENT_VERSION,
    NUM_CLASSES,
    TRAIN_PER_CLASS,
    atomic_write_csv,
    atomic_write_json,
    payload_hash,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FILENAME_KEYS = ("filename", "file", "image", "path", "source_path")
CSV_LABEL_KEYS = ("label", "class", "wnid", "class_name")
MANIFEST_FIELDS = (
    "class_index",
    "class_name",
    "source_path",
    "source_id",
    "split",
    "within_class_index",
    "manifest_hash",
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_image(raw: str, csv_path: Path, data_root: Path) -> Path | None:
    value = Path(raw)
    candidates = (
        value,
        csv_path.parent / value,
        data_root / value,
        data_root / "images" / value,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate.resolve()
    return None


def discover_from_csv(data_root: Path) -> list[tuple[str, Path]]:
    """Discover common mini-ImageNet CSV layouts without assuming few-shot splits."""

    discovered: dict[str, tuple[str, Path]] = {}
    for csv_path in sorted(data_root.rglob("*.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = {field.lower(): field for field in (reader.fieldnames or [])}
                filename_key = next((fields[key] for key in CSV_FILENAME_KEYS if key in fields), None)
                label_key = next((fields[key] for key in CSV_LABEL_KEYS if key in fields), None)
                if filename_key is None or label_key is None:
                    continue
                for row in reader:
                    path = _resolve_image(str(row[filename_key]), csv_path, data_root)
                    label = str(row[label_key]).strip()
                    if path is None or not label:
                        continue
                    source_id = _relative(path, data_root)
                    previous = discovered.get(source_id)
                    if previous is not None and previous[0] != label:
                        raise ValueError(
                            f"conflicting labels for {source_id}: {previous[0]!r} and {label!r}"
                        )
                    discovered[source_id] = (label, path)
        except (csv.Error, UnicodeDecodeError):
            continue
    return list(discovered.values())


def discover_from_folders(data_root: Path) -> list[tuple[str, Path]]:
    images = sorted(
        path.resolve()
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return [(path.parent.name, path) for path in images]


def discover_examples(data_root: Path) -> list[tuple[str, Path]]:
    csv_examples = discover_from_csv(data_root)
    if csv_examples:
        return csv_examples
    return discover_from_folders(data_root)


def build_manifests(data_root: Path) -> tuple[list[dict[str, Any]], str]:
    examples = discover_examples(data_root)
    by_class: dict[str, list[Path]] = {}
    for label, path in examples:
        by_class.setdefault(label, []).append(path)
    counts = {label: len(paths) for label, paths in by_class.items()}
    if len(by_class) != NUM_CLASSES:
        raise ValueError(
            f"mini-ImageNet must contain 100 classes; found {len(by_class)} with counts {counts}"
        )
    bad = {label: count for label, count in counts.items() if count != 600}
    if bad:
        raise ValueError(f"each class must contain exactly 600 distinct images; invalid: {bad}")

    rng = np.random.default_rng(DATA_SPLIT_SEED)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for class_index, class_name in enumerate(sorted(by_class)):
        paths = sorted(set(by_class[class_name]), key=lambda path: _relative(path, data_root))
        if len(paths) != 600:
            raise ValueError(f"class {class_name!r} contains duplicate source paths")
        permutation = rng.permutation(len(paths))
        for within_class_index, selected_index in enumerate(permutation):
            path = paths[int(selected_index)]
            source_id = _relative(path, data_root)
            if source_id in seen_ids:
                raise ValueError(f"source image appears more than once: {source_id}")
            seen_ids.add(source_id)
            rows.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "source_path": source_id,
                    "source_id": source_id,
                    "split": "train" if within_class_index < TRAIN_PER_CLASS else "eval",
                    "within_class_index": within_class_index,
                }
            )
    expected_total = NUM_CLASSES * (TRAIN_PER_CLASS + EVAL_PER_CLASS)
    if len(rows) != expected_total:
        raise AssertionError(f"manifest has {len(rows)} rows, expected {expected_total}")
    manifest_hash = payload_hash(rows)
    for row in rows:
        row["manifest_hash"] = manifest_hash
    return rows, manifest_hash


def write_manifests(data_root: Path, output_dir: Path) -> dict[str, Any]:
    rows, manifest_hash = build_manifests(data_root)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if len(train_rows) != NUM_CLASSES * TRAIN_PER_CLASS:
        raise AssertionError("train manifest count is inconsistent")
    if len(eval_rows) != NUM_CLASSES * EVAL_PER_CLASS:
        raise AssertionError("eval manifest count is inconsistent")
    train_ids = {row["source_id"] for row in train_rows}
    eval_ids = {row["source_id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise AssertionError("train/evaluation split leakage detected")

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "data_root": str(data_root.resolve()),
        "data_split_seed": DATA_SPLIT_SEED,
        "num_classes": NUM_CLASSES,
        "images_per_class": TRAIN_PER_CLASS + EVAL_PER_CLASS,
        "train_per_class": TRAIN_PER_CLASS,
        "eval_per_class": EVAL_PER_CLASS,
        "total_images": len(rows),
        "manifest_hash": manifest_hash,
        "records": rows,
    }
    atomic_write_json(output_dir / "miniimagenet_manifest.json", metadata)
    atomic_write_csv(
        output_dir / "miniimagenet_train_manifest.csv", train_rows, MANIFEST_FIELDS
    )
    atomic_write_csv(
        output_dir / "miniimagenet_eval_manifest.csv", eval_rows, MANIFEST_FIELDS
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/miniimagenet_resnet/data_manifest"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"mini-ImageNet data root is not a directory: {args.data_root}")
    metadata = write_manifests(args.data_root, args.output_dir)
    print(
        f"Wrote 50,000 train and 10,000 evaluation rows to {args.output_dir} "
        f"(manifest {metadata['manifest_hash'][:12]})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

