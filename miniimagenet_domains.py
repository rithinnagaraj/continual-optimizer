"""Deterministic visual domains and pretrained ResNet-50 preprocessing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset

from miniimagenet_scale_common import DOMAINS, DOMAIN_SEVERITY, keyed_seed


IMAGE_SIZE = 84
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
PREPROCESSING_NAME = "ResNet50_Weights.IMAGENET1K_V2.transforms"


def _as_rgb84(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected an 84x84 source image, found {image.size}")
    return image


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def _image(values: np.ndarray) -> Image.Image:
    values = np.clip(values * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def _motion_blur(image: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    length = (3, 5, 7, 9, 11)[severity - 1]
    angle = float(rng.uniform(0, np.pi))
    values = _array(image)
    total = np.zeros_like(values)
    center = (length - 1) / 2.0
    for index in range(length):
        distance = index - center
        dx = int(round(np.cos(angle) * distance))
        dy = int(round(np.sin(angle) * distance))
        total += np.roll(values, shift=(dy, dx), axis=(0, 1))
    return _image(total / length)


def _fog(image: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    coarse = rng.normal(size=(12, 12)).astype(np.float32)
    coarse = (coarse - coarse.min()) / max(float(np.ptp(coarse)), 1e-6)
    field = Image.fromarray((coarse * 255).astype(np.uint8), mode="L").resize(
        image.size, Image.Resampling.BICUBIC
    )
    field = field.filter(ImageFilter.GaussianBlur(radius=4.0))
    haze = np.asarray(field, dtype=np.float32)[..., None] / 255.0
    strength = (0.12, 0.18, 0.25, 0.32, 0.40)[severity - 1]
    values = _array(image)
    return _image(values * (1.0 - strength) + strength * (0.75 + 0.25 * haze))


def _frost(image: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    noise = rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    mask_image = Image.fromarray((noise * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=1.2))
    mask = np.asarray(mask_image, dtype=np.float32)[..., None] / 255.0
    frost_color = np.array([0.80, 0.91, 1.0], dtype=np.float32)
    strength = (0.12, 0.18, 0.24, 0.30, 0.36)[severity - 1]
    values = _array(image)
    textured_strength = strength * (0.45 + mask)
    return _image(values * (1.0 - textured_strength) + frost_color * textured_strength)


def apply_domain(
    image: Image.Image,
    domain: str | int,
    *,
    severity: int = DOMAIN_SEVERITY,
    seed: int = 0,
) -> Image.Image:
    """Apply one frozen domain corruption using only the supplied RNG seed."""

    image = _as_rgb84(image)
    if isinstance(domain, int):
        domain = DOMAINS[domain]
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    if severity not in range(1, 6):
        raise ValueError("corruption severity must be in 1..5")
    rng = np.random.default_rng(int(seed))
    if domain == "clean":
        return image.copy()
    if domain == "gaussian_noise":
        sigma = (0.04, 0.06, 0.09, 0.12, 0.16)[severity - 1]
        return _image(_array(image) + rng.normal(0.0, sigma, (*image.size[::-1], 3)))
    if domain == "shot_noise":
        scale = (120, 80, 50, 30, 18)[severity - 1]
        return _image(rng.poisson(_array(image) * scale) / scale)
    if domain == "defocus_blur":
        radius = (0.5, 0.8, 1.2, 1.7, 2.3)[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if domain == "motion_blur":
        return _motion_blur(image, severity, rng)
    if domain == "fog":
        return _fog(image, severity, rng)
    if domain == "frost":
        return _frost(image, severity, rng)
    if domain == "contrast":
        factor = (0.75, 0.60, 0.45, 0.30, 0.18)[severity - 1]
        return ImageEnhance.Contrast(image).enhance(factor)
    if domain == "jpeg_compression":
        quality = (65, 50, 35, 25, 15)[severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    raise AssertionError("domain dispatch is exhaustive")


def mild_train_augmentation(image: Image.Image, *, seed: int) -> Image.Image:
    """Deterministic horizontal flip plus four-pixel reflected padded crop."""

    rng = np.random.default_rng(int(seed))
    values = np.asarray(_as_rgb84(image))
    padded = np.pad(values, ((4, 4), (4, 4), (0, 0)), mode="reflect")
    top = int(rng.integers(0, 9))
    left = int(rng.integers(0, 9))
    cropped = padded[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
    if bool(rng.integers(0, 2)):
        cropped = cropped[:, ::-1]
    return Image.fromarray(np.ascontiguousarray(cropped), mode="RGB")


def to_normalized_tensor(image: Image.Image) -> Tensor:
    values = np.asarray(_as_rgb84(image), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values.transpose(2, 0, 1)).contiguous()
    mean = torch.tensor(NORMALIZATION_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(NORMALIZATION_STD, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean) / std


def to_pretrained_resnet50_tensor(image: Image.Image) -> Tensor:
    """Apply the transform bundled with the exact pretrained backbone weights."""

    from torchvision.models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V2.transforms()(image.convert("RGB"))


def source_domain_corruption_seed(source_id: str, domain_id: int, severity: int) -> int:
    """Stable seed making each corrupted source/domain feature cacheable."""

    return keyed_seed("frozen_feature_corruption", source_id, domain_id, severity)


class ManifestImageDataset(Dataset[tuple[Tensor, int]]):
    """Manifest-backed deterministic train or evaluation view of one domain."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        data_root: Path,
        domain_id: int,
        benchmark_seed: int,
        segment_index: int | None,
        training: bool,
        severity: int = DOMAIN_SEVERITY,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.data_root = data_root
        self.domain_id = int(domain_id)
        self.benchmark_seed = int(benchmark_seed)
        self.segment_index = segment_index
        self.training = bool(training)
        self.severity = int(severity)
        if self.training and segment_index is None:
            raise ValueError("training datasets require a segment index")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> tuple[Tensor, int]:
        record = self.records[position]
        source_id = str(record["source_id"])
        path_value = Path(str(record.get("source_path", source_id)))
        path = path_value if path_value.is_absolute() else self.data_root / path_value
        with Image.open(path) as source:
            image = _as_rgb84(source)
        # Pretrained feature extraction is deterministic and method-independent.
        # In particular, a source/domain pair is identical in train, evaluation,
        # online, and cached paths; there is no method-specific augmentation.
        corruption_seed = source_domain_corruption_seed(
            source_id, self.domain_id, self.severity
        )
        image = apply_domain(
            image,
            self.domain_id,
            severity=self.severity,
            seed=corruption_seed,
        )
        return to_pretrained_resnet50_tensor(image), int(record["class_index"])


def write_corruption_grid(
    records: Sequence[Mapping[str, Any]], data_root: Path, output_path: Path
) -> None:
    """Write a deterministic source-by-domain visual sanity grid."""

    if not records:
        raise ValueError("cannot render an empty corruption grid")
    chosen = list(records[: min(6, len(records))])
    margin = 2
    canvas = Image.new(
        "RGB",
        (
            len(DOMAINS) * (IMAGE_SIZE + margin) + margin,
            len(chosen) * (IMAGE_SIZE + margin) + margin,
        ),
        "white",
    )
    for row, record in enumerate(chosen):
        path_value = Path(str(record.get("source_path", record["source_id"])))
        path = path_value if path_value.is_absolute() else data_root / path_value
        with Image.open(path) as handle:
            source = _as_rgb84(handle)
        for column, domain in enumerate(DOMAINS):
            transformed = apply_domain(
                source,
                domain,
                seed=keyed_seed("sanity_grid", record["source_id"], domain),
            )
            canvas.paste(
                transformed,
                (margin + column * (IMAGE_SIZE + margin), margin + row * (IMAGE_SIZE + margin)),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def validate_corruptions(image: Image.Image) -> dict[str, dict[str, float]]:
    """Numerically validate every domain on one representative 84x84 image."""

    report: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        transformed = apply_domain(image, domain, seed=keyed_seed("domain_test", domain))
        values = np.asarray(transformed, dtype=np.float32) / 255.0
        if transformed.mode != "RGB" or transformed.size != (84, 84):
            raise RuntimeError(f"{domain} returned invalid shape/mode")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{domain} returned non-finite pixels")
        report[domain] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
        }
    return report


__all__ = [
    "IMAGE_SIZE",
    "ManifestImageDataset",
    "NORMALIZATION_MEAN",
    "NORMALIZATION_STD",
    "PREPROCESSING_NAME",
    "apply_domain",
    "mild_train_augmentation",
    "source_domain_corruption_seed",
    "to_normalized_tensor",
    "to_pretrained_resnet50_tensor",
    "validate_corruptions",
    "write_corruption_grid",
]
