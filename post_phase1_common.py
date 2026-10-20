"""Shared invariants for the preregistered Post-Phase-1 retention campaign."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POST_DIR = RESULTS / "post_phase1_v2"
FREEZE_RECORD = POST_DIR / "freeze_record.json"
WINNER_ARTIFACT = POST_DIR / "post_phase1_retention_configs.json"
GRID_DIR = POST_DIR / "retention_grids"
GRID_RESULTS = GRID_DIR / "grid_results.csv"
GRID_TASKS = GRID_DIR / "post_grid_tasks.csv"
LIVENESS_RECORD = POST_DIR / "post_knob_liveness.json"
UPGD_CORRECTED_DIR = POST_DIR / "upgd_corrected"
UPGD_CORRECTED_STREAM_TASKS = UPGD_CORRECTED_DIR / "post_streamb_tasks.csv"
POOL_PREFLIGHT_DIR = POST_DIR / "preflight_pool"
POOL_DECISION = POST_DIR / "pool_size_decision.json"
V2_ISOLATION = (
    "PP1-A1(C): no Phase-1 or v1 result enters v2 computation; "
    "all selected grid and final-stream values are produced fresh by v2"
)

METHODS = (
    "adamw",
    "adam_replay",
    "upgd",
    "ewc_oracle",
    "cap_release",
    "cap_ratchet",
)
SEEDS = (42, 123, 2026, 451, 789)
EXPECTED_GRID_COUNTS = {
    "adamw": 6,
    "adam_replay": 12,
    "upgd": 12,
    "ewc_oracle": 12,
    "cap_release": 48,
    "cap_ratchet": 24,
}


def retention_grids() -> dict[str, list[dict[str, Any]]]:
    shared = tuple(itertools.product((1e-4, 3e-4, 1e-3), (0.0, 1e-2)))
    grids = {
        "adamw": [
            {"lr": lr, "weight_decay": wd} for lr, wd in shared
        ],
        "adam_replay": [
            {
                "lr": lr,
                "weight_decay": wd,
                "replay_count": count,
                "buffer_capacity": 5_000,
            }
            for (lr, wd), count in itertools.product(shared, (32, 96))
        ],
        "upgd": [
            {
                "lr": lr,
                "weight_decay": wd,
                "sigma": sigma,
                "beta_utility": 0.9999,
            }
            for (lr, wd), sigma in itertools.product(shared, (1e-4, 1e-2))
        ],
        "ewc_oracle": [
            {
                "lambda": lam,
                "gamma": 0.95,
                "weight_decay": wd,
                "lr": lr,
            }
            for (lr, wd), lam in itertools.product(shared, (1e3, 1e5))
        ],
        "cap_release": [
            {
                "k": k,
                "T_a": t_a,
                "T_s": 2_000,
                "T_m": t_m,
                "lr": lr,
                "weight_decay": wd,
                "retention": "release",
            }
            for (lr, wd), k, t_a, t_m in itertools.product(
                shared, (1e-5, 7e-5), (400, 2_000), (10_000, 200_000)
            )
        ],
        "cap_ratchet": [
            {
                "k": k,
                "T_a": t_a,
                "T_s": 2_000,
                "T_m": 10_000,
                "lr": lr,
                "weight_decay": wd,
                "retention": "ratchet",
            }
            for (lr, wd), k, t_a in itertools.product(
                shared, (1e-5, 7e-5), (400, 2_000)
            )
        ],
    }
    observed = {method: len(cells) for method, cells in grids.items()}
    if observed != EXPECTED_GRID_COUNTS or sum(observed.values()) != 114:
        raise RuntimeError(f"retention grid changed: {observed}")
    return grids


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "MANIFEST.sha256"
    }


def render_manifest(root: Path) -> str:
    return "".join(
        f"{digest}  {relative}\n"
        for relative, digest in _manifest_entries(root).items()
    )


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64 or relative in entries:
            raise ValueError(f"invalid manifest line {number}: {path}")
        entries[relative] = digest
    return entries


def verify_manifest(root: Path, *, required: bool) -> dict[str, Any]:
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        if required:
            raise FileNotFoundError(f"required frozen manifest is missing: {manifest}")
        return {"root": str(root), "status": "NOT-YET-STAGED", "differences": []}
    expected = parse_manifest(manifest)
    observed = _manifest_entries(root)
    differences = sorted(
        relative
        for relative in set(expected) | set(observed)
        if expected.get(relative) != observed.get(relative)
    )
    if differences:
        print("FROZEN-MANIFEST-MISMATCH")
        for relative in differences:
            print(relative)
        raise RuntimeError(
            f"frozen manifest mismatch under {root}: {', '.join(differences)}"
        )
    return {
        "root": str(root),
        "status": "VERIFIED",
        "manifest_sha256": sha256_file(manifest),
        "file_count": len(observed),
        "differences": [],
    }


def assert_post_output(path: Path) -> Path:
    resolved = path.resolve()
    post_root = POST_DIR.resolve()
    if resolved != post_root and post_root not in resolved.parents:
        raise ValueError(f"Post-Phase-1 output must be under {POST_DIR}: {path}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_post_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def update_freeze_record(section: str, payload: Mapping[str, Any]) -> None:
    record = {"schema_version": 3, "isolation": V2_ISOLATION}
    record[section] = dict(payload)
    atomic_write_json(FREEZE_RECORD, record)


@contextmanager
def post_guard() -> Iterator[dict[str, str]]:
    """Protect immutable v2 decisions without reading any older result tree."""
    protected = {
        str(path): sha256_file(path)
        for path in (POOL_DECISION, WINNER_ARTIFACT)
        if path.exists()
    }
    try:
        yield protected
    finally:
        for raw_path, expected in protected.items():
            path = Path(raw_path)
            if not path.exists() or sha256_file(path) != expected:
                raise RuntimeError(f"immutable v2 artifact changed: {path}")


def load_winner_artifact() -> dict[str, Any]:
    artifact = load_json(WINNER_ARTIFACT)
    record = load_json(FREEZE_RECORD)
    if record.get("isolation") != V2_ISOLATION:
        raise RuntimeError("winner freeze record is not self-contained v2")
    expected = record.get("winner_artifact", {}).get("sha256")
    observed = sha256_file(WINNER_ARTIFACT)
    if not expected or expected != observed:
        raise RuntimeError("winner artifact is absent from freeze_record or changed")
    if set(artifact) != set(METHODS):
        raise RuntimeError("winner artifact has the wrong method roster")
    if any(
        not isinstance(artifact.get(method), Mapping)
        or artifact[method].get("isolation") != V2_ISOLATION
        for method in METHODS
    ):
        raise RuntimeError(
            "winner artifact predates PP1-A1(C) isolation or is malformed"
        )
    return artifact


def post_arm_config(
    method: str, config: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Return the Phase-1 constructor name and a minimal direct artifact."""
    if method == "cap_ratchet":
        return "cap_ratchet", {"derived_configs": {"cap_ratchet": dict(config)}}
    return method, {"methods": {method: {"config": dict(config)}}}


def post_deployment_artifact(
    method: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    constructor_method = (
        "cap_release" if method in {"cap_release", "cap_ratchet"} else method
    )
    return {"methods": {constructor_method: {"config": dict(config)}}}


def parameter_vector(model: Any) -> Any:
    import torch

    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def grid_knobs(method: str) -> tuple[str, ...]:
    return {
        "adamw": ("lr", "weight_decay"),
        "adam_replay": ("lr", "weight_decay", "replay_count"),
        "upgd": ("lr", "weight_decay", "sigma"),
        "ewc_oracle": ("lr", "weight_decay", "lambda"),
        "cap_release": ("lr", "weight_decay", "k", "T_a", "T_m"),
        "cap_ratchet": ("lr", "weight_decay", "k", "T_a"),
    }[method]


def values_for_knob(method: str, knob: str) -> list[Any]:
    seen: list[Any] = []
    for config in retention_grids()[method]:
        value = config[knob]
        if value not in seen:
            seen.append(value)
    return seen


def first_config(method: str) -> dict[str, Any]:
    return dict(retention_grids()[method][0])


def normalize_config(method: str, config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if method in METHODS:
        normalized.setdefault("weight_decay", 0.01)
    if method == "adam_replay":
        normalized.setdefault("buffer_capacity", 5_000)
    if method == "ewc_oracle":
        normalized.setdefault("lr", 1e-3)
    if method in {"cap_release", "cap_ratchet"}:
        normalized.setdefault("T_s", 2_000)
        normalized.setdefault("T_m", 10_000)
        normalized.setdefault("lr", 1e-3)
        normalized.setdefault(
            "retention", "ratchet" if method == "cap_ratchet" else "release"
        )
    return normalized


def complete_task200(rows: Sequence[Mapping[str, Any]], arm: str, seed: int) -> bool:
    return any(
        str(row.get("arm")) == arm
        and int(float(row.get("seed", -1))) == int(seed)
        and int(float(row.get("task", -1))) == 200
        for row in rows
    )


def block_stream_b_schedule(pool_size: int) -> list[int]:
    """The PP1-A1 Stream-B schedule: exact visits in shuffled blocks."""
    if pool_size not in (20, 40) or 200 % pool_size:
        raise ValueError("v2 Stream-B pool size must be 20 or 40")
    import torch

    generator = torch.Generator().manual_seed(555)
    schedule: list[int] = []
    for _ in range(200 // pool_size):
        schedule.extend(
            int(value) for value in torch.randperm(pool_size, generator=generator)
        )
    return schedule


def validation_protocol() -> dict[str, Any]:
    decision = load_pool_decision()
    pool_size = int(decision["chosen_pool_size"]) // 2
    import torch

    generator = torch.Generator().manual_seed(556)
    schedule: list[int] = []
    for _ in range(2):
        schedule.extend(
            int(value) for value in torch.randperm(pool_size, generator=generator)
        )
    return {
        "stream_pool_size": int(decision["chosen_pool_size"]),
        "pool_size": pool_size,
        "task_count": 2 * pool_size,
        "steps_per_task": 5_000,
        "total_steps": 10_000 * pool_size,
        "pool_seed_base": 300_001,
        "schedule_seed": 556,
        "schedule": schedule,
        "final_block_start_task": pool_size + 1,
    }


def load_pool_decision() -> dict[str, Any]:
    if not POOL_DECISION.exists():
        raise RuntimeError("pool-headroom preflight has not produced a decision")
    payload = load_json(POOL_DECISION)
    if payload.get("isolation") != V2_ISOLATION:
        raise RuntimeError(
            "pre-isolation pool-size decision cannot be used by PP1-A1(C)"
        )
    if payload.get("status") == "POOL-HEADROOM-ABSENT":
        raise RuntimeError("POOL-HEADROOM-ABSENT: v2 must stop on this axis")
    chosen = payload.get("chosen_pool_size")
    if payload.get("status") != "SELECTED" or chosen not in (20, 40):
        raise RuntimeError("invalid immutable pool-size decision")
    return payload
