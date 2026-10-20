import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from .constants import CAP, VERSION, SEEDS
from . import protocol
import math
from .seed import rng_state, restore_rng

def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def source_version():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted([*root.rglob("*.py"), *root.rglob("*.json")]):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)

def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)

def load(path):
    # Only suite-generated trusted checkpoints may be loaded (contains RNG state).
    return torch.load(path, map_location="cpu", weights_only=False)

def capture(model, optimizer, sampler, **metadata):
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "sampler": copy.deepcopy(sampler.state_dict()), "rng": rng_state(), **metadata}

def restore(checkpoint, model, optimizer, sampler):
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    sampler.load_state_dict(checkpoint["sampler"])
    restore_rng(checkpoint["rng"])

def prepare_run(path, config, resume=False):
    path = Path(path)
    manifest = path / "run_config.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"Existing run: {path}; use --resume or a new output directory")
        if json.loads(manifest.read_text()) != config:
            raise ValueError(f"Configuration/source hash changed: {path}; use a new version/output directory")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Refusing untracked existing output directory: {path}")
        atomic_json(manifest, config)
    for directory in ("raw", "summary"):
        (path / directory).mkdir(exist_ok=True)

def gate_payload(observed, valid_reference=True, **metadata):
    rows = [{"seed": seed, "metric": metric, "observed": observed.get(seed, {}).get(metric)}
            for seed in SEEDS for metric in ("P_sampled_100step", "R")]
    passed = valid_reference and all(r["observed"] is not None and math.isfinite(r["observed"])
                                     and 0 <= r["observed"] <= 1 for r in rows)
    return {"version": VERSION, "baseline_config_sha256": digest(CAP), "passed": bool(passed),
            "valid_reference": valid_reference, "rows": rows,
            "validation_scope": "reference completeness and integrity only; not historical reproduction",
            **metadata}

def require_gate(root, allow=False):
    path = Path(root)/"reproduction"/"reproduction_gate.json"
    if allow:
        return False
    if not path.exists():
        raise RuntimeError("Missing reproduction gate; run reproduce first or explicitly pass --allow-unverified-reference")
    gate = json.loads(path.read_text())
    observed = {}
    for r in gate.get("rows", []):
        observed.setdefault(r["seed"], {})[r["metric"]] = r["observed"]
    if (not gate.get("passed") or not gate_payload(observed, gate.get("valid_reference", False))["passed"]
        or gate.get("baseline_config_sha256") != digest(CAP)
        or gate.get("source_version") != source_version()
        or gate.get("protocol") != protocol.specification()):
        raise RuntimeError("CIFAR reference incomplete, wrong protocol, or source changed; mechanisms and analysis are blocked")
    required = {f"{stream}_seed{s}_task{t}.pt" for s in SEEDS for stream, tasks in
                (("a", (20, 50, 100, 150, 200)), ("b", (50, 100, 150, 200))) for t in tasks}
    if set(gate.get("checkpoint_hashes", {})) != required:
        raise RuntimeError("Reference checkpoint bank is incomplete or contains unexpected entries")
    for name, expected in gate["checkpoint_hashes"].items():
        p = Path(root)/"reproduction"/"checkpoints"/name
        if not p.exists() or file_hash(p) != expected:
            raise RuntimeError(f"Reference checkpoint changed: {name}")
    return True

def state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right
