"""Bounded same-environment admission policy; preserves the historical gate.

Uses pinned bundled historical optimizer/model sources for bounded real-data
comparison. The experiment core and its original verification labels stay intact.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compare_reference import historical_optimizer, reference_class, compare_states, compare_tensor
from cap_mech.checkpointing import atomic_json, digest, file_hash, load, restore, source_version
from cap_mech.constants import CAP, STREAMS, SEEDS
from cap_mech.model import MLP300, parameter_groups
from cap_mech.optim import CAPAdam
from cap_mech.seed import seed_everything
from cap_mech.streams import IndexSampler, permutation, task_domain

SOURCE = "06e8a0142325b8a9f4aac62798bb4f0e77f88c8e6ce328c4e3e849b2b471fd53"
PINS = {"reference_optimizer.py": "e5a5bacd3d82347d1c8d1b9d5c54562ce4f7a61d6f4d20e50590051257e9e6fa",
        "reference_model.py": "7078a6a9f344f2a13ee13df38992858e43610adeb6a725b5b0acef6da9a7f2f1"}
COMMANDS = ("m1-faithfulness", "m2-k-dial", "m3-state-causality",
            "m4-importance-assignment", "m5-predict", "m6-release", "analyze")


def environment(device):
    return {"python": sys.version, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def provenance(root, device, data_dir):
    if source_version() != SOURCE:
        raise RuntimeError("Experiment core changed; this validation policy requires the pinned experiment source")
    for name, sha in PINS.items():
        if file_hash(HERE/name) != sha:
            raise RuntimeError(f"External reference source changed: {name}")
    gate_path = root/"reproduction/reproduction_gate.json"
    gate = json.loads(gate_path.read_text())
    config_path = root/"reproduction/run_config.json"
    config = json.loads(config_path.read_text())
    if (gate.get("source_version") != SOURCE or gate.get("baseline_config_sha256") != digest(CAP)
            or not gate.get("valid_reference") or config.get("source_version") != SOURCE
            or config.get("smoke") or config.get("seeds") != list(SEEDS)
            or config.get("streams") != STREAMS or config.get("baseline") != CAP
            or config.get("torch_version") != str(torch.__version__)
            or config.get("device") != device or config.get("threads") != 1):
        raise RuntimeError("Saved reference protocol/source/environment is incompatible")
    required = {f"{stream}_seed{s}_task{t}.pt" for s in SEEDS for stream, tasks in
                (("a", (20, 50, 100, 150, 200)), ("b", (50, 100, 150, 200))) for t in tasks}
    hashes = gate.get("checkpoint_hashes", {})
    if set(hashes) != required:
        raise RuntimeError("Expected all 45 frozen reference checkpoints")
    for name, sha in hashes.items():
        if file_hash(root/"reproduction/checkpoints"/name) != sha:
            raise RuntimeError(f"Checkpoint changed: {name}")
    data_files = {p.name: file_hash(p) for p in sorted((data_dir/"MNIST/raw").glob("*-ubyte"))}
    if len(data_files) != 4:
        raise RuntimeError("Expected four existing MNIST/raw/*-ubyte files; set --data-dir correctly")
    return {"policy": "same-environment-final-implementation-v1",
            "historical_endpoint_passed": gate["passed"], "historical_gate_sha256": file_hash(gate_path),
            "reference_config_sha256": file_hash(config_path), "source_version": SOURCE,
            "external_sources": PINS, "checkpoint_hashes": hashes, "mnist_hashes": data_files,
            "validation_code": {p.name: file_hash(p) for p in (Path(__file__), HERE/"compare_reference.py")},
            "environment": environment(device), "baseline": CAP, "streams": STREAMS,
            "fresh_steps_per_seed_stream": 1000, "steps_per_checkpoint": 16,
            "limitation": "Bounded implementation equivalence, not historical endpoint reproduction or proof of full-run equivalence.",
            "infinity_intervention": "M6 holds bound importance unchanged at T_m=infinity; historical helper infinity special case is not used."}


def historical_model_class():
    spec = importlib.util.spec_from_file_location("_external_model", HERE/"reference_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLP300


def paired(raw, labels, dense, cls, model_cls, seed, stream, device, steps, checkpoint=None):
    seed_everything(seed)
    old_model = model_cls(784, 10).to(device)
    seed_everything(seed)
    new_model = MLP300().to(device)
    old, new = historical_optimizer(cls, old_model), CAPAdam(parameter_groups(new_model))
    sampler = IndexSampler(len(labels), seed, 128)
    task = 1
    if checkpoint is not None:
        if (checkpoint["source_version"] != SOURCE or checkpoint["baseline_config_sha256"] != digest(CAP)
                or checkpoint["within_task_update"] != 0 or checkpoint["stream_metadata"]["smoke"]
                or checkpoint["seed"] != seed or checkpoint["stream"] != stream):
            raise RuntimeError("Invalid checkpoint metadata")
        restore(checkpoint, new_model, new, sampler)
        # Map by identical parameter order, retaining historical optimizer group defaults.
        with torch.no_grad():
            for p, q in zip(old_model.parameters(), new_model.parameters()):
                p.copy_(q)
                old.state[p] = copy.deepcopy(new.state[q])
                old.state[p].setdefault("anchor", p.detach().clone())
                old.state[p].setdefault("imp", torch.zeros_like(p))
        task = min(checkpoint["task"]+1, 200)
    loader = None
    if checkpoint is None:
        # Real DataLoader checks the suite's sampler through two epoch boundaries.
        loader = DataLoader(TensorDataset(torch.arange(len(labels))), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=1, persistent_workers=True)
        iterator = iter(loader)
    perm = permutation(stream, task_domain(stream, task))
    transform = Normalize((.1307,), (.3081,))
    for step in range(1, steps+1):
        idx = sampler.next_indices()
        if loader is not None:
            try:
                historical_idx = next(iterator)[0]
            except StopIteration:
                iterator = iter(loader)
                historical_idx = next(iterator)[0]
            if not torch.equal(idx, historical_idx):
                raise RuntimeError(f"Historical DataLoader indices differ at {stream}/{seed}/{step}")
        legacy_x = torch.stack([transform(im.unsqueeze(0).float()/255) for im in raw[idx]])
        legacy_x = legacy_x.to(device).flatten(1)[:, perm.to(device)]
        x, y = dense[idx][:, perm].to(device), labels[idx].to(device)
        mismatch = compare_tensor(legacy_x, x, "input", step)
        if mismatch:
            raise RuntimeError(str(mismatch))
        old.zero_grad(set_to_none=True)
        new.zero_grad(set_to_none=True)
        old_logits, new_logits = old_model(legacy_x), new_model(x)
        mismatch = compare_tensor(old_logits, new_logits, "logits", step)
        if mismatch:
            raise RuntimeError(str(mismatch))
        F.cross_entropy(old_logits, y).backward()
        F.cross_entropy(new_logits, y).backward()
        for p, q in zip(old_model.parameters(), new_model.parameters()):
            mismatch = compare_tensor(p.grad, q.grad, "gradient", step)
            if mismatch:
                raise RuntimeError(str(mismatch))
        old.step()
        new.step()
        mismatch = compare_states(old_model, new_model, old, new, step)
        if mismatch:
            raise RuntimeError(str(mismatch))
        for p, q in zip(old_model.parameters(), new_model.parameters()):
            if q.ndim >= 2 and not torch.equal(old.state[p]["last_bind"], new.state[q]["last_bind"]):
                raise RuntimeError(f"Binding mask differs at step {step}")
    return {"seed": seed, "stream": stream, "task": task, "steps": steps, "exact_match": True,
            "sampler_checked_against_dataloader": loader is not None}


def validate(args):
    seed_everything(42)  # Record the same backend settings used by every training cell.
    print("Checking source, configuration, dataset and all 45 checkpoint hashes...", flush=True)
    identity = provenance(args.output_dir, args.device, args.data_dir)
    path = args.output_dir/"same_environment_validation.json"
    if path.exists():
        saved = json.loads(path.read_text())
        rows = saved.get("comparisons", [])
        expected_cells = {(s, stream) for s in SEEDS for stream in ("a", "b")}
        fresh = [r for r in rows if "checkpoint" not in r]
        bank = [r for r in rows if "checkpoint" in r]
        complete = (len(fresh) == 10 and len(bank) == 45
                    and {(r["seed"], r["stream"]) for r in fresh} == expected_cells
                    and {r["checkpoint"] for r in bank} == set(identity["checkpoint_hashes"])
                    and all(r.get("exact_match") and r.get("steps") == 1000
                            and r.get("sampler_checked_against_dataloader") for r in fresh)
                    and all(r.get("exact_match") and r.get("steps") == 16 for r in bank))
        if saved.get("passed") and saved.get("identity") == identity and complete:
            print("Reusing matching same-environment validation certificate.", flush=True)
            return saved
        raise RuntimeError("Existing validation does not match; preserve it under another filename before revalidation")
    data = MNIST(args.data_dir, train=True, download=False)
    dense = data.data.flatten(1).float().div_(255).sub_(.1307).div_(.3081)
    cls, model_cls = reference_class(HERE/"reference_optimizer.py"), historical_model_class()
    rows = []
    for stream in ("a", "b"):
        for seed in SEEDS:
            print(f"Validating real MNIST: stream {stream}, seed {seed}, 1000 updates", flush=True)
            rows.append(paired(data.data, data.targets, dense, cls, model_cls, seed, stream, args.device, 1000))
    for name in sorted(identity["checkpoint_hashes"]):
        print(f"Validating continuation: {name}", flush=True)
        cp = load(args.output_dir/"reproduction/checkpoints"/name)
        row = paired(data.data, data.targets, dense, cls, model_cls, cp["seed"], cp["stream"], args.device, 16, cp)
        rows.append(dict(row, checkpoint=name))
    result = {"passed": True, "identity": identity, "comparisons": rows}
    atomic_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate(args)  # Any mismatch prevents every mechanism launch.
    if args.validate_only:
        return
    from cap_mech.cli import main as run
    policy = args.output_dir/"VALIDATION_POLICY.md"
    policy.write_text("# Bounded implementation-validation policy\n\n"
        "Admission uses same_environment_validation.json: pinned historical final contest-blocked CAP, "
        "five seeds, both streams, 1000 real MNIST updates each, plus 16 updates from each of 45 checkpoints.\n\n"
        "Historical endpoint reproduction remains as recorded in reproduction/reproduction_gate.json. "
        "It is not relabeled as passed. The runner uses its explicit unverified-reference option. "
        "Its scientifically_verified=false and conservative claim labels refer to that original endpoint gate; "
        "this separate certificate records bounded implementation agreement.\n\n"
        "These checks do not prove long-run equivalence or validate every intervention. "
        "M6 holds bound importance constant at infinite T_m; the historical helper's special case is not used.\n",
        encoding="utf-8")
    for command in COMMANDS:
        print(f"Starting {command} under documented same-environment policy", flush=True)
        argv = [command, "--output-dir", str(args.output_dir), "--data-dir", str(args.data_dir),
                "--device", args.device, "--threads", "1", "--allow-unverified-reference"]
        if args.resume:
            argv.append("--resume")
        run(argv)


if __name__ == "__main__":
    main()
