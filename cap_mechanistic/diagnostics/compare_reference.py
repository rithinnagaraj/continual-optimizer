"""Read-only, short numerical comparison with the vendored historical optimizer.

This tool does not change the experiment implementation, configuration, results,
or gate. Passing it does not establish historical million-update reproduction.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cap_mech.constants import CAP
from cap_mech.model import MLP300, parameter_groups
from cap_mech.optim import CAPAdam
from cap_mech.seed import seed_everything
from cap_mech.streams import IndexSampler


def reference_class(path):
    spec = importlib.util.spec_from_file_location("_historical_reference_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CAPAdam


def historical_optimizer(cls, model):
    return cls(parameter_groups(model), **{k: CAP[k] for k in
               ("lr", "betas", "eps", "weight_decay", "k", "T_a", "T_s", "T_m")},
               retention="release", protection="box", release_update="contest_blocked",
               box_geometry="coordinate", momentum_response="none", passive=False,
               amsgrad=False, maximize=False, freeze_stats=False)


def compare_tensor(a, b, label, step):
    if torch.equal(a, b):
        return None
    return {"field": label, "step": step, "max_absolute_difference": float((a-b).abs().max()),
            "different_elements": int((a != b).sum())}


def compare_states(old_model, new_model, old, new, step):
    for index, (p, q) in enumerate(zip(old_model.parameters(), new_model.parameters())):
        mismatch = compare_tensor(p, q, f"parameter_{index}", step)
        if mismatch:
            return mismatch
        for key in ("exp_avg", "exp_avg_sq", "anchor", "imp"):
            if key in new.state[q]:
                mismatch = compare_tensor(old.state[p][key], new.state[q][key], f"parameter_{index}.{key}", step)
                if mismatch:
                    return mismatch
        if old.state[p]["step"] != new.state[q]["step"]:
            return {"field": "Adam step", "step": step}
    return None


def optimizer_check(cls, device, steps):
    seed_everything(42)
    first = nn.Linear(17, 9).to(device)
    second = nn.Linear(17, 9).to(device)
    second.load_state_dict(first.state_dict())
    old, new = historical_optimizer(cls, first), CAPAdam(parameter_groups(second))
    generator = torch.Generator().manual_seed(71)
    bound_count = 0
    for step in range(1, steps+1):
        for p, q in zip(first.parameters(), second.parameters()):
            gradient = 10*torch.randn(p.shape, generator=generator).to(device)
            p.grad, q.grad = gradient.clone(), gradient.clone()
        old.step()
        new.step()
        bound_count += sum(int(new.state[p]["last_bind"].sum()) for p in second.parameters() if p.ndim >= 2)
        mismatch = compare_states(first, second, old, new, step)
        if mismatch:
            return {"exact_match": False, "first_mismatch": mismatch, "bound_coordinate_updates": bound_count}
    return {"exact_match": True, "steps": steps, "bound_coordinate_updates": bound_count}


def training_check(cls, device, steps):
    seed_everything(42)
    old_model = nn.Sequential(nn.Flatten(1), nn.Linear(784, 300), nn.ReLU(),
                              nn.Linear(300, 300), nn.ReLU(), nn.Linear(300, 10)).to(device)
    seed_everything(42)
    new_model = MLP300().to(device)
    old, new = historical_optimizer(cls, old_model), CAPAdam(parameter_groups(new_model))
    generator = torch.Generator().manual_seed(72)
    raw = torch.randint(0, 256, (512, 1, 28, 28), generator=generator, dtype=torch.uint8)
    labels = torch.randint(0, 10, (512,), generator=generator)
    dense = raw.flatten(1).float().div_(255).sub_(.1307).div_(.3081)
    permutation = torch.randperm(784, generator=torch.Generator().manual_seed(100001)).to(device)
    sampler = IndexSampler(512, 42, 128)
    for step in range(1, steps+1):
        indices = sampler.next_indices()
        # Preserve both preprocessing routes: per-image torchvision then device
        # permutation (historical), and flattened CPU preprocessing (standalone).
        legacy_input = torch.stack([Normalize((.1307,), (.3081,))(im.float()/255) for im in raw[indices]])
        legacy_input = legacy_input.to(device).view(-1, 784)[:, permutation].view(-1, 1, 28, 28)
        standalone_input = dense[indices][:, permutation.cpu()].to(device)
        mismatch = compare_tensor(legacy_input.flatten(1), standalone_input, "normalized_permuted_input", step)
        if mismatch:
            return {"exact_match": False, "first_mismatch": mismatch}
        old.zero_grad(set_to_none=True)
        new.zero_grad(set_to_none=True)
        old_logits, new_logits = old_model(legacy_input), new_model(standalone_input)
        mismatch = compare_tensor(old_logits, new_logits, "logits", step)
        if mismatch:
            return {"exact_match": False, "first_mismatch": mismatch}
        y = labels[indices].to(device)
        F.cross_entropy(old_logits, y).backward()
        F.cross_entropy(new_logits, y).backward()
        for index, (p, q) in enumerate(zip(old_model.parameters(), new_model.parameters())):
            mismatch = compare_tensor(p.grad, q.grad, f"gradient_{index}", step)
            if mismatch:
                return {"exact_match": False, "first_mismatch": mismatch}
        old.step()
        new.step()
        mismatch = compare_states(old_model, new_model, old, new, step)
        if mismatch:
            return {"exact_match": False, "first_mismatch": mismatch}
    return {"exact_match": True, "steps": steps, "data": "synthetic MNIST-shaped byte images"}


def journal_check(output_dir):
    """Check saved sampler fingerprints without loading or modifying checkpoints."""
    root = output_dir/"reproduction"
    config = json.loads((root/"run_config.json").read_text())
    result = {"recorded_run_config": {k: config.get(k) for k in
              ("torch_version", "device", "threads", "seeds", "source_version", "streams")}, "sampler_checks": []}
    for stream in ("a", "b"):
        file = root/"raw"/f"{stream}_seed42__baseline"/"online.jsonl"
        if not file.exists():
            continue
        samplers = {"persistent": IndexSampler(60000, 42, 128, True),
                    "nonpersistent": IndexSampler(60000, 42, 128, False)}
        first_mismatch = {name: None for name in samplers}
        checked = 0
        with file.open() as handle:
            for line in handle:
                row = json.loads(line)
                for name, sampler in samplers.items():
                    indices = sampler.next_indices().tolist()
                    digest = hashlib.sha256(json.dumps(indices, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                    if digest != row["batch_indices_sha256"] and first_mismatch[name] is None:
                        first_mismatch[name] = row["global_update"]
                checked += 1
                if checked == 1000:
                    break
        result["sampler_checks"].append({"stream": stream, "checked_updates": checked,
                                         "first_mismatch_by_profile": first_mismatch})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, help="Existing scientific output root; read-only inspection")
    parser.add_argument("--legacy-optimizer", type=Path, default=Path(__file__).with_name("reference_optimizer.py"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    cls = reference_class(args.legacy_optimizer)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "device": args.device}
    if str(args.device).startswith("cuda"):
        environment["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    result = {"environment": environment,
              "reference_file_sha256": hashlib.sha256(args.legacy_optimizer.read_bytes()).hexdigest(),
              "baseline": CAP, "finite_baseline_only": True,
              "optimizer_same_gradients": optimizer_check(cls, args.device, args.steps),
              "paired_short_training": training_check(cls, args.device, args.steps),
              "limitation": "This does not identify the historical launch environment or prove long-run reproduction."}
    if args.output_dir:
        result["saved_run"] = journal_check(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
