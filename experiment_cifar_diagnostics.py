"""Priority 4: authenticated, resumable matched CIFAR update diagnostics."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from cap_mechanistic_cifar.cifar_mech import protocol
from cap_mechanistic_cifar.cifar_mech.checkpointing import (
    atomic_json, atomic_save, capture, restore, load, file_hash, digest, source_version)
from cap_mechanistic_cifar.cifar_mech.data import Data
from cap_mechanistic_cifar.cifar_mech.model import make_model, parameter_groups
from cap_mechanistic_cifar.cifar_mech.seed import seed_everything
from cap_mechanistic_cifar.cifar_mech.streams import IndexSampler, permutation, old_tasks
from cap_mechanistic_cifar.cifar_mech.metrics import evaluate, task_metrics
from cap_mechanistic_cifar.cifar_mech.interventions import apply
from cifar_update_probe import InstrumentedCAP
from upgd_rebuttal import file_lock
from upgd_rebuttal_runtime import environment

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 451, 789, 2026]
AGES = [20, 50, 100, 150]
BRANCHES = ["baseline", "box_off", "importance_reset", "recenter_only"]


def reference():
    return json.loads((ROOT/"docs/priority4_reference.json").read_text())["architectures"]


def identity():
    files = ["experiment_cifar_diagnostics.py", "cifar_update_probe.py", "docs/priority4_reference.json",
             "upgd_rebuttal.py", "upgd_rebuttal_runtime.py"]
    return {"files": {f: file_hash(ROOT/f) for f in files}, "frozen_cifar": source_version(),
            "seeds": SEEDS, "ages": AGES, "branches": BRANCHES, "updates": 5000,
            "diagnostic_steps": "1..20 and every 100", "version": 1}


def jobs():
    return [(a, s, age, b) for a in ["mlp", "cnn"] for s in SEEDS for age in AGES for b in BRANCHES]


def job_id(j):
    return "-".join(map(str, j))


def initialize(args):
    refs = reference()
    for arch in refs:
        bank = args.bank_root/arch/"scientific/reproduction"
        for name, pinned in refs[arch].items():
            assert file_hash(bank/name) == pinned["sha256"], (arch, name)
        gate = refs[arch]["reproduction_gate.json"]["content"]
        assert gate["passed"] and gate["valid_reference"] and len(gate["checkpoint_hashes"]) == 45
        for name, sha in gate["checkpoint_hashes"].items():
            assert file_hash(bank/"checkpoints"/name) == sha, name
        print(arch, "45 checkpoint hashes authenticated", flush=True)
    cfg = {"identity": identity(), "environment": environment(torch.device(args.device))}
    target = args.output/"campaign.json"
    if target.exists():
        assert json.loads(target.read_text()) == cfg, "Campaign changed"
    else:
        atomic_json(target, cfg)


def run_job(args, data, j, campaign):
    arch, seed, age, branch = j
    name = job_id(j)
    directory = args.output/"jobs"/name
    directory.mkdir(parents=True, exist_ok=True)
    progress = directory/"resume.pt"
    protocol.configure(arch, 4)
    seed_everything(seed)
    pinned = reference()[arch]
    config = pinned["run_config.json"]["content"]
    cpname = f"a_seed{seed}_task{age}.pt"
    sha = pinned["reproduction_gate.json"]["content"]["checkpoint_hashes"][cpname]
    source = args.bank_root/arch/"scientific/reproduction/checkpoints"/cpname
    assert file_hash(source) == sha
    model = make_model().to(args.device)
    opt = InstrumentedCAP(parameter_groups(model), **config["baseline"])
    sampler = IndexSampler(len(data.train_y), seed)
    key = {"campaign": digest(campaign), "job": list(j), "source_sha256": sha}
    if progress.exists():
        if not args.resume:
            raise FileExistsError("Use --resume")
        cp = load(progress)
        assert cp["key"] == key
        restore(cp, model, opt, sampler)
        state = cp["record"]
    else:
        cp = load(source)
        assert cp["task"] == age and cp["seed"] == seed and cp["stream"] == "a"
        assert cp["within_task_update"] == 0 and not cp["stream_metadata"]["smoke"]
        assert cp["source_version"] == config["source_version"] and cp["protocol"] == config["protocol"]
        assert cp["baseline_config_sha256"] == digest(config["baseline"])
        restore(cp, model, opt, sampler)
        initial = {n: p.clone() for n, p in model.named_parameters()}
        apply(opt, {"kind": branch})
        assert all(torch.equal(initial[n], p) for n, p in model.named_parameters())
        state = {"step": 0, "accuracies": [], "online": [], "traces": [], "retention": {},
                 "elapsed_seconds": 0., "complete": False}
    started = time.perf_counter()
    previous_elapsed = state["elapsed_seconds"]

    def save():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        state["elapsed_seconds"] = previous_elapsed + time.perf_counter()-started
        atomic_save(progress, capture(model, opt, sampler, key=key, record=state))
        atomic_json(directory/"status.json", {"key": key, "step": state["step"],
                    "complete": state["complete"], "elapsed_seconds": state["elapsed_seconds"]})

    if state["complete"]:
        if not (directory/"result.json").exists():
            atomic_json(directory/"result.json", {"key": key, **state})
        return
    domains = old_tasks(age)
    if "0" not in state["retention"]:
        state["retention"]["0"] = evaluate(model, data, "a", domains, args.device)
        save()
    perm = permutation("a", age+1)
    target = min(5000, state["step"] + args.stop_after_updates) if args.stop_after_updates else 5000
    while state["step"] < target:
        step = state["step"]+1
        indices = sampler.next_indices()
        x, y = data.batch(indices, perm, args.device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(name)
        acc = (logits.argmax(1) == y).float().mean().item()
        loss.backward()
        opt.measure = step <= 20 or step % 100 == 0
        opt.step()
        state["step"] = step
        state["accuracies"].append(acc)
        state["online"].append({"step": step, "loss": loss.item(), "accuracy": acc,
                                "bind_rate": opt.last_bind_rate, "batch_sha256": digest(indices.tolist())})
        state["traces"].extend({"step": step, **r} for r in opt.measurements)
        if step in [1000, 5000]:
            state["retention"][str(step)] = evaluate(model, data, "a", domains, args.device)
        if step == 5000:
            state["current_test"] = evaluate(model, data, "a", [age+1], args.device)
            state["metrics"] = task_metrics(state["accuracies"])
            state["complete"] = True
        if step % 1000 == 0 or step == target:
            save()
            print(name, step, f'{state["elapsed_seconds"]:.1f}s', flush=True)
    if state["complete"]:
        atomic_json(directory/"result.json", {"key": key, **state})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "status"])
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cifar_diagnostics"))
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--job-ids", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    if args.stop_after_updates is not None and args.stop_after_updates <= 0:
        parser.error("Bound must be positive")
    torch.set_num_threads(1)
    seed_everything(42)
    args.output.mkdir(parents=True, exist_ok=True)
    with file_lock(args.output/"campaign.lock"):
        if args.command == "init":
            initialize(args)
            return
        campaign = json.loads((args.output/"campaign.json").read_text())
        assert campaign["identity"] == identity(), "Code changed since initialization"
        selected = jobs()
        if args.command == "status":
            print(json.dumps({job_id(j): json.loads((args.output/"jobs"/job_id(j)/"status.json").read_text())
                              for j in selected if (args.output/"jobs"/job_id(j)/"status.json").exists()}, indent=2))
            return
        assert campaign["environment"] == environment(torch.device(args.device)), "Environment changed"
        if args.job_ids:
            assert set(args.job_ids) <= {job_id(j) for j in selected}, "Unknown job"
            selected = [j for j in selected if job_id(j) in args.job_ids]
        elif not args.all:
            parser.error("Specify --job-ids or --all")
        data = Data(args.data)
        for j in selected:
            run_job(args, data, j, campaign)


if __name__ == "__main__":
    main()
