"""Shared reference and counterfactual runner. No task-boundary optimizer updates."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .checkpointing import (atomic_save, atomic_json, capture, restore, load, digest,
                            file_hash, source_version, prepare_run, gate_payload, state_equal)
from .constants import CAP, STREAMS, MECHANISMS, SEEDS, VERSION
from .data import Data
from .model import make_model, parameter_groups
from . import protocol
from .optim import CAPAdam
from .seed import seed_everything
from .streams import IndexSampler, permutation, task_domain, old_tasks, schedule
from .metrics import evaluate, task_metrics, end_retention
from .probes.state_features import state_features
from .interventions import apply
from .records import Journal, read_table, write_csv

NAMES = {"m1": "m1_importance_faithfulness", "m2": "m2_k_causal_dial",
         "m3": "m3_stream_age_state", "m4": "m4_importance_assignment"}

def manifest(args, experiment, sources=None):
    return {"version": VERSION, "experiment": experiment, "baseline": CAP,
            "protocol": protocol.specification(), "streams": STREAMS, "mechanisms": MECHANISMS, "smoke": args.smoke,
            "seeds": list(args.seeds), "device": args.device,
            "torch_version": torch.__version__, "threads": getattr(args, "threads", 1), "source_version": source_version(),
            "checkpoint_hashes": sources or {},
            "allow_unverified_reference": args.allow_unverified_reference}

def bank_path(root, stream, seed, task):
    return Path(root)/"reproduction"/"checkpoints"/f"{stream}_seed{seed}_task{task}.pt"

def provenance(source_hash, config):
    return {"source_checkpoint_sha256": source_hash, "baseline_config_sha256": digest(CAP),
            "experiment_config_sha256": digest(config), "source_version": source_version()}

def new_state(seed, data, device, smoke):
    seed_everything(seed)
    model = make_model().to(device)
    optimizer = CAPAdam(parameter_groups(model))
    sampler = IndexSampler(len(data.train_y), seed, 8 if smoke else 128)
    return model, optimizer, sampler

def train_cell(args, data, run_dir, config, stream, seed, cell, source=None,
               intervention=None, horizon=None, experiment="reference"):
    """Commit logs and all within-task accumulators every 1000 updates.

    A resumed cell trims each journal to the last committed row count. Frozen
    bank checkpoints contain enough state to branch without prefix retraining.
    """
    cell_dir = Path(run_dir)/"raw"/cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    progress = cell_dir/"resume.pt"
    model, optimizer, sampler = new_state(seed, data, args.device, args.smoke)
    source_cp = load(source) if source else None
    source_hash = file_hash(source) if source else None
    base_task = source_cp["task"] if source_cp else 0
    updates = 3 if args.smoke else STREAMS["updates_per_task"]
    tasks = (base_task+(horizon or 0)) if source else (4 if args.smoke else 200)
    if tasks > 200:
        raise ValueError("Continuation exceeds reference schedule")
    if source_cp and (source_cp["source_version"] != source_version() or
                      source_cp["baseline_config_sha256"] != digest(CAP) or
                      source_cp.get("protocol") != protocol.specification()):
        raise ValueError("Source checkpoint code/config changed")
    if source_cp and (source_cp["stream"] != stream or source_cp["seed"] != seed or
                      source_cp["within_task_update"] != 0 or source_cp["stream_metadata"]["smoke"] != args.smoke):
        raise ValueError("Source checkpoint stream/seed/boundary/smoke mismatch")
    meta = {"seed": seed, "stream": stream, "checkpoint_task": base_task,
            "branch": cell.split("__")[-1], "branch_id": cell,
            "intervention": intervention or {"kind": "baseline"}}
    task, within, step = base_task, 0, base_task*updates
    accs, binds, previous_accuracy = [], deque(maxlen=500), 0.
    before, features, last_seen = {}, {}, {}
    task_bind_sum = 0.
    if progress.exists():
        if not args.resume:
            raise FileExistsError(progress)
        cp = load(progress)
        if cp["config_hash"] != digest(config) or cp.get("input_checkpoint_hash") != source_hash:
            raise ValueError("Resume configuration/source checkpoint mismatch")
        restore(cp, model, optimizer, sampler)
        task, within, step = cp["task"], cp["within_task_update"], cp["global_update"]
        accs, binds = cp["accuracies"], deque(cp["trailing_binds"], maxlen=500)
        before, features = cp["before"], cp["features"]
        previous_accuracy, last_seen = cp["previous_accuracy"], cp["last_seen"]
        task_bind_sum = cp["task_bind_sum"]
        prov = cp["provenance"]
        journal = Journal(cell_dir, cp["row_counts"])
        if cp["complete"]:
            return cp
    else:
        if source_cp:
            restore(source_cp, model, optimizer, sampler)
            binds = deque(source_cp.get("trailing_binds", []), maxlen=500)
            previous_accuracy = source_cp.get("previous_accuracy", 0.)
            last_seen = source_cp.get("last_seen", {})
            # Exact prediction invariance is asserted on a fixed held-out batch.
            with torch.no_grad():
                logits = model(data.test_x[:8].to(args.device)).clone()
            apply(optimizer, intervention)
            with torch.no_grad():
                if not torch.equal(logits, model(data.test_x[:8].to(args.device))):
                    raise AssertionError("Intervention changed initial predictions")
        else:
            initial = cell_dir/"initial.pt"
            if not initial.exists():
                atomic_save(initial, capture(model, optimizer, sampler, seed=seed))
            else:
                restore(load(initial), model, optimizer, sampler)
            source_hash_initial = file_hash(initial)
        prov = provenance(source_hash if source else source_hash_initial, config)
        journal = Journal(cell_dir)
    domains = old_tasks(base_task) if stream == "a" and source else list(range(STREAMS["pool_size"])) if stream == "b" else []
    common = {**prov, **meta}

    def add(table, row):
        journal.add(table, {**common, **row})

    def retention(local, task_now):
        ev = evaluate(model, data, stream, domains, args.device)
        current = task_domain(stream, task_now)
        for r in ev:
            d = r["domain"]
            add("retention", {**r, "task": task_now, "within_task_update": local,
                              "before_accuracy": before[d], "forgetting": before[d]-r["accuracy"],
                              "is_current": d == current,
                              "revisit_gap": task_now-last_seen[d] if d in last_seen else None})
        return ev

    if source and not before:
        before = {r["domain"]: r["accuracy"] for r in evaluate(model, data, stream, domains, args.device)}
        for d, a in before.items():
            add("retention", {"domain": d, "accuracy": a, "before_accuracy": a,
                              "forgetting": 0., "task": base_task, "within_task_update": 0,
                              "is_current": d == task_domain(stream, base_task), "revisit_gap": None})

    def checkpoint(complete=False):
        return capture(model, optimizer, sampler, **common, protocol=protocol.specification(), task=task,
                       within_task_update=within, global_update=step, config_hash=digest(config),
                       input_checkpoint_hash=source_hash, row_counts=dict(journal.counts),
                       accuracies=list(accs), trailing_binds=list(binds), before=before,
                       features=features, previous_accuracy=previous_accuracy, last_seen=last_seen,
                       task_bind_sum=task_bind_sum,
                       provenance=prov, complete=complete, stream_metadata={"schedule": schedule().tolist(),
                       "old_probe_tasks": domains, "test_indices": data.test_indices.tolist(),
                       "updates_per_task": updates, "smoke": args.smoke})

    # Commit the initial intervention and initial retention before training.
    atomic_save(progress, checkpoint())
    while task < tasks or within:
        active_task = task+1
        if within == 0:
            features = state_features(optimizer, float(np.mean(binds)) if binds else 0., previous_accuracy)
        perm = permutation(stream, task_domain(stream, active_task))
        indices = sampler.next_indices()
        x, y = data.batch(indices, perm, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        accuracy = (logits.argmax(1) == y).float().mean().item()
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {cell} at {step}")
        loss.backward()
        optimizer.step()
        within += 1
        step += 1
        accs.append(accuracy)
        binds.append(optimizer.last_bind_rate)
        task_bind_sum += optimizer.last_bind_rate
        add("online", {"task": active_task, "within_task_update": within, "global_update": step,
                       "accuracy": accuracy, "loss": loss.item(), "bind_rate": optimizer.last_bind_rate,
                       "batch_indices_sha256": digest(indices.tolist())})
        if within in MECHANISMS["curve_steps"] or args.smoke:
            add("learning_curves", {"task": active_task, "within_task_update": within,
                                    "accuracy": accuracy})
        if step % (1 if args.smoke else MECHANISMS["diagnostic_interval"]) == 0:
            add("mechanism_traces", {"task": active_task, "within_task_update": within,
                                    "global_update": step, **state_features(optimizer),
                                    "bind_rate": optimizer.last_bind_rate,
                                    "bound_importance_decay": optimizer.last_bound_importance_decay})
        if source and experiment == "m3" and within == min(1000, updates-1):
            retention(within, active_task)
        if within == updates:
            domain = task_domain(stream, active_task)
            heldout = evaluate(model, data, stream, [domain], args.device)[0]["accuracy"]
            result = {"task": active_task, **task_metrics(accs, updates), "task_end_accuracy": heldout,
                      "bind_rate_mean": task_bind_sum/updates,
                      "historical_sampled_plasticity": float(np.mean(accs[(0 if args.smoke else 99)::(1 if args.smoke else 100)]))}
            add("tasks", result)
            if stream == "a" and not source and active_task >= 2:
                add("task_features", {**features, **result})
            if source:
                retention(within, active_task)
            if stream == "b":
                for r in evaluate(model, data, stream, range(STREAMS["pool_size"]), args.device):
                    d = r["domain"]
                    add("pool", {**r, "task": active_task, "is_current": d == domain,
                                 "revisit_gap": active_task-last_seen[d] if d in last_seen else None})
                last_seen[domain] = active_task
            previous_accuracy = result["plasticity"]
            task, within, accs = active_task, 0, []
            task_bind_sum = 0.
            bank_tasks = ([1, 2, 3, 4] if args.smoke else STREAMS[f"{stream}_checkpoints"])
            if not source and task in bank_tasks:
                path = bank_path(args.output_dir, stream, seed, task)
                if not path.exists():
                    atomic_save(path, checkpoint())
                else:
                    existing = load(path)
                    current = checkpoint()
                    if any(not state_equal(existing[k], current[k]) for k in ("model", "optimizer", "sampler", "global_update", "config_hash")):
                        raise RuntimeError(f"Immutable bank checkpoint disagrees with replay: {path}")
            print(f"{cell}: task {task}/{tasks}, online={previous_accuracy:.5f}", flush=True)
        if step % (1 if args.smoke else MECHANISMS["checkpoint_interval"]) == 0 or within == 0:
            atomic_save(progress, checkpoint(complete=task == tasks and within == 0))
        if task == tasks and within == 0:
            break
    return load(progress)

def reproduce(args):
    root = Path(args.output_dir)/"reproduction"
    config = manifest(args, "reference")
    prepare_run(root, config, args.resume)
    data = Data(args.data_dir, args.smoke)
    for stream in ("a", "b"):
        for seed in args.seeds:
            train_cell(args, data, root, config, stream, seed, f"{stream}_seed{seed}__baseline")
    tasks = read_table(root, "tasks")
    pool = read_table(root, "pool")
    observed = {}
    for seed in args.seeds:
        a = tasks[(tasks.seed == seed) & (tasks.stream == "a") & (tasks.task > (2 if args.smoke else 100))]
        b = pool[pool.seed == seed]
        observed[seed] = {"P_sampled_100step": float(a.historical_sampled_plasticity.mean()),
                          "P_dense_all_updates": float(a.plasticity.mean()),
                          "R": end_retention(b.to_dict("records"), 4 if args.smoke else 200)}
    hashes = {p.name: file_hash(p) for p in sorted((root/"checkpoints").glob("*.pt"))}
    gate = gate_payload(observed, not args.smoke, source_version=source_version(), checkpoint_hashes=hashes, protocol=protocol.specification())
    gate["P_dense_all_updates"] = {str(s): observed[s]["P_dense_all_updates"] for s in args.seeds}
    gate["P_definition"] = "P_sampled_100step: updates 500100, 500200, ..., 1000000; sampled descriptive metric; no historical CIFAR target"
    gate["dense_P_definition"] = "P_dense_all_updates: updates 500001 through 1000000; descriptive CIFAR metric"
    atomic_json(root/"reproduction_gate.json", gate)
    for table in ("tasks", "pool", "task_features"):
        write_csv(root/"summary"/f"{table}.csv", read_table(root, table))
    # Separate stream exports keep reference history directly reconstructible.
    for stream in ("a", "b"):
        for table in ("tasks", "pool"):
            frame = read_table(root, table)
            if not frame.empty:
                write_csv(root/f"stream_{stream}"/f"{table}.csv", frame[frame.stream == stream])
    if not gate["passed"] and not args.smoke:
        raise RuntimeError("Incomplete CIFAR reference: STOP. Inspect reproduction_gate.json")
    return gate

def branches(experiment, seed, task):
    if experiment == "m2":
        return [(f"k_{m:g}", {"kind": "k", "value": CAP["k"]*m}) for m in MECHANISMS["k_multipliers"]] + [("box_off", {"kind": "box_off"})]
    if experiment == "m3":
        return [(b, {"kind": b}) for b in MECHANISMS["reset_branches"]]
    if experiment == "m4":
        return [("true_importance", {"kind": "true_importance"}), ("reverse", {"kind": "reverse"})] + [
            (f"shuffle_{i}", {"kind": "shuffle", "seed": MECHANISMS["shuffle_seed"]+seed*10000+task*10+i})
            for i in range(MECHANISMS["shuffle_replicates"])]
    raise ValueError(experiment)

def causal(args, experiment):
    from .checkpointing import require_gate
    require_gate(args.output_dir, args.allow_unverified_reference)
    stream = "a" if experiment in ("m2", "m3") else "b"
    tasks = ([2] if args.smoke else MECHANISMS["a_checkpoints" if stream == "a" else "b_checkpoints"])
    sources = {f"{seed}_{task}": bank_path(args.output_dir, stream, seed, task)
               for seed in args.seeds for task in tasks}
    config = manifest(args, experiment, {k: file_hash(v) for k, v in sources.items()})
    root = Path(args.output_dir)/NAMES[experiment]
    prepare_run(root, config, args.resume)
    data = Data(args.data_dir, args.smoke)
    for seed in args.seeds:
        for task in tasks:
            for name, intervention in branches(experiment, seed, task):
                train_cell(args, data, root, config, stream, seed, f"seed{seed}_task{task}__{name}",
                           sources[f"{seed}_{task}"], intervention,
                           1 if args.smoke else MECHANISMS["horizons"][experiment], experiment)
    from .analysis import summarize_causal
    summarize_causal(root, experiment)
    atomic_json(root/"complete.json", {"config_hash": digest(config), "scientifically_verified": not args.allow_unverified_reference and not args.smoke})
