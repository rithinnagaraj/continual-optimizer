"""Numerical worker; imported only by an explicit run command.

Each job owns its checkpoint, logs and OS lock. Data order is restored directly
from the existing exact sampler, not by replaying a million-update loader.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import math
import os
from pathlib import Path
import platform
import time
import uuid

import torch
from torch.nn import functional as F

import upgd_rebuttal as protocol
from model import MLP300
from cap_mechanistic.cap_mech.streams import IndexSampler
from cap_mechanistic.cap_mech.seed import seed_everything, rng_state, restore_rng

SCALAR_FIELDS = ["step", "task", "accuracy", "loss"]
TASK_FIELDS = ["task", "current_domain", "online_accuracy", "task_end_accuracy"]
POOL_FIELDS = ["task", "domain", "accuracy", "is_current"]


class NumericalDivergence(RuntimeError):
    pass


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def split_indices(labels, count, seed):
    generator = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    selected = []
    for i, label in enumerate(classes):
        candidates = torch.where(labels == label)[0]
        take = count // len(classes) + int(i < count % len(classes))
        if take >= len(candidates):
            raise ValueError("Holdout leaves no training examples in a class")
        selected.append(candidates[torch.randperm(len(candidates), generator=generator)[:take]])
    validation = torch.cat(selected)
    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[validation] = False
    return torch.where(mask)[0], validation


def load_data(p, stage, root):
    """Tuning never opens the official test set; final evaluation trains on all 60k."""
    if p["smoke"]:
        generator = torch.Generator().manual_seed(9876)
        raw_x = torch.randn(60, 784, generator=generator)
        raw_y = torch.arange(60) % 10
        if stage == "eval":
            train_indices = torch.arange(60)
            evaluation_indices = torch.arange(20)
            train_x, train_y = raw_x, raw_y
            evaluation_x = torch.randn(20, 784, generator=generator)
            evaluation_y = torch.arange(20) % 10
        else:
            train_indices, evaluation_indices = split_indices(raw_y, p["holdout_examples"], p["split_seed"])
            train_x, train_y = raw_x[train_indices], raw_y[train_indices]
            evaluation_x, evaluation_y = raw_x[evaluation_indices], raw_y[evaluation_indices]
    else:
        from torchvision.datasets import MNIST
        train = MNIST(str(root), train=True, download=True)
        raw_y = train.targets
        raw_x = train.data.flatten(1)
        if stage == "eval":
            test = MNIST(str(root), train=False, download=True)
            train_indices = torch.arange(len(raw_y))
            evaluation_indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(p["test_subset_seed"]))[:p["test_examples"]]
            train_x, train_y = raw_x, raw_y
            evaluation_x, evaluation_y = test.data.flatten(1)[evaluation_indices], test.targets[evaluation_indices]
        else:
            train_indices, evaluation_indices = split_indices(raw_y, p["holdout_examples"], p["split_seed"])
            train_x, train_y = raw_x[train_indices], raw_y[train_indices]
            evaluation_x, evaluation_y = raw_x[evaluation_indices], raw_y[evaluation_indices]
        train_x = train_x.float().div_(255).sub_(.1307).div_(.3081)
        evaluation_x = evaluation_x.float().div_(255).sub_(.1307).div_(.3081)
    identity = dict(train_examples=len(train_y), evaluation_examples=len(evaluation_y),
                    train_indices_sha256=tensor_hash(train_indices), eval_indices_sha256=tensor_hash(evaluation_indices),
                    train_x_sha256=tensor_hash(train_x), train_y_sha256=tensor_hash(train_y),
                    eval_x_sha256=tensor_hash(evaluation_x), eval_y_sha256=tensor_hash(evaluation_y),
                    evaluation_source="official_test" if stage == "eval" else "train_holdout",
                    smoke=p["smoke"])
    return train_x, train_y, evaluation_x, evaluation_y, identity, train_indices, evaluation_indices


def make_optimizer(model, job):
    if job["arm"] == "cap":
        from cap_adam import CAPAdam
        return CAPAdam(model.parameters(), **job["config"])
    source = protocol.ROOT / protocol.UPGD_SOURCE
    if protocol.file_hash(source) != protocol.UPGD_SHA256:
        raise RuntimeError("Changed UPGD source")
    spec = importlib.util.spec_from_file_location("rebuttal_pinned_upgd", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FirstOrderGlobalUPGD(model.named_parameters(), **job["config"])


def stream_spec(p, job):
    if job["stage"] != "eval":
        pool, tasks, base, seed = p["search_pool"], p["search_tasks"], p["search_permutation_base"], p["search_schedule_seed"]
    elif job["stream"] == "b":
        pool, tasks, base, seed = p["eval_pool"], p["eval_tasks"], p["stream_b_permutation_base"], p["stream_b_schedule_seed"]
    else:
        tasks = p["eval_tasks"]
        domains = list(range(1, tasks+1))
        return domains, {d: torch.randperm(784, generator=torch.Generator().manual_seed(p["stream_a_permutation_base"]+d)) for d in domains}
    generator = torch.Generator().manual_seed(seed)
    schedule = torch.cat([torch.randperm(pool, generator=generator) for _ in range(tasks//pool)]).tolist()
    permutations = {d: torch.randperm(784, generator=torch.Generator().manual_seed(base+d)) for d in range(pool)}
    return schedule, permutations


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    try:
        torchvision = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision = None
    return dict(python=platform.python_version(), platform=platform.system(), torch=torch.__version__,
                torchvision=torchvision, cuda=torch.version.cuda, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                tf32=torch.backends.cuda.matmul.allow_tf32,
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


@torch.no_grad()
def evaluate(model, x, y, permutations, domains, device, batch_size):
    model.eval()
    rows = []
    try:
        for domain in domains:
            correct = 0
            for start in range(0, len(y), batch_size):
                logits = model(x[start:start+batch_size][:, permutations[domain]].to(device))
                if not torch.isfinite(logits).all().item():
                    raise NumericalDivergence("Nonfinite evaluation logits")
                correct += (logits.argmax(1) == y[start:start+batch_size].to(device)).sum().item()
            rows.append(dict(domain=domain, accuracy=correct/len(y)))
    finally:
        model.train()
    return rows


def atomic_checkpoint(path, state):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_rows(p, job, state):
    """Coverage, duplicate and current-domain checks before a result is admitted."""
    if state["step"] != job["updates"]:
        raise RuntimeError("Incomplete training horizon")
    tasks = p["eval_tasks"] if job["stage"] == "eval" else p["search_tasks"]
    if [r["task"] for r in state["tasks"]] != list(range(1, tasks+1)):
        raise RuntimeError("Incomplete task rows")
    if [r["step"] for r in state["scalars"]] != list(range(p["observation_interval"], job["updates"]+1, p["observation_interval"])):
        raise RuntimeError("Incomplete scalar observations")
    if job["stream"] != "a":
        pool = p["eval_pool"] if job["stage"] == "eval" else p["search_pool"]
        expected = {(t, d) for t in range(1, tasks+1) for d in range(pool)}
        actual = [(r["task"], r["domain"]) for r in state["pool"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError("Missing/duplicate pool evaluations")
        for row in state["tasks"]:
            current = [r["domain"] for r in state["pool"] if r["task"] == row["task"] and r["is_current"]]
            if current != [row["current_domain"]]:
                raise RuntimeError("Wrong current-domain labels")


def metrics(p, job, state):
    validate_rows(p, job, state)
    if job["stage"] != "eval":
        plasticity = sum(r["task_end_accuracy"] for r in state["tasks"][p["search_pool"]:])/p["search_pool"]
        retained = [r["accuracy"] for r in state["pool"] if r["task"] == p["search_tasks"] and not r["is_current"]]
        retention = sum(retained)/len(retained)
        return dict(plasticity=plasticity, retention=retention, balanced=min(plasticity, retention))
    if job["stream"] == "a":
        values = [r["accuracy"] for r in state["scalars"] if r["step"] > job["updates"]//2]
        return dict(plasticity=sum(values)/len(values))
    retained = [r["accuracy"] for r in state["pool"] if r["task"] == p["eval_tasks"] and not r["is_current"]]
    return dict(retention=sum(retained)/len(retained))


def run_job(output, job, *, data_dir, device="cuda", resume=False, checkpoint_every=1000,
            stop_after_updates=None):
    if checkpoint_every <= 0 or (stop_after_updates is not None and stop_after_updates <= 0):
        raise ValueError("Checkpoint/stop intervals must be positive")
    p = protocol.campaign(output)
    # Do not allow custom job configs, seeds or shortened horizons through the API.
    allowed = protocol.jobs(output, job["stage"], (job["arm"],) if job["stage"] == "eval" else ("selected",))
    if job not in allowed:
        raise RuntimeError("Job is not in the frozen campaign")
    directory = Path(output)/"jobs"/job["id"]
    with protocol.file_lock(directory/"worker.lock"):
        existing = protocol.result(output, job, required=False)
        if existing:
            if not resume:
                raise FileExistsError(f"{job['id']} is complete; use --resume to verify and skip")
            print(f"Verified {job['id']}: {existing['status']}; skipped", flush=True)
            return existing
        progress = directory/"latest.pt"
        if (progress.exists() or (directory/"job.json").exists()) and not resume:
            raise FileExistsError(f"Partial {job['id']}; use --resume")
        protocol.freeze_json(directory/"job.json", job)
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu only for smoke/testing")
        torch.set_num_threads(1)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        seed_everything(job["seed"])
        protocol.freeze_json(directory/"environment.json", environment(device))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        attempt = uuid.uuid4().hex
        attempt_started = time.perf_counter()
        state = dict(step=0, scalars=[], tasks=[], pool=[], task_accuracy_sum=0., task_count=0,
                     training_seconds=0., evaluation_seconds=0., status="running")

        def event(kind, **extra):
            synchronize(device)
            protocol.append_event(directory/"timing.jsonl", dict(attempt=attempt, event=kind,
                unix_time=time.time(), elapsed_seconds=time.perf_counter()-attempt_started,
                step=state["step"], **extra))

        event("start", checkpoint_every=checkpoint_every, stop_after_updates=stop_after_updates)
        try:
            tx, ty, ex, ey, data_identity, train_indices, eval_indices = load_data(p, job["stage"], data_dir)
            protocol.freeze_json(directory/"data_identity.json", data_identity)
            model = MLP300(784, 10).to(device)
            optimizer = make_optimizer(model, job)
            sampler = IndexSampler(len(ty), job["seed"], p["batch_size"], persistent_workers=True)
            schedule, permutations = stream_spec(p, job)
            identity = dict(job=job, environment=environment(device), data=data_identity)
            if progress.exists():
                # Trusted local campaign checkpoints include Python/NumPy RNG state.
                saved = torch.load(progress, map_location="cpu", weights_only=False)
                if saved["identity"] != identity:
                    raise RuntimeError("Checkpoint identity differs from job/environment/data")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                sampler.load_state_dict(saved["sampler"])
                state = saved["progress"]
                restore_rng(saved["rng"])
            start_step = state["step"]
            steps_per_task = p["updates_per_task"]
            last_commit_step = start_step

            def checkpoint():
                nonlocal last_commit_step
                synchronize(device)
                started = time.perf_counter()
                saved = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             sampler=sampler.state_dict(), rng=rng_state(), progress=state,
                             parameter_names=[name for name, _ in model.named_parameters()],
                             train_indices=train_indices, evaluation_indices=eval_indices,
                             schedule=schedule)
                atomic_checkpoint(progress, saved)
                if job["stage"] == "eval" and job["stream"] == "b" and state["step"] % steps_per_task == 0:
                    task = state["step"]//steps_per_task
                    if task in p["evaluation_checkpoint_tasks"] and state["status"] != "diverged":
                        # Hard links preserve the committed inode across later atomic replacements.
                        target = directory/f"task_{task:03d}.pt"
                        if not target.exists():
                            try:
                                os.link(progress, target)
                            except OSError:
                                atomic_checkpoint(target, saved)
                        else:
                            archived = torch.load(target, map_location="cpu", weights_only=False)
                            if archived["identity"] != identity or archived["progress"]["step"] != state["step"]:
                                raise RuntimeError("Archived evaluation checkpoint identity changed")
                event("checkpoint", committed_updates=state["step"]-last_commit_step,
                      checkpoint_seconds=time.perf_counter()-started,
                      training_seconds=state["training_seconds"], evaluation_seconds=state["evaluation_seconds"])
                last_commit_step = state["step"]

            if not progress.exists():
                checkpoint()
            segment_started = time.perf_counter()
            while state["step"] < job["updates"] and state["status"] == "running":
                task = state["step"]//steps_per_task+1
                domain = schedule[task-1]
                indices = sampler.next_indices()
                x, y = tx[indices][:, permutations[domain]].to(device), ty[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    state["status"] = "diverged"
                    state["failure_reason"] = "Nonfinite training loss"
                    break
                accuracy = (logits.argmax(1) == y).float().mean().item()
                loss.backward()
                # Match the existing harness adaptation of the upstream scalar factory.
                with torch.device(device):
                    optimizer.step()
                state["step"] += 1
                step = state["step"]
                state["task_accuracy_sum"] += accuracy
                state["task_count"] += 1
                if step % p["observation_interval"] == 0:
                    state["scalars"].append(dict(step=step, task=task, accuracy=accuracy, loss=loss_value))
                boundary = step % steps_per_task == 0
                pause = stop_after_updates is not None and step-start_step >= stop_after_updates
                commit = boundary or step % checkpoint_every == 0 or pause
                if not commit:
                    continue
                synchronize(device)
                state["training_seconds"] += time.perf_counter()-segment_started
                if boundary:
                    started = time.perf_counter()
                    domains = [domain] if job["stream"] == "a" else list(permutations)
                    try:
                        evaluated = evaluate(model, ex, ey, permutations, domains, device, p["eval_batch_size"])
                    except NumericalDivergence as error:
                        state["status"], state["failure_reason"] = "diverged", str(error)
                        evaluated = []
                    synchronize(device)
                    state["evaluation_seconds"] += time.perf_counter()-started
                    if evaluated:
                        current_accuracy = next(r["accuracy"] for r in evaluated if r["domain"] == domain)
                        state["tasks"].append(dict(task=task, current_domain=domain,
                            online_accuracy=state["task_accuracy_sum"]/state["task_count"], task_end_accuracy=current_accuracy))
                        if job["stream"] != "a":
                            state["pool"].extend(dict(task=task, **r, is_current=r["domain"] == domain) for r in evaluated)
                        state["task_accuracy_sum"], state["task_count"] = 0., 0
                    print(f"{job['id']}: task {task}, step {step}/{job['updates']}", flush=True)
                if state["step"] == job["updates"] and state["status"] == "running":
                    state["status"] = "complete"
                checkpoint()
                if pause and state["status"] == "running":
                    event("paused", invocation_updates=step-start_step)
                    return dict(status="paused", step=step)
                segment_started = time.perf_counter()
            if state["status"] == "diverged":
                synchronize(device)
                state["training_seconds"] += time.perf_counter()-segment_started
                checkpoint()
            values = metrics(p, job, state) if state["status"] == "complete" else {}
            for name, fields in (("scalars", SCALAR_FIELDS), ("tasks", TASK_FIELDS), ("pool", POOL_FIELDS)):
                protocol.write_csv(directory/f"{name}.csv", state[name], fields)
            event(state["status"], invocation_updates=state["step"]-start_step)
            output_result = dict(job=job, status=state["status"], step=state["step"], metrics=values,
                failure_reason=state.get("failure_reason"), smoke=p["smoke"],
                training_seconds=state["training_seconds"], evaluation_seconds=state["evaluation_seconds"],
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                timing=protocol.timing_summary(directory/"timing.jsonl"),
                checkpoint_hashes={f"task_{task:03d}.pt": protocol.file_hash(directory/f"task_{task:03d}.pt")
                                   for task in job["checkpoint_tasks"]} if state["status"] == "complete" else {},
                artifact_hashes={name: protocol.file_hash(directory/name) for name in ("latest.pt", "scalars.csv", "tasks.csv", "pool.csv")})
            protocol.atomic_json(directory/"result.json", output_result)
            return output_result
        except BaseException as error:
            event("error", error_type=type(error).__name__, message=str(error))
            raise
