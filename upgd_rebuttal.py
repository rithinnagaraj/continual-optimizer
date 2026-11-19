"""Frozen, standard-library-only protocol and job management for the rebuttal."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import uuid

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results/upgd_rebuttal"
VERSION = "upgd-rebuttal-v1"
REPORT_SEEDS = [42, 123, 451, 789, 2026]
INCUMBENT = dict(lr=1e-3, sigma=1e-4, beta_utility=.9999, weight_decay=0.)
CAP_CONFIG = dict(lr=1/300, k=7e-5, T_a=2000., T_s=2000., T_m=1e6)
UPGD_SOURCE = "external/upgd/core/optim/weight_upgd/first_order.py"
UPGD_SHA256 = "d976b8ca34cced48b79a2002a638f717b72ad00368f6eeb7d1d424fa5724cdef"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def file_lock(path):
    """An OS lock is released on process death; never remove the lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process holds {path}; do not share a running job") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def freeze_json(path, value):
    path = Path(path)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            if read_json(path) != value:
                raise RuntimeError(f"Frozen artifact changed: {path}; use a new campaign directory")
        else:
            atomic_json(path, value)
    return value


def source_hashes():
    paths = ["upgd_rebuttal.py", "upgd_rebuttal_runtime.py", "experiment_upgd_tuning.py",
             "experiment_upgd_evaluation.py", "model.py", "cap_adam.py", UPGD_SOURCE]
    # The existing exact sampler is imported without modifying its frozen package.
    paths += [p.relative_to(ROOT).as_posix() for p in sorted(
        (ROOT / "cap_mechanistic/cap_mech").rglob("*")) if p.suffix in {".py", ".json"}]
    result = {name: file_hash(ROOT / name) for name in paths}
    if result[UPGD_SOURCE] != UPGD_SHA256:
        raise RuntimeError("Vendored UPGD differs from the paper's pinned implementation")
    return result


def grid():
    return [dict(lr=lr, sigma=sigma, beta_utility=beta, weight_decay=wd)
            for lr, sigma, beta, wd in itertools.product(
                (1e-4, 1e-3, 1e-2, 3e-2), (0., 1e-4, 1e-2, 1e-1),
                (.999, .9999), (0., .01))]


def protocol(smoke=False):
    return dict(version=VERSION, smoke=smoke, configurations=grid(), incumbent=INCUMBENT,
                cap_reference=CAP_CONFIG, search_seed=31415, confirmation_seed=27182,
                report_seeds=REPORT_SEEDS, shortlist_top=3,
                selection="mean per-seed min(P_val,R_val); then mean (P_val+R_val)/2; then grid index",
                search_pool=2 if smoke else 20, search_tasks=4 if smoke else 40,
                eval_pool=2 if smoke else 40, eval_tasks=4 if smoke else 200,
                updates_per_task=4 if smoke else 5000, batch_size=8 if smoke else 128,
                observation_interval=2 if smoke else 100, eval_batch_size=128,
                holdout_examples=20 if smoke else 5000, split_seed=777,
                test_examples=20 if smoke else 2000, test_subset_seed=999,
                search_permutation_base=300001, search_schedule_seed=556,
                stream_a_permutation_base=100000, stream_b_permutation_base=200001,
                stream_b_schedule_seed=555, evaluation_checkpoint_tasks=[2, 4] if smoke else [100, 200],
                sampler="existing IndexSampler; persistent-worker RNG semantics; retain short batches",
                selection_data="stratified MNIST train holdout; official test unused for selection",
                tuning_P="mean current-domain held-out task-end accuracy in round two",
                tuning_R="final held-out mean over non-current validation domains",
                evaluation_P="pre-update incoming-batch accuracy every 100 updates in second half of A",
                evaluation_R="final official-test mean over 39 non-current B domains",
                source_hashes=source_hashes())


def initialize(output, smoke=False):
    return freeze_json(Path(output) / "campaign.json", protocol(smoke))


def campaign(output):
    value = read_json(Path(output) / "campaign.json")
    if value != protocol(value["smoke"]):
        raise RuntimeError("Campaign source or protocol changed; create a new campaign, do not mix runs")
    return value


def job_spec(p, stage, index=None, seed=None, stream=None, arm="upgd"):
    config = p["cap_reference"] if arm == "cap" else p["configurations"][index]
    if stage == "reference":
        name = f"reference-cap-{seed}"
    elif stage in {"search", "confirm"}:
        seed = p["search_seed"] if stage == "search" else p["confirmation_seed"]
        name = f"{stage}-{index:03d}"
    else:
        name = f"eval-{arm}-{seed}-{stream}"
    return dict(id=name, stage=stage, config_index=index, config=config, seed=seed,
                stream=stream or "validation", arm=arm, campaign_hash=digest(p),
                checkpoint_tasks=p["evaluation_checkpoint_tasks"] if stage == "eval" and stream == "b" else [],
                updates=p["updates_per_task"] * p["eval_tasks" if stage == "eval" else "search_tasks"])


def jobs(output, stage, arms=("selected",)):
    p = campaign(output)
    if stage == "search":
        return [job_spec(p, stage, i) for i in range(len(p["configurations"]))]
    if stage == "reference":
        return [job_spec(p, stage, seed=s, arm="cap") for s in (p["search_seed"], p["confirmation_seed"])]
    if stage == "confirm":
        shortlist = make_shortlist(output, write=False)
        if read_json(Path(output) / "shortlist.json") != shortlist:
            raise RuntimeError("Shortlist does not match committed search results")
        return [job_spec(p, stage, i) for i in shortlist["indices"]]
    if stage != "eval":
        raise ValueError(stage)
    selected = choose(output, write=False)
    if read_json(Path(output) / "selected_config.json") != selected:
        raise RuntimeError("Selection does not match committed confirmation results")
    incumbent_index = p["configurations"].index(p["incumbent"])
    result = []
    for arm in arms:
        if arm not in {"selected", "incumbent", "cap"}:
            raise ValueError(arm)
        index = selected["config_index"] if arm == "selected" else incumbent_index if arm == "incumbent" else None
        result += [job_spec(p, "eval", index, seed, stream, arm)
                   for seed in p["report_seeds"] for stream in ("a", "b")]
    return result


def result(output, job, required=True):
    directory = Path(output) / "jobs" / job["id"]
    path = directory / "result.json"
    if not path.exists():
        if required:
            raise RuntimeError(f"Missing terminal result: {job['id']}")
        return None
    value = read_json(path)
    if value["job"] != job or value["status"] not in {"complete", "diverged"}:
        raise RuntimeError(f"Invalid result identity/status: {job['id']}")
    for name, expected in value["artifact_hashes"].items():
        if name not in {"latest.pt", "scalars.csv", "tasks.csv", "pool.csv"}:
            raise RuntimeError("Unexpected committed artifact")
        if not (directory / name).is_file() or file_hash(directory / name) != expected:
            raise RuntimeError(f"Changed/missing committed artifact: {job['id']}/{name}")
    if set(value["artifact_hashes"]) != {"latest.pt", "scalars.csv", "tasks.csv", "pool.csv"}:
        raise RuntimeError("Incomplete result artifact list")
    if value["status"] == "complete":
        expected_metrics = {"plasticity", "retention", "balanced"} if job["stage"] != "eval" else {
            "plasticity" if job["stream"] == "a" else "retention"}
        if (set(value["metrics"]) != expected_metrics or value["step"] != job["updates"]
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in value["metrics"].values())):
            raise RuntimeError("Incomplete/nonfinite scientific result")
        expected_checkpoints = {f"task_{t:03d}.pt" for t in job["checkpoint_tasks"]}
        if set(value.get("checkpoint_hashes", {})) != expected_checkpoints:
            raise RuntimeError("Missing retained evaluation checkpoints")
        for name, expected in value["checkpoint_hashes"].items():
            if not (directory/name).is_file() or file_hash(directory/name) != expected:
                raise RuntimeError(f"Retained evaluation checkpoint changed: {name}")
    return value


def score_key(value):
    m = value["metrics"]
    return (-m["balanced"], -(m["plasticity"] + m["retention"])/2, value["job"]["config_index"])


def make_shortlist(output, write=True):
    p = campaign(output)
    rows = [result(output, j) for j in jobs(output, "search")]
    finite = sorted((r for r in rows if r["status"] == "complete"), key=score_key)
    if not finite:
        raise RuntimeError("No finite candidates; search cannot select a winner")
    incumbent_index = p["configurations"].index(p["incumbent"])
    if rows[incumbent_index]["status"] != "complete":
        raise RuntimeError("Incumbent failed; inspect integration before confirmation")
    indices = [r["job"]["config_index"] for r in finite[:p["shortlist_top"]]]
    if incumbent_index not in indices:
        indices.append(incumbent_index)
    value = dict(campaign_hash=digest(p), indices=indices,
                 search_result_hashes={r["job"]["id"]: digest(r) for r in rows})
    return freeze_json(Path(output)/"shortlist.json", value) if write else value


def choose(output, write=True):
    p = campaign(output)
    confirms = [result(output, j) for j in jobs(output, "confirm")]
    candidates, excluded = [], []
    for r in confirms:
        index = r["job"]["config_index"]
        primary = result(output, job_spec(p, "search", index))
        if r["status"] != "complete":
            excluded.append(index)
            continue
        candidates.append(dict(config_index=index, config=p["configurations"][index],
            balanced=mean([primary["metrics"]["balanced"], r["metrics"]["balanced"]]),
            mean_pr=mean([v["metrics"][key] for v in (primary, r) for key in ("plasticity", "retention")])))
    if not candidates:
        raise RuntimeError("Every confirmation failed; cannot freeze selection")
    candidates.sort(key=lambda r: (-r["balanced"], -r["mean_pr"], r["config_index"]))
    value = dict(campaign_hash=digest(p), **candidates[0], candidates=candidates,
                 excluded_confirmation_indices=excluded, selection_rule=p["selection"],
                 shortlist_hash=digest(read_json(Path(output)/"shortlist.json")),
                 confirmation_result_hashes={r["job"]["id"]: digest(r) for r in confirms})
    return freeze_json(Path(output)/"selected_config.json", value) if write else value


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_events(path):
    if not Path(path).exists():
        return []
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        if not line.endswith(b"\n") and i == len(lines)-1:
            break  # an interrupted final append is not a committed event
        rows.append(json.loads(line))
    return rows


def append_event(path, value):
    path = Path(path)
    # Called only under the exclusive job lock; discard a crash-truncated tail.
    if path.exists():
        with path.open("r+b") as handle:
            data = handle.read()
            if data and not data.endswith(b"\n"):
                handle.truncate(data.rfind(b"\n") + 1)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def timing_summary(path):
    attempts = {}
    committed = 0
    for row in read_events(path):
        if row["event"] in {"checkpoint", "complete", "paused", "diverged"}:
            committed = max(committed, row.get("step", 0))
        previous = attempts.get(row["attempt"], {})
        if row["elapsed_seconds"] >= previous.get("elapsed_seconds", -1):
            attempts[row["attempt"]] = row
    return dict(attempts=len(attempts), last_committed_step=committed,
                observed_wall_seconds=sum(r["elapsed_seconds"] for r in attempts.values()),
                interrupted_attempts=sum(r["event"] not in {"complete", "paused", "diverged", "error"}
                                         for r in attempts.values()),
                wall_time_scope="sum of attempt durations; excludes downtime; killed tails beyond last event are unknown")


def evaluation_report(output, arms):
    p = campaign(output)
    all_jobs = jobs(output, "eval", arms)
    rows = {j["id"]: result(output, j) for j in all_jobs}
    if any(r["status"] != "complete" for r in rows.values()):
        raise RuntimeError("Final evaluation contains a divergence; no complete benchmark table")
    per_seed = []
    for arm in arms:
        for seed in p["report_seeds"]:
            a, b = (rows[f"eval-{arm}-{seed}-{s}"] for s in ("a", "b"))
            plasticity, retention = a["metrics"]["plasticity"], b["metrics"]["retention"]
            per_seed.append(dict(arm=arm, seed=seed, plasticity=plasticity,
                                 retention=retention, balanced=min(plasticity, retention)))
    summary = []
    for arm in arms:
        subset = [r for r in per_seed if r["arm"] == arm]
        summary.append(dict(arm=arm, seeds=len(subset), **{
            k: mean(r[k] for r in subset) for k in ("plasticity", "retention", "balanced")}, **{
            k+"_std": stdev(r[k] for r in subset) for k in ("plasticity", "retention", "balanced")}))
    root = Path(output)/"analysis"
    write_csv(root/"per_seed_metrics.csv", per_seed, list(per_seed[0]))
    write_csv(root/"benchmark.csv", summary, list(summary[0]))
    atomic_json(root/"evaluation_manifest.json", dict(campaign_hash=digest(p), smoke=p["smoke"],
        selected_config_hash=digest(read_json(Path(output)/"selected_config.json")),
        result_hashes={k: digest(v) for k, v in rows.items()}))
    return summary


def tuning_report(output):
    p = campaign(output)
    rows, curve = [], []
    best = None
    for j in jobs(output, "search"):
        r = result(output, j)
        row = dict(index=j["config_index"], **j["config"], status=r["status"],
                   **{k: r["metrics"].get(k) for k in ("plasticity", "retention", "balanced")})
        row.update(timing_summary(Path(output)/"jobs"/j["id"]/"timing.jsonl"))
        rows.append(row)
        if r["status"] == "complete":
            best = r["metrics"]["balanced"] if best is None else max(best, r["metrics"]["balanced"])
        curve.append(dict(trials=len(rows), scheduled_updates=len(rows)*j["updates"], best_balanced=best))
    root = Path(output)/"analysis"
    write_csv(root/"search.csv", rows, list(rows[0]))
    write_csv(root/"search_budget_curve.csv", curve, list(curve[0]))
    atomic_json(root/"search_manifest.json", dict(campaign_hash=digest(p), smoke=p["smoke"],
        curve_order="fixed grid order, descriptive validation curve, not randomized search-efficiency inference"))
    references = []
    for j in jobs(output, "reference"):
        r = result(output, j, required=False)
        if r is not None:
            references.append(dict(seed=j["seed"], status=r["status"], **{
                k: r["metrics"].get(k) for k in ("plasticity", "retention", "balanced")}))
    if references:
        write_csv(root/"cap_validation_reference.csv", references, list(references[0]))
    return {"completed_trials": len(rows), "best_validation_balanced": best}


def dispatch(args, stage, arms=("selected",)):
    available = jobs(args.output_dir, stage, arms)
    requested = set(args.job_ids or [])
    if requested - {j["id"] for j in available}:
        raise ValueError("Unknown job id for this stage")
    chosen = available if args.all else [j for j in available if j["id"] in requested]
    if not chosen:
        raise ValueError("Specify --job-ids or --all explicitly")
    if args.max_jobs is not None:
        pending = [j for j in chosen if result(args.output_dir, j, required=False) is None]
        chosen = pending[:args.max_jobs]
    if args.dry_run:
        print(json.dumps(chosen, indent=2))
        return
    from upgd_rebuttal_runtime import run_job
    for job in chosen:
        run_job(args.output_dir, job, data_dir=args.data_dir, device=args.device,
                resume=args.resume, checkpoint_every=args.checkpoint_every,
                stop_after_updates=args.stop_after_updates)


def add_run_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-ids", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=positive)
    parser.add_argument("--checkpoint-every", type=positive, default=1000)
    parser.add_argument("--stop-after-updates", type=positive,
                        help="Pause each job after this many additional updates, without changing its scientific horizon")


def positive(value):
    import argparse
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number
