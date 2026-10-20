"""Symmetric Phase-1 tuning on Stream-A tasks 1--60, seed 42 only."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import torch

from experiment_e2 import read_csv, write_csv
from experiment_e3 import (
    fixed_permutation,
    load_mnist,
    make_loader,
    repeat_batches,
    seed_everything,
)
from experiment_e4a import (
    BATCH_SIZE,
    C_REFRESH_STEPS,
    EVALUATION_BATCH_SIZE,
    FULL_SEEDS,
    atomic_write_json,
    fixed_test_subset,
    refresh_global_c,
    utc_now,
    verify_preconditions,
)
from model import MLP300
from phase1_arms import (
    P1_A3_DISCLOSURE,
    UPGD_AMENDMENT,
    UPGD_DEFAULTS,
    build_arm,
    cap_release_matches_core,
    upgd_provenance,
)
from phase1_stream_common import (
    StepsToThreshold,
    evaluate_permutation,
    permute_batch,
    training_step,
)


TUNING_SEED = 42
TASK_COUNT = 60
STEPS_PER_TASK = 5_000
TOTAL_STEPS = TASK_COUNT * STEPS_PER_TASK
SELECTION_START_STEP = 100_001
PERMUTATION_SEED_BASE = 100_000
UPGD_SMOKE_TASKS = 3
UPGD_SMOKE_STEPS = UPGD_SMOKE_TASKS * STEPS_PER_TASK
UPGD_SMOKE_TAIL = 1_000
UPGD_SMOKE_ACCURACY = 0.90

METHODS = (
    "adamw",
    "adam_replay",
    "upgd",
    "ewc_oracle",
    "l2_init",
    "cap_release",
)

TUNING_FIELDS = (
    "method",
    "config_index",
    "config_json",
    "task",
    "task_end_accuracy",
    "steps_to_085",
    "selection_metric",
    "tie_break_metric",
    "runtime_seconds",
    "status",
)

TUNING_DISCLOSURE = (
    "Tuning uses seed 42 and the deployment permutation formula. "
    "Online-stream tuning has no held-out notion (the accepted streaming-CL "
    "convention); every method receives an identical budget and stream, so "
    "there is no relative bias. Seed 42 remains in final aggregates."
)
EWC_DISCLOSURE = (
    "ewc_oracle receives task-boundary signals unavailable to every other "
    "arm; its results are boundary-privileged."
)
REPLAY_DISCLOSURE = (
    "adam_replay is an out-of-class reference (stores data; violates the "
    "streaming constraint)."
)
PRE_AMENDMENT_ARCHIVE_DIR = "p1_a3_pre_amendment"
PRE_AMENDMENT_SMOKE = {
    "status": "INTEGRATION-FAIL",
    "threshold": 0.90,
    "attempts": [
        {
            "lr": 1e-5,
            "last_1000_task1_online_accuracy": 0.04746875000745058,
            "no_nan": True,
            "passed": False,
        },
        {
            "lr": 1e-5 / 3.0,
            "last_1000_task1_online_accuracy": 0.04370312500745058,
            "no_nan": True,
            "passed": False,
        },
        {
            "lr": 3e-5,
            "last_1000_task1_online_accuracy": 0.06286979167163372,
            "no_nan": True,
            "passed": False,
        },
    ],
}


def display_method(method: str) -> str:
    return (
        "adam_replay — out-of-class reference (stores data; violates the "
        "streaming constraint)"
        if method == "adam_replay"
        else method
    )


def _is_pre_amendment_failure(config: Mapping[str, Any]) -> bool:
    gate = config.get("upgd_smoke_gate", {})
    return (
        config.get("amendment") != UPGD_AMENDMENT
        and isinstance(gate, Mapping)
        and gate.get("status") == "INTEGRATION-FAIL"
    )


def _atomic_preserve(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(
                f"preserved P1-A3 evidence differs from {source}: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def preserve_pre_amendment_evidence(
    *, config_path: Path, report_path: Path, existing: Mapping[str, Any]
) -> bool:
    """Archive the registered README-default failure before v4 overwrites it."""
    if not _is_pre_amendment_failure(existing):
        return False
    if not report_path.exists():
        raise FileNotFoundError(
            "P1-A3 requires preserving the original failed smoke report, but "
            f"it is missing: {report_path}"
        )
    archive = config_path.parent / PRE_AMENDMENT_ARCHIVE_DIR
    _atomic_preserve(config_path, archive / "run_config.json")
    _atomic_preserve(report_path, archive / "report.md")
    return True


def registered_grids() -> dict[str, list[dict[str, Any]]]:
    """Return all 41 configurations in their preregistered tie-break order."""
    upgd_lr = float(UPGD_DEFAULTS["lr"])
    upgd_sigma = float(UPGD_DEFAULTS["sigma"])
    upgd_beta = float(UPGD_DEFAULTS["beta_utility"])
    return {
        "adamw": [
            {"lr": lr}
            for lr in (3e-4, 1e-3, 3e-3)
        ],
        "adam_replay": [
            {"lr": lr, "replay_count": replay_count}
            for lr, replay_count in itertools.product(
                (3e-4, 1e-3, 3e-3), (16, 32, 64)
            )
        ],
        "upgd": [
            {
                "lr": lr,
                "weight_decay": float(UPGD_DEFAULTS["weight_decay"]),
                "sigma": sigma,
                "beta_utility": beta,
            }
            for lr, sigma, beta in itertools.product(
                (upgd_lr / 3.0, upgd_lr, upgd_lr * 3.0),
                (upgd_sigma, 0.01),
                (upgd_beta, 0.99999),
            )
        ],
        "ewc_oracle": [
            {"lambda": lam, "gamma": gamma}
            for lam, gamma in itertools.product((1e3, 1e4, 1e5), (0.95, 1.0))
        ],
        "l2_init": [
            {"lambda": lam}
            for lam in (1e-4, 1e-3, 1e-2)
        ],
        "cap_release": [
            {"k": k, "T_m": t_m, "T_a": t_a, "retention": "release"}
            for k, t_m, t_a in itertools.product(
                (3e-5, 7e-5), (10_000, 50_000), (2_000, 10_000)
            )
        ],
    }


def _direct_configs(method: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {"methods": {method: {"config": dict(config)}}}


def run_upgd_smoke_gate(
    *,
    mnist_train,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    """Run the mandatory three-task integration gate and its LR neighbors."""
    default_lr = float(UPGD_DEFAULTS["lr"])
    attempts = []
    for lr in (default_lr, default_lr / 3.0, default_lr * 3.0):
        config = {
            "lr": lr,
            "weight_decay": float(UPGD_DEFAULTS["weight_decay"]),
            "sigma": float(UPGD_DEFAULTS["sigma"]),
            "beta_utility": float(UPGD_DEFAULTS["beta_utility"]),
        }
        seed_everything(TUNING_SEED)
        model = MLP300(input_units=784, output_units=10).to(device)
        optimizer, hooks = build_arm(
            "upgd", model, _direct_configs("upgd", config), TUNING_SEED, device
        )
        loader = make_loader(
            mnist_train,
            BATCH_SIZE,
            device,
            num_workers,
            shuffle=True,
            seed=TUNING_SEED,
        )
        batches = repeat_batches(loader)
        permutations = {
            task: fixed_permutation(784, PERMUTATION_SEED_BASE + task).to(device)
            for task in range(1, UPGD_SMOKE_TASKS + 1)
        }
        task1_tail: deque[float] = deque(maxlen=UPGD_SMOKE_TAIL)
        no_nan = True
        for step in range(1, UPGD_SMOKE_STEPS + 1):
            task = ((step - 1) // STEPS_PER_TASK) + 1
            inputs, targets = next(batches)
            inputs = permute_batch(
                inputs.to(device, non_blocking=True), permutations[task]
            )
            targets = targets.to(device, non_blocking=True)
            loss, accuracy = training_step(
                model=model,
                optimizer=optimizer,
                hooks=hooks,
                inputs=inputs,
                targets=targets,
            )
            if not bool(torch.isfinite(loss).item()):
                no_nan = False
                break
            if task == 1 and step > STEPS_PER_TASK - UPGD_SMOKE_TAIL:
                task1_tail.append(accuracy)
        no_nan = no_nan and all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in model.parameters()
        )
        tail_accuracy = mean(task1_tail) if task1_tail else None
        passed = (
            no_nan
            and len(task1_tail) == UPGD_SMOKE_TAIL
            and tail_accuracy is not None
            and tail_accuracy >= UPGD_SMOKE_ACCURACY
        )
        attempts.append(
            {
                "lr": lr,
                "last_1000_task1_online_accuracy": tail_accuracy,
                "no_nan": no_nan,
                "passed": passed,
                "resolved_kwargs": dict(config),
            }
        )
        del model, optimizer, loader, batches
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if passed:
            break
    return {
        "status": "PASS" if any(item["passed"] for item in attempts) else "INTEGRATION-FAIL",
        "threshold": UPGD_SMOKE_ACCURACY,
        "attempts": attempts,
    }


def run_tuning_cell(
    *,
    method: str,
    config_index: int,
    config: Mapping[str, Any],
    mnist_train,
    test_loader,
    permutations_cpu: Mapping[int, torch.Tensor],
    device: torch.device,
    num_workers: int,
) -> list[dict[str, Any]]:
    seed_everything(TUNING_SEED)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, hooks = build_arm(
        method,
        model,
        _direct_configs(method, config),
        TUNING_SEED,
        device,
    )
    loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=TUNING_SEED,
    )
    batches = repeat_batches(loader)
    permutations = {
        task: permutation.to(device)
        for task, permutation in permutations_cpu.items()
    }
    threshold = StepsToThreshold()
    task_records: list[dict[str, Any]] = []
    selection_sum = 0.0
    selection_count = 0
    started = time.perf_counter()
    status = "OK"

    for step in range(1, TOTAL_STEPS + 1):
        task = ((step - 1) // STEPS_PER_TASK) + 1
        within_task_step = ((step - 1) % STEPS_PER_TASK) + 1
        inputs, targets = next(batches)
        inputs = permute_batch(
            inputs.to(device, non_blocking=True), permutations[task]
        )
        targets = targets.to(device, non_blocking=True)
        loss, accuracy = training_step(
            model=model,
            optimizer=optimizer,
            hooks=hooks,
            inputs=inputs,
            targets=targets,
        )
        if not bool(torch.isfinite(loss).item()):
            status = "EWC-DIVERGED" if method == "ewc_oracle" else "DIVERGED"
            break
        if bool(hooks["is_cap"]) and step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        threshold.update(accuracy, within_task_step)
        if step >= SELECTION_START_STEP:
            selection_sum += accuracy
            selection_count += 1

        if within_task_step == STEPS_PER_TASK:
            task_accuracy = evaluate_permutation(
                model, test_loader, permutations[task], device
            )
            task_records.append(
                {
                    "task": task,
                    "task_end_accuracy": task_accuracy,
                    "steps_to_085": threshold.reached if threshold.reached is not None else "",
                }
            )
            if hooks["on_boundary"] is not None:
                hooks["on_boundary"](model)
            threshold.reset()

    runtime = time.perf_counter() - started
    selection_metric = (
        selection_sum / selection_count
        if status == "OK" and selection_count == TOTAL_STEPS - SELECTION_START_STEP + 1
        else None
    )
    tie_rows = [
        row for row in task_records if 21 <= int(row["task"]) <= TASK_COUNT
    ]
    tie_break = (
        mean(float(row["task_end_accuracy"]) for row in tie_rows)
        if status == "OK" and len(tie_rows) == 40
        else None
    )
    config_json = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    rows = []
    for record in task_records:
        rows.append(
            {
                "method": method,
                "config_index": config_index,
                "config_json": config_json,
                **record,
                "selection_metric": selection_metric if selection_metric is not None else "",
                "tie_break_metric": tie_break if tie_break is not None else "",
                "runtime_seconds": runtime,
                "status": status,
            }
        )
    if not rows or status != "OK":
        rows.append(
            {
                "method": method,
                "config_index": config_index,
                "config_json": config_json,
                "task": "",
                "task_end_accuracy": "",
                "steps_to_085": "",
                "selection_metric": "",
                "tie_break_metric": "",
                "runtime_seconds": runtime,
                "status": status,
            }
        )
    del model, optimizer, loader, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def completed_cell(rows: Sequence[Mapping[str, Any]], method: str, index: int) -> bool:
    cell = [
        row
        for row in rows
        if str(row["method"]) == method
        and int(float(row["config_index"])) == index
    ]
    if not cell:
        return False
    if any(str(row["status"]) != "OK" for row in cell):
        return True
    task_rows = [row for row in cell if row["task"] != ""]
    return len(task_rows) == TASK_COUNT and {
        int(float(row["task"])) for row in task_rows
    } == set(range(1, TASK_COUNT + 1))


def build_tuned_artifact(
    rows: Sequence[Mapping[str, Any]],
    grids: Mapping[str, Sequence[Mapping[str, Any]]],
    amended_smoke: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        tried = []
        for index, config in enumerate(grids[method]):
            cell = [
                row
                for row in rows
                if str(row["method"]) == method
                and int(float(row["config_index"])) == index
            ]
            if not cell:
                raise RuntimeError(f"missing tuning cell {method}[{index}]")
            status = str(cell[-1]["status"])
            score_rows = [row for row in cell if row["selection_metric"] != ""]
            score = float(score_rows[-1]["selection_metric"]) if score_rows else None
            tie = float(score_rows[-1]["tie_break_metric"]) if score_rows else None
            runtime = max(float(row["runtime_seconds"]) for row in cell)
            tried.append(
                {
                    "config_index": index,
                    "config": dict(config),
                    "selection_metric": score,
                    "tie_break_metric": tie,
                    "runtime_seconds": runtime,
                    "status": status,
                }
            )
        valid = [item for item in tried if item["status"] == "OK"]
        if not valid:
            raise RuntimeError(f"no valid tuning config for {method}")
        winner = min(
            valid,
            key=lambda item: (
                -float(item["selection_metric"]),
                -float(item["tie_break_metric"]),
                int(item["config_index"]),
            ),
        )
        methods[method] = {
            "config": dict(winner["config"]),
            "selection_metric": winner["selection_metric"],
            "tie_break_metric": winner["tie_break_metric"],
            "winner_config_index": winner["config_index"],
            "tried_configs": tried,
        }

    artifact: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "amendment": UPGD_AMENDMENT,
        "p1_a3_disclosure": P1_A3_DISCLOSURE,
        "pre_amendment_upgd_smoke_gate": PRE_AMENDMENT_SMOKE,
        "amended_upgd_smoke_gate": dict(amended_smoke or {}),
        "tuning_seed": TUNING_SEED,
        "tuning_disclosure": TUNING_DISCLOSURE,
        "ewc_disclosure": EWC_DISCLOSURE,
        "adam_replay_label": REPLAY_DISCLOSURE,
        "upgd": upgd_provenance(),
        "methods": methods,
    }
    artifact["derived_configs"] = {
        "cap_ratchet": {
            **methods["cap_release"]["config"],
            "retention": "ratchet",
        },
        "cap_leak": {
            **methods["cap_release"]["config"],
            "retention": "leak",
        },
    }
    artifact["stream_a_release_arm"] = (
        "cap_release_10000"
        if cap_release_matches_core(artifact)
        else "cap_release_tuned"
    )
    return artifact


def build_report(
    artifact: Mapping[str, Any], smoke: Mapping[str, Any], runtime_seconds: float
) -> str:
    lines = [
        "# Phase 1 symmetric tuning report",
        "",
        "## Protocol header",
        "",
        f"Forced deviations: {P1_A3_DISCLOSURE}",
        f"Seed set: ({TUNING_SEED},); final aggregate seed set is {FULL_SEEDS}.",
        TUNING_DISCLOSURE,
        (
            "Settings: Stream A tasks 1-60, 5,000 steps/task, MLP-300, batch "
            "128, subset-999 boundary evaluation; selection is mean per-step "
            "training-batch accuracy over steps 100,001-300,000 inclusive."
        ),
        (
            "Shared optimizer settings: AdamW/CAP betas=(0.9,0.999), "
            "eps=1e-8, matrix-weight decay=0.01; CAP T_s=2000, box, blend, "
            "c refreshed every 100 steps. Every swept value is printed in "
            "the registered-cell and winner tables below."
        ),
        EWC_DISCLOSURE,
        REPLAY_DISCLOSURE,
        "",
        "## P1-A3 UPGD amendment",
        "",
        (
            "The original generic-README-default smoke remains an explicit "
            "INTEGRATION-FAIL; its attempts were lr=1e-5 (0.047469), "
            "lr=1e-5/3 (0.043703), and lr=3e-5 (0.062870), all finite, "
            "against the unchanged 0.90 threshold. Preserved artifacts: "
            f"`{PRE_AMENDMENT_ARCHIVE_DIR}/run_config.json` and "
            f"`{PRE_AMENDMENT_ARCHIVE_DIR}/report.md`."
        ),
        (
            "The amended smoke below uses the published task configuration "
            f"with resolved defaults `{json.dumps(upgd_provenance()['resolved_default_kwargs'], sort_keys=True)}`."
        ),
        "",
        f"UPGD integration gate: **{smoke['status']}**.",
        "",
        "| Smoke LR | Last-1,000 task-1 online accuracy | No NaN | Pass |",
        "|---:|---:|:---:|:---:|",
    ]
    for attempt in smoke.get("attempts", []):
        accuracy = attempt["last_1000_task1_online_accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{float(accuracy):.6f}"
        lines.append(
            f"| {float(attempt['lr']):.6g} | "
            f"{accuracy_text} | "
            f"{'yes' if attempt['no_nan'] else 'no'} | "
            f"{'yes' if attempt['passed'] else 'no'} |"
        )
    lines.extend(
        [
        "",
        "| Method | Winner config | Selection metric | Tie-break metric | Cell runtime (s) |",
        "|---|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        entry = artifact["methods"][method]
        winner_index = int(entry["winner_config_index"])
        runtime = float(entry["tried_configs"][winner_index]["runtime_seconds"])
        lines.append(
            f"| {display_method(method)} | "
            f"`{json.dumps(entry['config'], sort_keys=True)}` | "
            f"{float(entry['selection_metric']):.6f} | "
            f"{float(entry['tie_break_metric']):.6f} | {runtime:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Measured time for every tuning cell",
            "",
            "| Method | Grid index | Runtime (s) | Status | Selection metric |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for method in METHODS:
        for tried in artifact["methods"][method]["tried_configs"]:
            score = tried["selection_metric"]
            score_text = f"{float(score):.6f}" if score is not None else "n/a"
            lines.append(
                f"| {display_method(method)} | {int(tried['config_index'])} | "
                f"{float(tried['runtime_seconds']):.1f} | {tried['status']} | "
                f"{score_text} |"
            )
    lines.extend(
        [
            "",
            f"Stream-A release arm selected: `{artifact['stream_a_release_arm']}`.",
            f"Total measured runtime: {runtime_seconds:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/tuning")
    )
    parser.add_argument(
        "--tuned-config-path",
        type=Path,
        default=Path("results/phase1/phase1_tuned_configs.json"),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    verify_preconditions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "tuning_results.csv"
    report_path = args.output_dir / "report.md"
    config_path = args.output_dir / "run_config.json"
    mirror_path = args.output_dir / "phase1_tuned_configs.json"
    device = torch.device(args.device)
    grids = registered_grids()
    if sum(len(grid) for grid in grids.values()) != 41:
        raise RuntimeError("registered Phase-1 grid must contain exactly 41 cells")

    existing_config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    archived_pre_amendment = preserve_pre_amendment_evidence(
        config_path=config_path,
        report_path=report_path,
        existing=existing_config,
    )
    p1_a3_final_failure = bool(
        existing_config.get("amendment") == UPGD_AMENDMENT
        and existing_config.get("upgd_smoke_gate", {}).get("status")
        == "INTEGRATION-FAIL"
    )
    previous_config = (
        existing_config
        if p1_a3_final_failure
        or (args.resume and not archived_pre_amendment)
        else {}
    )
    prior_runtime = float(previous_config.get("runtime_seconds", 0.0))
    started = time.perf_counter()
    mnist_train, mnist_test = load_mnist(args.data_dir)
    prior_smoke = previous_config.get("upgd_smoke_gate", {})
    smoke = (
        prior_smoke
        if prior_smoke.get("status") == "PASS" or p1_a3_final_failure
        else run_upgd_smoke_gate(
            mnist_train=mnist_train,
            device=device,
            num_workers=args.num_workers,
        )
    )
    run_config = {
        "experiment": "phase1_tuning",
        "amendment": UPGD_AMENDMENT,
        "p1_a3_disclosure": P1_A3_DISCLOSURE,
        "pre_amendment_upgd_smoke_gate": PRE_AMENDMENT_SMOKE,
        "pre_amendment_artifacts": {
            "run_config": str(
                args.output_dir / PRE_AMENDMENT_ARCHIVE_DIR / "run_config.json"
            ),
            "report": str(
                args.output_dir / PRE_AMENDMENT_ARCHIVE_DIR / "report.md"
            ),
        },
        "created_at_utc": previous_config.get("created_at_utc", utc_now()),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "tuning_seed": TUNING_SEED,
        "scientific_settings": {
            "tasks": TASK_COUNT,
            "steps_per_task": STEPS_PER_TASK,
            "selection_steps": [SELECTION_START_STEP, TOTAL_STEPS],
            "permutation_seed_formula": "100000 + t",
            "grid_sizes": {method: len(grid) for method, grid in grids.items()},
        },
        "forced_deviations": [P1_A3_DISCLOSURE],
        "tuning_disclosure": TUNING_DISCLOSURE,
        "ewc_disclosure": EWC_DISCLOSURE,
        "adam_replay_label": REPLAY_DISCLOSURE,
        "upgd": upgd_provenance(),
        "upgd_smoke_gate": smoke,
    }
    atomic_write_json(config_path, run_config)
    if smoke["status"] != "PASS":
        report_path.write_text(
            "# Phase 1 symmetric tuning report\n\n"
            "## Protocol header\n\n"
            f"Forced deviations: {P1_A3_DISCLOSURE}\n\n"
            "Seed set: (42,).\n\n"
            "Settings: Stream-A tasks 1-3 integration smoke, 5,000 "
            "steps/task, MLP-300, batch 128.\n\n"
            "ewc_oracle receives task-boundary signals unavailable to every "
            "other arm; its results are boundary-privileged.\n\n"
            "adam_replay is an out-of-class reference (stores data; violates "
            "the streaming constraint).\n\n"
            "P1-A3 preserves the original generic-README-default failure at "
            f"`{PRE_AMENDMENT_ARCHIVE_DIR}/run_config.json` and "
            f"`{PRE_AMENDMENT_ARCHIVE_DIR}/report.md`. The amended smoke uses "
            "the official published input-permuted-MNIST task configuration.\n\n"
            "UPGD integration gate: **INTEGRATION-FAIL**. Tuning stopped.\n\n"
            "Resolved attempts: `"
            + json.dumps(smoke.get("attempts", []), sort_keys=True)
            + "`.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "INTEGRATION-FAIL: all P1-A3 UPGD smoke LRs failed; the "
            "configuration path is final and may not be rerun without a new "
            "versioned amendment"
        )
    if args.smoke_only:
        print(json.dumps(smoke, indent=2))
        return

    test_loader = make_loader(
        fixed_test_subset(mnist_test, 999),
        EVALUATION_BATCH_SIZE,
        device,
        args.num_workers,
        shuffle=False,
    )
    permutations = {
        task: fixed_permutation(784, PERMUTATION_SEED_BASE + task)
        for task in range(1, TASK_COUNT + 1)
    }
    rows = read_csv(results_path) if args.resume and results_path.exists() else []
    if not args.resume:
        write_csv(results_path, [], TUNING_FIELDS)

    for method in METHODS:
        for config_index, config in enumerate(grids[method]):
            existing_cell = [
                row
                for row in rows
                if str(row["method"]) == method
                and int(float(row["config_index"])) == config_index
            ]
            expected_config_json = json.dumps(
                dict(config), sort_keys=True, separators=(",", ":")
            )
            if existing_cell and any(
                row["config_json"] != expected_config_json
                for row in existing_cell
            ):
                raise RuntimeError(
                    f"saved tuning cell {method}[{config_index}] does not "
                    "match the frozen registered grid"
                )
            if completed_cell(rows, method, config_index):
                print(f"Skipping completed tuning cell {method}[{config_index}]")
                continue
            rows = [
                row
                for row in rows
                if not (
                    str(row["method"]) == method
                    and int(float(row["config_index"])) == config_index
                )
            ]
            print(f"Starting tuning cell {method}[{config_index}] {config}")
            rows.extend(
                run_tuning_cell(
                    method=method,
                    config_index=config_index,
                    config=config,
                    mnist_train=mnist_train,
                    test_loader=test_loader,
                    permutations_cpu=permutations,
                    device=device,
                    num_workers=args.num_workers,
                )
            )
            write_csv(results_path, rows, TUNING_FIELDS)

    artifact = build_tuned_artifact(rows, grids, smoke)
    atomic_write_json(args.tuned_config_path, artifact)
    # Required output mirror; downstream programs read only the canonical path.
    atomic_write_json(mirror_path, artifact)
    runtime = prior_runtime + (time.perf_counter() - started)
    run_config["completed_at_utc"] = utc_now()
    run_config["runtime_seconds"] = runtime
    run_config["canonical_tuned_config_path"] = str(args.tuned_config_path)
    atomic_write_json(config_path, run_config)
    report = build_report(artifact, smoke, runtime)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
