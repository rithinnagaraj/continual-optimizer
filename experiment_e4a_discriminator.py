"""E4a discriminator: optimizer-state staleness and onset-kick diagnosis.

The discriminator reuses the original synthetic release rows and runs plain
AdamW plus CAP release with a moments-only B-state refresh.  An optional
release warm-up arm tests the F5 clamp burn-in candidate.  It writes a
discriminator report and the ``regrade.json`` artifact consumed by E4b.

The registered D statistic requires L=1000, and the 30/45-run budget requires
exactly three lengths. The discriminator uses (1000, 4000, 20000). An earlier
protocol grid listed L=0 instead of L=1000; reports and run configurations
record that replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiment_e2 import prefix_preliminary_verdict, read_csv, write_csv
from experiment_e3 import (
    PermutedDataset,
    adamw_parameter_groups,
    fixed_permutation,
    load_mnist,
    make_loader,
    repeat_batches,
    cap_parameter_groups,
    seed_everything,
)
from experiment_e4a import (
    ADAM_EPS,
    A_TEST_SUBSET_SEED,
    BATCH_SIZE,
    BETAS,
    B_EVALUATE_EVERY,
    B_PERMUTATION_SEED,
    CONFIG_ACCURACY_FLOOR,
    C_REFRESH_STEPS,
    DECAY_MODE,
    EVALUATION_BATCH_SIZE,
    FULL_SEEDS,
    GRADE_DELTA,
    K,
    LEARNING_RATE,
    PHASE_A_STEPS,
    PHASE_B_STEPS,
    PROTECTION,
    T_A,
    T_M_RELEASE,
    T_S,
    TRACK_BIND_STATS,
    WEIGHT_DECAY,
    atomic_write_json,
    evaluate_and_resume_training,
    fixed_test_subset,
    grade_e4a,
    protected_bind_counts,
    read_optional_csv,
    read_optional_json,
    refresh_global_c,
    utc_now,
    verify_preconditions,
)
from model import MLP300
from cap_adam_experimental import CAPAdam


REQUIRED_ARMS = ("adamw", "release_refresh")
OPTIONAL_ARM = "release_warmup"
SILENCE_LENGTHS = (1_000, 4_000, 20_000)
REFRESH_ITERATIONS = 200
WARMUP_CLAMP_DISABLED_STEPS = 2_000
ONSET_STEPS = 100
ONSET_BUCKET_SIZE = 10
ADAMW_CHANNEL_THRESHOLD = 0.03
ADAMW_RELATIVE_MARGIN = 0.02

RESULT_FIELDS = (
    "arm",
    "L",
    "seed",
    "a_end",
    "a_after_silence",
    "a_final",
    "min_a_during_b",
    "silence_bind_rate",
    "status",
)
ONSET_FIELDS = (
    "arm",
    "L",
    "seed",
    "b_step_start",
    "b_step_end",
    "mean_abs_delta_theta",
)
TRACE_FIELDS = (
    "arm",
    "L",
    "seed",
    "b_step",
    "a_test_accuracy",
)


def active_arms(include_release_warmup: bool) -> tuple[str, ...]:
    return REQUIRED_ARMS + ((OPTIONAL_ARM,) if include_release_warmup else ())


def create_optimizer(
    model: nn.Module, arm: str
) -> tuple[torch.optim.Optimizer, bool]:
    if arm == "adamw":
        return (
            torch.optim.AdamW(
                adamw_parameter_groups(model, WEIGHT_DECAY),
                lr=LEARNING_RATE,
                betas=BETAS,
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            ),
            False,
        )
    if arm not in ("release_refresh", "release_warmup"):
        raise ValueError(f"unknown discriminator arm: {arm}")
    return (
        CAPAdam(
            cap_parameter_groups(model, WEIGHT_DECAY),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
            k=K,
            T_s=T_S,
            T_a=T_A,
            T_m=T_M_RELEASE,
            passive=False,
            protection=PROTECTION,
            retention="release",
            decay_mode=DECAY_MODE,
            track_bind_stats=TRACK_BIND_STATS,
        ),
        True,
    )


def cap_step_with_optional_clamp_disable(
    optimizer: CAPAdam, disable_clamp: bool
) -> None:
    """Take one CAP step, optionally suppressing only box projection."""
    if not disable_clamp:
        optimizer.step()
        return
    prior_values = [
        ("disable_box_clamp" in group, group.get("disable_box_clamp"))
        for group in optimizer.param_groups
    ]
    try:
        for group in optimizer.param_groups:
            if bool(group["protect"]):
                group["disable_box_clamp"] = True
        optimizer.step()
    finally:
        for group, (was_present, prior_value) in zip(
            optimizer.param_groups, prior_values
        ):
            if was_present:
                group["disable_box_clamp"] = prior_value
            else:
                group.pop("disable_box_clamp", None)


def mean_abs_parameter_change(
    model: nn.Module, before: Sequence[Tensor]
) -> float:
    absolute_sum = 0.0
    numel = 0
    with torch.no_grad():
        for parameter, previous in zip(model.parameters(), before):
            absolute_sum += float((parameter.detach() - previous).abs().sum().item())
            numel += parameter.numel()
    return absolute_sum / max(numel, 1)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
    optimizer_step: int,
    *,
    is_cap: bool,
    disable_clamp: bool = False,
    measure_delta: bool = False,
) -> tuple[float, Optional[float]]:
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    before = (
        [parameter.detach().clone() for parameter in model.parameters()]
        if measure_delta
        else []
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    if is_cap:
        cap_step_with_optional_clamp_disable(optimizer, disable_clamp)
        if optimizer_step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
    else:
        optimizer.step()
    delta = mean_abs_parameter_change(model, before) if measure_delta else None
    return float(loss.item()), delta


def synthetic_silence_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    *,
    is_cap: bool,
) -> tuple[int, int]:
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    if not is_cap:
        return 0, 0
    if optimizer_step % C_REFRESH_STEPS == 0:
        refresh_global_c(optimizer)
    return protected_bind_counts(optimizer)


def refresh_adam_moments_only(
    model: nn.Module,
    optimizer: CAPAdam,
    batches,
    device: torch.device,
    iterations: Optional[int] = None,
) -> None:
    """Update only Adam m/v from B gradients; leave every other state fixed."""
    if not isinstance(optimizer, CAPAdam):
        raise TypeError("moments-only refresh requires CAPAdam")
    if iterations is None:
        iterations = REFRESH_ITERATIONS
    for _ in range(iterations):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        with torch.no_grad():
            for group in optimizer.param_groups:
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        raise RuntimeError(
                            "refresh requires a gradient tensor for every parameter"
                        )
                    if gradient.is_sparse:
                        raise RuntimeError("moments-only refresh rejects sparse gradients")
                    state = optimizer.state.get(parameter)
                    if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
                        raise RuntimeError("Adam moments are not initialized before refresh")
                    effective_gradient = (
                        -gradient if bool(group.get("maximize", False)) else gradient
                    )
                    state["exp_avg"].mul_(beta1).add_(
                        effective_gradient, alpha=1.0 - beta1
                    )
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        effective_gradient,
                        effective_gradient,
                        value=1.0 - beta2,
                    )


def run_one(
    *,
    arm: str,
    silence_length: int,
    seed: int,
    mnist_train,
    a_test_loader,
    device: torch.device,
    num_workers: int,
) -> tuple[dict, list[dict], list[dict]]:
    if arm not in ("adamw", "release_refresh", "release_warmup"):
        raise ValueError(f"unknown discriminator arm: {arm}")
    if silence_length not in SILENCE_LENGTHS:
        raise ValueError(f"unknown discriminator silence length: {silence_length}")

    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer, is_cap = create_optimizer(model, arm)
    model.train()
    a_loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    a_batches = repeat_batches(a_loader)
    optimizer_step = 0
    for a_step in range(1, PHASE_A_STEPS + 1):
        optimizer_step += 1
        train_step(
            model,
            optimizer,
            next(a_batches),
            device,
            optimizer_step,
            is_cap=is_cap,
            disable_clamp=(
                arm == OPTIONAL_ARM and a_step <= WARMUP_CLAMP_DISABLED_STEPS
            ),
        )

    a_end = evaluate_and_resume_training(model, a_test_loader, device)
    if a_end < CONFIG_ACCURACY_FLOOR:
        return (
            {
                "arm": arm,
                "L": silence_length,
                "seed": seed,
                "a_end": a_end,
                "a_after_silence": "",
                "a_final": "",
                "min_a_during_b": "",
                "silence_bind_rate": "",
                "status": "CONFIG-SUSPECT",
            },
            [],
            [],
        )

    silence_bind_count = 0
    silence_bind_numel = 0
    for _ in range(silence_length):
        optimizer_step += 1
        count, numel = synthetic_silence_step(
            model,
            optimizer,
            optimizer_step,
            is_cap=is_cap,
        )
        silence_bind_count += count
        silence_bind_numel += numel
    a_after_silence = evaluate_and_resume_training(model, a_test_loader, device)

    task_b = PermutedDataset(
        mnist_train, fixed_permutation(784, B_PERMUTATION_SEED)
    )
    if arm == "release_refresh":
        refresh_loader = make_loader(
            task_b,
            BATCH_SIZE,
            device,
            num_workers,
            shuffle=True,
            seed=seed + 20_000,
        )
        refresh_adam_moments_only(
            model,
            optimizer,
            repeat_batches(refresh_loader),
            device,
        )

    # Rebuild the B loader after refresh so every arm's 3,000 parameter-update
    # steps see the identical seeded B-batch sequence.
    b_loader = make_loader(
        task_b,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed + 20_000,
    )
    b_batches = repeat_batches(b_loader)
    onset_rows: list[dict] = []
    trace_rows: list[dict] = []
    bucket_deltas: list[float] = []
    minimum_a = math.inf
    a_final = math.nan
    for b_step in range(1, PHASE_B_STEPS + 1):
        optimizer_step += 1
        _, delta = train_step(
            model,
            optimizer,
            next(b_batches),
            device,
            optimizer_step,
            is_cap=is_cap,
            measure_delta=b_step <= ONSET_STEPS,
        )
        if delta is not None:
            bucket_deltas.append(delta)
            if b_step % ONSET_BUCKET_SIZE == 0:
                onset_rows.append(
                    {
                        "arm": arm,
                        "L": silence_length,
                        "seed": seed,
                        "b_step_start": b_step - ONSET_BUCKET_SIZE + 1,
                        "b_step_end": b_step,
                        "mean_abs_delta_theta": mean(bucket_deltas),
                    }
                )
                bucket_deltas.clear()
        if b_step % B_EVALUATE_EVERY == 0 or b_step == PHASE_B_STEPS:
            accuracy = evaluate_and_resume_training(model, a_test_loader, device)
            minimum_a = min(minimum_a, accuracy)
            a_final = accuracy
            trace_rows.append(
                {
                    "arm": arm,
                    "L": silence_length,
                    "seed": seed,
                    "b_step": b_step,
                    "a_test_accuracy": accuracy,
                }
            )

    return (
        {
            "arm": arm,
            "L": silence_length,
            "seed": seed,
            "a_end": a_end,
            "a_after_silence": a_after_silence,
            "a_final": a_final,
            "min_a_during_b": minimum_a,
            "silence_bind_rate": (
                silence_bind_count / max(silence_bind_numel, 1)
                if is_cap
                else ""
            ),
            "status": "ok",
        },
        onset_rows,
        trace_rows,
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def cell_values(
    rows: Sequence[Mapping],
    *,
    arm_field: str,
    arm: str,
    construction: Optional[str],
    silence_length: int,
    seeds: Sequence[int],
) -> list[float]:
    values_by_seed: dict[int, float] = {}
    for row in rows:
        if str(row[arm_field]) != arm or int(float(row["L"])) != silence_length:
            continue
        if construction is not None and str(row["construction"]) != construction:
            continue
        if str(row.get("status", "ok")) != "ok":
            continue
        seed = int(float(row["seed"]))
        if seed not in set(seeds):
            continue
        if seed in values_by_seed:
            raise ValueError(
                f"duplicate result for {arm_field}={arm}, construction={construction}, "
                f"L={silence_length}, seed={seed}"
            )
        values_by_seed[seed] = float(row["a_final"])
    missing = [seed for seed in seeds if seed not in values_by_seed]
    if missing:
        raise RuntimeError(
            f"missing ok rows for {arm_field}={arm}, construction={construction}, "
            f"L={silence_length}, seeds={missing}"
        )
    return [values_by_seed[seed] for seed in seeds]


def damage(
    rows: Sequence[Mapping],
    *,
    arm_field: str,
    arm: str,
    construction: Optional[str],
    seeds: Sequence[int],
) -> float:
    low = cell_values(
        rows,
        arm_field=arm_field,
        arm=arm,
        construction=construction,
        silence_length=1_000,
        seeds=seeds,
    )
    high = cell_values(
        rows,
        arm_field=arm_field,
        arm=arm,
        construction=construction,
        silence_length=20_000,
        seeds=seeds,
    )
    return mean(low) - mean(high)


def grade_discriminator(
    discriminator_rows: Sequence[Mapping],
    existing_e4a_rows: Sequence[Mapping],
    seeds: Sequence[int],
) -> dict:
    original_grade = grade_e4a(existing_e4a_rows)
    damages = {
        "adamw": damage(
            discriminator_rows,
            arm_field="arm",
            arm="adamw",
            construction=None,
            seeds=seeds,
        ),
        "release_refresh": damage(
            discriminator_rows,
            arm_field="arm",
            arm="release_refresh",
            construction=None,
            seeds=seeds,
        ),
        "ratchet_synthetic": damage(
            existing_e4a_rows,
            arm_field="policy",
            arm="ratchet",
            construction="synthetic",
            seeds=seeds,
        ),
        "release_synthetic": damage(
            existing_e4a_rows,
            arm_field="policy",
            arm="release",
            construction="synthetic",
            seeds=seeds,
        ),
        "ratchet_realistic": damage(
            existing_e4a_rows,
            arm_field="policy",
            arm="ratchet",
            construction="realistic",
            seeds=seeds,
        ),
        "release_realistic": damage(
            existing_e4a_rows,
            arm_field="policy",
            arm="release",
            construction="realistic",
            seeds=seeds,
        ),
    }
    if any(str(row["arm"]) == OPTIONAL_ARM for row in discriminator_rows):
        damages[OPTIONAL_ARM] = damage(
            discriminator_rows,
            arm_field="arm",
            arm=OPTIONAL_ARM,
            construction=None,
            seeds=seeds,
        )

    channel_is_staleness = damages["adamw"] >= ADAMW_CHANNEL_THRESHOLD
    synthetic_limit = damages["adamw"] + ADAMW_RELATIVE_MARGIN
    amended_criteria = {
        "ratchet_synthetic": damages["ratchet_synthetic"] <= synthetic_limit,
        "release_synthetic": damages["release_synthetic"] <= synthetic_limit,
        "ratchet_realistic": damages["ratchet_realistic"] <= GRADE_DELTA,
        "release_realistic": damages["release_realistic"] <= GRADE_DELTA,
    }
    amended_treatment_ok = all(amended_criteria.values())

    if not bool(original_grade["control_ok"]):
        verdict = "SILENCE-CONTROL-ABSENT"
    elif not channel_is_staleness:
        verdict = "PENDING-PROTECTION-SILENCE-INVESTIGATION"
    elif amended_treatment_ok:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "damages": damages,
        "adamw_channel_threshold": ADAMW_CHANNEL_THRESHOLD,
        "channel_is_optimizer_state_staleness": channel_is_staleness,
        "amendment_applied": channel_is_staleness,
        "synthetic_adamw_relative_limit": synthetic_limit,
        "realistic_limit": GRADE_DELTA,
        "amended_criteria": amended_criteria,
        "amended_treatment_ok": amended_treatment_ok,
        "original_control_ok": bool(original_grade["control_ok"]),
        "original_release_bug": bool(original_grade["release_bug"]),
        "refresh_pattern_note": (
            "D(release_refresh) and D(release) are reported without a binary "
            "causal verdict because approximately-zero and much-greater-than "
            "were not assigned numerical tolerances."
        ),
    }


def build_regrade_artifact(
    grade: Mapping,
    seeds: Sequence[int],
    e4a_results_path: Path,
    discriminator_results_path: Path,
) -> dict:
    displayed_verdict = prefix_preliminary_verdict(
        str(grade["verdict"]), seeds, FULL_SEEDS
    )
    return {
        "experiment": "E4a discriminator re-grade",
        "created_at_utc": utc_now(),
        "verdict": displayed_verdict,
        "raw_verdict": str(grade["verdict"]),
        "seeds": list(seeds),
        "source_artifacts": {
            "e4a_results": str(e4a_results_path),
            "discriminator_results": str(discriminator_results_path),
        },
        "protocol_clarification": (
            "Used L=(1000,4000,20000): D requires L=1000 and the registered "
            "30/45-run count requires exactly three lengths; the earlier L=0 "
            "entry is treated as a transcription error."
        ),
        "grade": {
            key: value
            for key, value in grade.items()
            if key != "verdict"
        },
    }


def build_report(
    discriminator_rows: Sequence[Mapping],
    onset_rows: Sequence[Mapping],
    existing_e4a_rows: Sequence[Mapping],
    grade: Mapping,
    seeds: Sequence[int],
    include_release_warmup: bool,
    runtime_seconds: float,
) -> str:
    verdict = prefix_preliminary_verdict(
        str(grade["verdict"]), seeds, FULL_SEEDS
    )
    lines = [
        "# E4a discriminator report",
        "",
        "## Protocol header",
        "",
        f"Seeds: {tuple(seeds)}.",
        (
            "Protocol clarification: the earlier grid named L=0, but D(arm) "
            "requires L=1000 and the stated cost requires three lengths. This "
            "run therefore uses L=(1000,4000,20000) and records that "
            "clarification in regrade.json."
        ),
        (
            f"Arms run: {active_arms(include_release_warmup)}; synthetic only; "
            f"A={PHASE_A_STEPS} steps; B={PHASE_B_STEPS} steps; "
            f"release refresh={REFRESH_ITERATIONS} moments-only B-gradient "
            "iterations; onset kick=mean absolute parameter change over B "
            "steps 1-100 in 10-step buckets."
        ),
        "Existing release/ratchet E4a rows were reused and not modified.",
        "",
        "## Pre-registered outcomes and re-grade",
        "",
        (
            f"- D(adamw)={float(grade['damages']['adamw']):.4f}; channel "
            f"threshold >= {ADAMW_CHANNEL_THRESHOLD:.2f}; optimizer-state "
            f"staleness={grade['channel_is_optimizer_state_staleness']}."
        ),
        (
            f"- D(release_refresh)={float(grade['damages']['release_refresh']):.4f}; "
            f"D(existing release)={float(grade['damages']['release_synthetic']):.4f}."
        ),
        f"- {grade['refresh_pattern_note']}",
    ]
    if bool(grade["amendment_applied"]):
        lines.extend(
            [
                (
                    "- Written amendment applied on re-grade: synthetic ratchet "
                    "and release damage must be <= D(adamw)+0.02 = "
                    f"{float(grade['synthetic_adamw_relative_limit']):.4f}; "
                    f"realistic damage remains <= {GRADE_DELTA:.2f}."
                ),
                (
                    "- Amended criteria: "
                    + ", ".join(
                        f"{name}={passed}"
                        for name, passed in grade["amended_criteria"].items()
                    )
                    + f"; treatment passes={grade['amended_treatment_ok']}."
                ),
            ]
        )
    else:
        lines.append(
            "- D(adamw) is below 0.03: the staleness explanation is rejected; "
            "CAP-specific protection-silence investigation is required before "
            "E4b interpretation."
        )
    lines.extend(
        [
            f"- Re-grade verdict: **{verdict}**.",
            "",
            "| Damage statistic | Value |",
            "|---|---:|",
        ]
    )
    for name, value in grade["damages"].items():
        lines.append(f"| D({name}) | {float(value):.4f} |")

    lines.extend(
        [
            "",
            "## New-arm final A accuracy",
            "",
            "| Arm | L | a_final mean +/- std | silence bind rate mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in active_arms(include_release_warmup):
        for silence_length in SILENCE_LENGTHS:
            cell = [
                row
                for row in discriminator_rows
                if str(row["arm"]) == arm
                and int(float(row["L"])) == silence_length
                and str(row["status"]) == "ok"
            ]
            finals = [float(row["a_final"]) for row in cell]
            bind_rates = [
                float(row["silence_bind_rate"])
                for row in cell
                if row["silence_bind_rate"] != ""
            ]
            final_mean, final_std = mean_std(finals)
            bind_text = f"{mean(bind_rates):.6g}" if bind_rates else "n/a"
            lines.append(
                f"| {arm} | {silence_length} | {final_mean:.4f} +/- "
                f"{final_std:.4f} | {bind_text} |"
            )

    lines.extend(
        [
            "",
            "## B-onset mean absolute parameter change",
            "",
            "| Arm | L | B steps | mean +/- std over seeds |",
            "|---|---:|---:|---:|",
        ]
    )
    onset_groups: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for row in onset_rows:
        onset_groups[
            (
                str(row["arm"]),
                int(float(row["L"])),
                int(float(row["b_step_start"])),
                int(float(row["b_step_end"])),
            )
        ].append(float(row["mean_abs_delta_theta"]))
    for key in sorted(onset_groups):
        arm, silence_length, start, end = key
        value_mean, value_std = mean_std(onset_groups[key])
        lines.append(
            f"| {arm} | {silence_length} | {start}-{end} | "
            f"{value_mean:.6g} +/- {value_std:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Runtime: {runtime_seconds:.1f} seconds ({runtime_seconds / 3600.0:.3f} hours).",
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing_e4a(
    rows: Sequence[Mapping], seeds: Sequence[int]
) -> None:
    for policy in ("leak", "ratchet", "release"):
        for construction in ("synthetic", "realistic"):
            for silence_length in (1_000, 20_000):
                cell_values(
                    rows,
                    arm_field="policy",
                    arm=policy,
                    construction=construction,
                    silence_length=silence_length,
                    seeds=seeds,
                )


def validate_discriminator_coverage(
    rows: Sequence[Mapping], arms: Sequence[str], seeds: Sequence[int]
) -> None:
    expected = {
        (arm, silence_length, seed)
        for arm in arms
        for silence_length in SILENCE_LENGTHS
        for seed in seeds
    }
    observed = {
        (str(row["arm"]), int(float(row["L"])), int(float(row["seed"])))
        for row in rows
    }
    if observed != expected:
        raise RuntimeError(
            f"incomplete discriminator coverage; missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )


def discriminator_key(row: Mapping) -> tuple[str, int, int]:
    return (
        str(row["arm"]),
        int(float(row["L"])),
        int(float(row["seed"])),
    )


def repair_incomplete_resume_cells(
    results: list[dict], onset_rows: list[dict], trace_rows: list[dict]
) -> None:
    """Drop interrupted cells so atomic resume recomputes the whole cell."""
    invalid = set()
    for result in results:
        key = discriminator_key(result)
        onset_count = sum(discriminator_key(row) == key for row in onset_rows)
        trace_count = sum(discriminator_key(row) == key for row in trace_rows)
        if str(result["status"]) == "CONFIG-SUSPECT":
            expected_onset = 0
            expected_trace = 0
        else:
            expected_onset = ONSET_STEPS // ONSET_BUCKET_SIZE
            expected_trace = PHASE_B_STEPS // B_EVALUATE_EVERY
        if onset_count != expected_onset or trace_count != expected_trace:
            invalid.add(key)
    if not invalid:
        return
    results[:] = [row for row in results if discriminator_key(row) not in invalid]
    onset_rows[:] = [
        row for row in onset_rows if discriminator_key(row) not in invalid
    ]
    trace_rows[:] = [
        row for row in trace_rows if discriminator_key(row) not in invalid
    ]


def base_run_config(
    args: argparse.Namespace, seeds: Sequence[int], arms: Sequence[str]
) -> dict:
    return {
        "experiment": "E4a discriminator",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seeds": list(seeds),
        "arms": list(arms),
        "silence_lengths": list(SILENCE_LENGTHS),
        "protocol_clarification": (
            "L=1000 replaces the earlier L=0 because D requires L=1000 and "
            "the stated 30/45-run count requires three lengths."
        ),
        "refresh_iterations": REFRESH_ITERATIONS,
        "warmup_clamp_disabled_steps": WARMUP_CLAMP_DISABLED_STEPS,
        "onset_steps": ONSET_STEPS,
        "onset_bucket_size": ONSET_BUCKET_SIZE,
        "adamw_channel_threshold": ADAMW_CHANNEL_THRESHOLD,
        "adamw_relative_margin": ADAMW_RELATIVE_MARGIN,
        "runtime_seconds": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FULL_SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--e4a-results-path",
        type=Path,
        default=Path("results/e4a/e4a_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/e4a/discriminator"),
    )
    parser.add_argument("--include-release-warmup", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        parser.error("seeds must be nonempty and unique")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    verify_preconditions()
    if not args.e4a_results_path.exists():
        raise FileNotFoundError(
            f"existing five-seed E4a CSV is required: {args.e4a_results_path}"
        )
    existing_e4a_rows = read_csv(args.e4a_results_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "discriminator_results.csv"
    onset_path = args.output_dir / "discriminator_onset.csv"
    trace_path = args.output_dir / "discriminator_trace.csv"
    report_path = args.output_dir / "report.md"
    config_path = args.output_dir / "run_config.json"
    arms = active_arms(args.include_release_warmup)

    existing_config = read_optional_json(config_path)
    if args.summarize_only and existing_config:
        seeds = tuple(int(seed) for seed in existing_config["seeds"])
        arms = tuple(str(arm) for arm in existing_config["arms"])
    else:
        seeds = args.seeds
    validate_existing_e4a(existing_e4a_rows, seeds)
    full_seed_set_ran = (
        len(seeds) == len(FULL_SEEDS) and set(seeds) == set(FULL_SEEDS)
    )
    regrade_path = (
        args.e4a_results_path.parent / "regrade.json"
        if full_seed_set_ran
        else args.output_dir / "regrade.preliminary.json"
    )
    if args.resume and existing_config:
        if set(seeds) != {int(seed) for seed in existing_config["seeds"]}:
            raise ValueError("resume seed set does not match run_config.json")
        if tuple(existing_config["arms"]) != tuple(arms):
            raise ValueError("resume arm set does not match run_config.json")
        config = existing_config
    else:
        config = base_run_config(args, seeds, arms)

    prior_runtime = float(config.get("runtime_seconds", 0.0))
    invocation_start = time.perf_counter()
    if args.summarize_only:
        results = read_csv(result_path)
        onset_rows = read_csv(onset_path)
        trace_rows = read_csv(trace_path)
    else:
        results = read_optional_csv(result_path) if args.resume else []
        onset_rows = read_optional_csv(onset_path) if args.resume else []
        trace_rows = read_optional_csv(trace_path) if args.resume else []
        if args.resume:
            repair_incomplete_resume_cells(results, onset_rows, trace_rows)
            write_csv(result_path, results, RESULT_FIELDS)
            write_csv(onset_path, onset_rows, ONSET_FIELDS)
            write_csv(trace_path, trace_rows, TRACE_FIELDS)
        completed = {
            (str(row["arm"]), int(float(row["L"])), int(float(row["seed"])))
            for row in results
        }
        device = torch.device(args.device)
        mnist_train, mnist_test = load_mnist(args.data_dir)
        a_test_loader = make_loader(
            fixed_test_subset(mnist_test, A_TEST_SUBSET_SEED),
            EVALUATION_BATCH_SIZE,
            device,
            args.num_workers,
            shuffle=False,
        )
        for arm in arms:
            for silence_length in SILENCE_LENGTHS:
                for seed in seeds:
                    key = (arm, silence_length, seed)
                    if key in completed:
                        print(f"Skipping completed discriminator cell {key}")
                        continue
                    print(f"Starting discriminator cell {key}")
                    result, onset, trace = run_one(
                        arm=arm,
                        silence_length=silence_length,
                        seed=seed,
                        mnist_train=mnist_train,
                        a_test_loader=a_test_loader,
                        device=device,
                        num_workers=args.num_workers,
                    )
                    results.append(result)
                    onset_rows.extend(onset)
                    trace_rows.extend(trace)
                    completed.add(key)
                    write_csv(result_path, results, RESULT_FIELDS)
                    write_csv(onset_path, onset_rows, ONSET_FIELDS)
                    write_csv(trace_path, trace_rows, TRACE_FIELDS)
                    config["updated_at_utc"] = utc_now()
                    atomic_write_json(config_path, config)

    validate_discriminator_coverage(results, arms, seeds)
    total_runtime = prior_runtime + (time.perf_counter() - invocation_start)
    grade = grade_discriminator(results, existing_e4a_rows, seeds)
    report = build_report(
        results,
        onset_rows,
        existing_e4a_rows,
        grade,
        seeds,
        OPTIONAL_ARM in arms,
        total_runtime,
    )
    report_path.write_text(report, encoding="utf-8")
    regrade = build_regrade_artifact(
        grade, seeds, args.e4a_results_path, result_path
    )
    atomic_write_json(regrade_path, regrade)
    config["updated_at_utc"] = utc_now()
    config["runtime_seconds"] = total_runtime
    config["grade"] = grade
    config["regrade_path"] = str(regrade_path)
    atomic_write_json(config_path, config)
    print(report)
    print(f"Discriminator artifacts written to {args.output_dir.resolve()}")
    print(f"E4a re-grade written to {regrade_path.resolve()}")


if __name__ == "__main__":
    main()
