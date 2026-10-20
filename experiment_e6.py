"""E6 idle-GPU overhead and optimizer-state benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet18

from experiment_e2 import read_csv, write_csv
from experiment_e3 import cap_parameter_groups, seed_everything
from experiment_e4a import atomic_write_json, utc_now
from model import MLP300
from cap_adam_experimental import CAPAdam


WORKLOADS = ("mlp", "resnet18", "gpt2")
ARMS = ("adamw", "cap_diag_on", "cap_diag_off")
EXPERIMENT_VERSION = "e6-final-contest-blocked-v1"
# Frozen final CAP configuration (cap_mechanistic/cap_mech/configs/final_cap.json).
LEARNING_RATE = 1.0 / 300.0
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
K = 7e-5
T_S = 2_000.0
T_A = 2_000.0
T_M = 1_000_000.0
REPEATS = 3
WARMUP_STEPS = 50
TIMED_STEPS = 200

RESULT_FIELDS = (
    "workload",
    "arm",
    "repeat",
    "full_step_ms",
    "opt_step_ms",
    "max_memory_bytes",
    "parameter_count",
    "state_bytes",
    "state_bytes_excluding_bind",
    "expected_state_bytes",
    "state_ratio",
    "status",
    "error",
)


class GPT2Workload(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        return self.model(input_ids=input_ids, labels=labels).loss


def driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip().splitlines()[0]
    except Exception:
        return "unavailable"


def system_record(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver_version(),
        "cudnn": torch.backends.cudnn.version(),
    }


def fixed_batch(workload: str, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(60_000 + WORKLOADS.index(workload))
    if workload == "mlp":
        return (
            torch.randn(128, 1, 28, 28, generator=generator, device=device),
            torch.randint(0, 10, (128,), generator=generator, device=device),
        )
    if workload == "resnet18":
        return (
            torch.randn(128, 3, 32, 32, generator=generator, device=device),
            torch.randint(0, 10, (128,), generator=generator, device=device),
        )
    if workload == "gpt2":
        tokens = torch.randint(
            0, 50_257, (8, 512), generator=generator, device=device
        )
        return tokens, tokens
    raise ValueError(f"unknown E6 workload: {workload}")


def build_model(workload: str, device: torch.device) -> nn.Module:
    if workload == "mlp":
        return MLP300(input_units=784, output_units=10).to(device)
    if workload == "resnet18":
        return resnet18(num_classes=10).to(device)
    if workload == "gpt2":
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as error:
            raise ModuleNotFoundError("transformers") from error
        return GPT2Workload(GPT2LMHeadModel(GPT2Config())).to(device)
    raise ValueError(f"unknown E6 workload: {workload}")


def build_optimizer(model: nn.Module, arm: str):
    if arm == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
    if arm in {"cap_diag_on", "cap_diag_off"}:
        return CAPAdam(
            cap_parameter_groups(model, WEIGHT_DECAY),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
            k=K,
            T_s=T_S,
            T_a=T_A,
            T_m=T_M,
            passive=False,
            protection="box",
            retention="release",
            release_update="contest_blocked",
            box_geometry="coordinate",
            momentum_response="none",
            decay_mode="zero",
            track_bind_stats=arm == "cap_diag_on",
        )
    raise ValueError(f"unknown E6 arm: {arm}")


def workload_loss(model: nn.Module, workload: str, batch: tuple[Tensor, Tensor]) -> Tensor:
    inputs, targets = batch
    if workload == "gpt2":
        return model(inputs, targets)
    return F.cross_entropy(model(inputs), targets)


def optimizer_state_bytes(
    optimizer: torch.optim.Optimizer, *, exclude_bind: bool
) -> int:
    total = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor) and not (exclude_bind and key == "last_bind"):
                total += value.numel() * value.element_size()
    return total


def state_audit(
    optimizer: torch.optim.Optimizer, arm: str, parameter_count: int
) -> dict[str, Any]:
    measured = optimizer_state_bytes(optimizer, exclude_bind=False)
    excluding_bind = optimizer_state_bytes(optimizer, exclude_bind=True)
    if arm == "adamw":
        expected = 8 * parameter_count
        audited = measured
    else:
        expected = 16 * parameter_count
        audited = excluding_bind
    ratio = audited / expected
    if abs(ratio - 1.0) > 0.01:
        raise AssertionError(
            f"{arm} state audit ratio {ratio:.6f} is outside the 1% tolerance"
        )
    if arm == "cap_diag_on":
        total_expected = 17 * parameter_count
        if abs(measured / total_expected - 1.0) > 0.01:
            raise AssertionError("cap_diag_on bool-state audit is outside 1%")
    return {
        "state_bytes": measured,
        "state_bytes_excluding_bind": excluding_bind,
        "expected_state_bytes": expected,
        "state_ratio": ratio,
    }


def run_repeat(
    *,
    workload: str,
    arm: str,
    repeat: int,
    batch: tuple[Tensor, Tensor],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(90_000 + repeat)
    model = build_model(workload, device)
    optimizer = build_optimizer(model, arm)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(WARMUP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = workload_loss(model, workload, batch)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        block_start = time.perf_counter()
        optimizer_times = []
        for _ in range(TIMED_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = workload_loss(model, workload, batch)
            loss.backward()
            torch.cuda.synchronize(device)
            optimizer_start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_times.append(time.perf_counter() - optimizer_start)
        torch.cuda.synchronize(device)
        block_seconds = time.perf_counter() - block_start
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite benchmark loss")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        audit = state_audit(optimizer, arm, parameter_count)
        return {
            "workload": workload,
            "arm": arm,
            "repeat": repeat,
            "full_step_ms": 1_000.0 * block_seconds / TIMED_STEPS,
            "opt_step_ms": 1_000.0 * statistics.mean(optimizer_times),
            "max_memory_bytes": torch.cuda.max_memory_allocated(device),
            "parameter_count": parameter_count,
            **audit,
            "status": "OK",
            "error": "",
        }
    except Exception as error:
        status = "ERROR"
        return {
            "workload": workload,
            "arm": arm,
            "repeat": repeat,
            "status": status,
            "error": repr(error),
        }
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for workload in WORKLOADS:
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if str(row["workload"]) == workload
                and str(row["arm"]) == arm
                and str(row["status"]) == "OK"
            ]
            key = f"{workload}/{arm}"
            repeat_ids = {int(float(row["repeat"])) for row in selected}
            if len(selected) != REPEATS or repeat_ids != set(range(REPEATS)):
                statuses = sorted(
                    {
                        str(row["status"])
                        for row in rows
                        if str(row["workload"]) == workload and str(row["arm"]) == arm
                    }
                )
                output[key] = {"complete": False, "statuses": statuses}
                continue
            output[key] = {
                "complete": True,
                "full_step_ms": statistics.median(float(row["full_step_ms"]) for row in selected),
                "opt_step_ms": statistics.mean(float(row["opt_step_ms"]) for row in selected),
                "max_memory_bytes": max(int(float(row["max_memory_bytes"])) for row in selected),
                "state_ratio": statistics.mean(float(row["state_ratio"]) for row in selected),
            }
    return output


def grade(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    overheads = {}
    for workload in WORKLOADS:
        baseline = summary[f"{workload}/adamw"]
        cap = summary[f"{workload}/cap_diag_off"]
        if baseline.get("complete") and cap.get("complete"):
            overheads[workload] = cap["full_step_ms"] / baseline["full_step_ms"] - 1.0
        else:
            overheads[workload] = None
    state_records = [
        record
        for key, record in summary.items()
        if "/cap_diag_off" in key and record.get("complete")
    ]
    state_ok = bool(state_records) and all(
        abs(float(record["state_ratio"]) - 1.0) <= 0.01
        for record in state_records
    )
    if overheads["gpt2"] is None:
        verdict = "SKIPPED-DEPENDENCY"
    elif overheads["resnet18"] is None:
        verdict = "INCOMPLETE"
    else:
        maximum = max(float(overheads["resnet18"]), float(overheads["gpt2"]))
        if maximum <= 0.05 and state_ok:
            verdict = "PASS"
        elif maximum <= 0.10:
            verdict = "ENGINEERING-FIX"
        else:
            verdict = "FAIL"
    return {"overheads": overheads, "state_ok": state_ok, "verdict": verdict}


def build_report(
    summary: Mapping[str, Mapping[str, Any]],
    grade_record: Mapping[str, Any],
    system: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    lines = [
        "# E6 report — idle-GPU overhead benchmark",
        "",
        "## Protocol header",
        "",
        "Forced deviations: none.",
        "Seed set: synthetic repeats (90000, 90001, 90002).",
        (
            f"Settings: fp32; {REPEATS} repeats; {WARMUP_STEPS} warmup + "
            f"{TIMED_STEPS} timed steps; fixed synthetic batches; "
            f"lr={LEARNING_RATE}, betas={BETAS}, eps={ADAM_EPS}, matrix-weight "
            f"decay={WEIGHT_DECAY}; CAP k={K}, T_s={T_S}, T_a={T_A}, "
            f"T_m={T_M:g}, raw g^2, contest-blocked release, coordinate box, "
            "zero decay, no c recalibration; all arms eager/uncompiled; "
            "E6 executed last and must be run on an otherwise idle GPU."
        ),
        f"GPU: {system['gpu']}; torch={system['torch']}; CUDA={system['cuda']}; driver={system['driver']}.",
        "Timing covers synchronized synthetic-batch training steps, excluding data loading; optimizer timing includes synchronization. Baseline: AdamW with zero weight decay.",
        "",
        "| Workload | Arm | Full-step median (ms) | Optimizer mean (ms) | Peak memory (bytes) | State ratio | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for workload in WORKLOADS:
        for arm in ARMS:
            record = summary[f"{workload}/{arm}"]
            if record.get("complete"):
                lines.append(
                    f"| {workload} | {arm} | {record['full_step_ms']:.4f} | "
                    f"{record['opt_step_ms']:.4f} | {record['max_memory_bytes']} | "
                    f"{record['state_ratio']:.6f} | OK |"
                )
            else:
                lines.append(
                    f"| {workload} | {arm} | n/a | n/a | n/a | n/a | "
                    f"{','.join(record.get('statuses', [])) or 'MISSING'} |"
                )
    lines.extend(["", "## Criteria", ""])
    for workload in WORKLOADS:
        overhead = grade_record["overheads"][workload]
        lines.append(
            f"- cap_diag_off full-step overhead on {workload}: "
            + ("n/a." if overhead is None else f"{100.0 * overhead:.3f}%.")
        )
    lines.append(
        f"- CAP state-byte audit excluding the bind byte: target 2.0x AdamW ±1%; passes={grade_record['state_ok']}."
    )
    for workload in WORKLOADS:
        baseline = summary[f"{workload}/cap_diag_off"]
        diagnostic = summary[f"{workload}/cap_diag_on"]
        if baseline.get("complete") and diagnostic.get("complete"):
            delta = diagnostic["full_step_ms"] / baseline["full_step_ms"] - 1.0
            lines.append(
                f"- cap_diag_on delta versus cap_diag_off on {workload}: {100.0 * delta:.3f}%."
            )
    if grade_record["verdict"] == "ENGINEERING-FIX":
        lines.append(
            "Required engineering response: profile the table above; the named fallback is bf16 anchor/imp storage."
        )
    lines.extend(
        [
            f"- E6 verdict: **{grade_record['verdict']}**.",
            f"Runtime: {runtime_seconds:.1f} seconds.",
            "",
        ]
    )
    return "\n".join(lines)


def skipped_rows(workload: str, status: str, error: str) -> list[dict[str, Any]]:
    return [
        {
            "workload": workload,
            "arm": arm,
            "repeat": repeat,
            "status": status,
            "error": error,
        }
        for arm in ARMS
        for repeat in range(REPEATS)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/e6"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "e6_results.csv"
    config_path = args.output_dir / "run_config.json"
    report_path = args.output_dir / "report.md"
    existing_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if (args.resume or args.summarize_only) and (
        existing_config.get("experiment_version") != EXPERIMENT_VERSION
    ):
        raise ValueError("Resume/summary requires matching final-CAP E6 results; use a fresh output directory for legacy runs")
    prior_runtime = float(existing_config.get("runtime_seconds", 0.0)) if args.resume or args.summarize_only else 0.0
    rows = read_csv(result_path) if args.resume or args.summarize_only else []
    started = time.perf_counter()

    if not args.summarize_only:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E6 is a CUDA wall-clock benchmark and requires a GPU")
        system = system_record(device)
        # Save provenance before timing so an interrupted first run can resume.
        atomic_write_json(config_path, {
            "experiment": "E6", "experiment_version": EXPERIMENT_VERSION,
            "created_at_utc": existing_config.get("created_at_utc", utc_now()) if args.resume else utc_now(),
            "system": system, "runtime_seconds": prior_runtime,
        })
        for workload in WORKLOADS:
            try:
                batch = fixed_batch(workload, device)
                # Probe the optional dependency before creating per-arm rows.
                probe = build_model(workload, device)
                del probe
                torch.cuda.empty_cache()
            except ModuleNotFoundError as error:
                rows = [row for row in rows if str(row["workload"]) != workload]
                rows.extend(skipped_rows(workload, "SKIPPED-DEPENDENCY", repr(error)))
                write_csv(result_path, rows, RESULT_FIELDS)
                continue
            for arm in ARMS:
                for repeat in range(REPEATS):
                    complete = any(
                        str(row["workload"]) == workload
                        and str(row["arm"]) == arm
                        and int(float(row["repeat"])) == repeat
                        and str(row["status"]) == "OK"
                        for row in rows
                    )
                    if complete and args.resume:
                        continue
                    rows = [
                        row for row in rows
                        if not (
                            str(row["workload"]) == workload
                            and str(row["arm"]) == arm
                            and int(float(row["repeat"])) == repeat
                        )
                    ]
                    rows.append(
                        run_repeat(
                            workload=workload,
                            arm=arm,
                            repeat=repeat,
                            batch=batch,
                            device=device,
                        )
                    )
                    write_csv(result_path, rows, RESULT_FIELDS)
        runtime = prior_runtime + time.perf_counter() - started
    else:
        if not existing_config:
            raise FileNotFoundError(f"missing E6 run config: {config_path}")
        system = existing_config["system"]
        runtime = prior_runtime

    summary = summarize(rows)
    grade_record = grade(summary)
    report = build_report(summary, grade_record, system, runtime)
    report_path.write_text(report, encoding="utf-8")
    atomic_write_json(
        config_path,
        {
            "experiment": "E6",
            "experiment_version": EXPERIMENT_VERSION,
            "created_at_utc": existing_config.get("created_at_utc", utc_now()),
            "updated_at_utc": utc_now(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "system": system,
            "scientific_settings": {
                "workloads": list(WORKLOADS),
                "arms": list(ARMS),
                "repeats": REPEATS,
                "warmup_steps": WARMUP_STEPS,
                "timed_steps": TIMED_STEPS,
                "fp32": True,
                "idle_gpu_required": True,
                "optimizer": {"lr": LEARNING_RATE, "betas": BETAS, "eps": ADAM_EPS,
                              "weight_decay": WEIGHT_DECAY, "k": K, "T_s": T_S,
                              "T_a": T_A, "T_m": T_M, "importance": "raw_g2",
                              "release_update": "contest_blocked", "protection": "coordinate_box"},
            },
            "summary": summary,
            "grade": grade_record,
            "runtime_seconds": runtime,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
