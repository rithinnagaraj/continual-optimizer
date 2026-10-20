"""E5 drain and single-task sanity checks for anchored CAP decay."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torchvision import datasets, transforms

from experiment_e2 import read_csv, write_csv
from experiment_e3 import (
    adamw_parameter_groups,
    evaluate,
    load_mnist,
    make_loader,
    repeat_batches,
    cap_parameter_groups,
    seed_everything,
)
from experiment_e4a import (
    ADAM_EPS,
    BATCH_SIZE,
    BETAS,
    C_REFRESH_STEPS,
    EVALUATION_BATCH_SIZE,
    FULL_SEEDS,
    K,
    LEARNING_RATE,
    PROTECTION,
    T_A,
    T_S,
    TRACK_BIND_STATS,
    WEIGHT_DECAY,
    atomic_write_json,
    refresh_global_c,
    utc_now,
    verify_preconditions,
)
from model import MLP300
from cap_adam_experimental import CAPAdam


DRAIN_MODES = ("zero", "anchor", "blend")
DRAIN_TRAIN_STEPS = 5_000
DRAIN_STEPS = 100_000
DRAIN_LOG_EVERY = 1_000
DRAIN_CONTROL_THRESHOLD = 0.20
DRAIN_TREATMENT_THRESHOLD = 0.02
TAG_FRACTION = 0.01
T_M = 10_000.0

SANITY_EPOCHS = 20
SANITY_EXPECTED_STEPS = 9_380
SANITY_PRIMARY_K = 3e-5
SANITY_RETRY_K = 7e-5
SANITY_DEVIATION = (
    "the original plan named a CIFAR-10 CNN; the repo has no CNN, so this "
    "uses MLP-300/Fashion-MNIST — representativeness caveat"
)

DRAIN_FIELDS = (
    "decay_mode",
    "seed",
    "step",
    "relative_drift",
    "mean_abs_theta_tagged",
    "mean_abs_anchor_gap",
    "median_imp_tagged",
)
SANITY_FIELDS = (
    "arm",
    "seed",
    "k",
    "steps",
    "test_loss",
    "test_accuracy",
)


def load_fashion_mnist(root: Path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    return (
        datasets.FashionMNIST(root=root, train=True, download=True, transform=transform),
        datasets.FashionMNIST(root=root, train=False, download=True, transform=transform),
    )


def create_cap_optimizer(
    model: nn.Module, *, decay_mode: str, k: float = K
) -> CAPAdam:
    if decay_mode not in DRAIN_MODES:
        raise ValueError(f"unknown E5 decay mode: {decay_mode}")
    return CAPAdam(
        cap_parameter_groups(model, WEIGHT_DECAY),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        k=float(k),
        T_s=T_S,
        T_a=T_A,
        T_m=T_M,
        passive=False,
        protection=PROTECTION,
        retention="release",
        decay_mode=decay_mode,
        track_bind_stats=TRACK_BIND_STATS,
    )


@torch.no_grad()
def select_top_importance_tags(
    model: nn.Module, optimizer: CAPAdam
) -> list[dict[str, Any]]:
    protected = []
    groups_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        group = groups_by_parameter[id(parameter)]
        if not bool(group["protect"]):
            continue
        state = optimizer.state[parameter]
        protected.append((name, parameter, state))
    importance = torch.cat(
        [state["imp"].detach().reshape(-1) for _, _, state in protected]
    )
    tag_count = max(1, math.ceil(TAG_FRACTION * importance.numel()))
    global_indices = torch.topk(importance, tag_count, largest=True).indices
    tags = []
    offset = 0
    for name, parameter, state in protected:
        end = offset + parameter.numel()
        selected = global_indices[
            (global_indices >= offset) & (global_indices < end)
        ] - offset
        if selected.numel():
            selected = selected.to(parameter.device)
            tags.append(
                {
                    "name": name,
                    "indices": selected.detach().cpu(),
                    "theta_tag": parameter.detach().reshape(-1)[selected].cpu(),
                    "imp_tag": state["imp"].detach().reshape(-1)[selected].cpu(),
                }
            )
        offset = end
    if sum(int(tag["indices"].numel()) for tag in tags) != tag_count:
        raise RuntimeError("E5 top-importance tag mapping lost indices")
    return tags


@torch.no_grad()
def drain_measurement(
    model: nn.Module, optimizer: CAPAdam, tags: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    parameters = dict(model.named_parameters())
    current_values = []
    tagged_values = []
    anchor_values = []
    importance_values = []
    for tag in tags:
        parameter = parameters[str(tag["name"])]
        indices = tag["indices"].to(parameter.device)
        current_values.append(parameter.detach().reshape(-1)[indices])
        tagged_values.append(tag["theta_tag"].to(parameter.device))
        state = optimizer.state[parameter]
        anchor_values.append(state["anchor"].detach().reshape(-1)[indices])
        importance_values.append(state["imp"].detach().reshape(-1)[indices])
    current = torch.cat(current_values)
    tagged = torch.cat(tagged_values)
    anchor = torch.cat(anchor_values)
    importance = torch.cat(importance_values)
    denominator = tagged.abs().mean()
    if float(denominator.item()) == 0.0:
        raise RuntimeError("tagged E5 weights have zero mean absolute magnitude")
    return {
        "relative_drift": float(((current - tagged).abs().mean() / denominator).item()),
        "mean_abs_theta_tagged": float(current.abs().mean().item()),
        "mean_abs_anchor_gap": float((anchor - tagged).abs().mean().item()),
        "median_imp_tagged": float(importance.median().item()),
    }


def run_drain_cell(
    *,
    decay_mode: str,
    seed: int,
    mnist_train,
    device: torch.device,
    num_workers: int,
    tag_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer = create_cap_optimizer(model, decay_mode=decay_mode)
    loader = make_loader(
        mnist_train,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    batches = repeat_batches(loader)
    model.train()
    for step in range(1, DRAIN_TRAIN_STEPS + 1):
        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite E5 drain training loss {decay_mode}/{seed}/{step}"
            )

    tags = select_top_importance_tags(model, optimizer)
    tag_dir.mkdir(parents=True, exist_ok=True)
    tag_path = tag_dir / f"{decay_mode}_seed{seed}.pt"
    torch.save(
        {
            "decay_mode": decay_mode,
            "seed": seed,
            "fraction": TAG_FRACTION,
            "tags": list(tags),
        },
        tag_path,
    )
    rows = []
    for step in range(1, DRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        if step % C_REFRESH_STEPS == 0:
            refresh_global_c(optimizer)
        if step % DRAIN_LOG_EVERY == 0:
            rows.append(
                {
                    "decay_mode": decay_mode,
                    "seed": seed,
                    "step": step,
                    **drain_measurement(model, optimizer, tags),
                }
            )
    metadata = {
        "decay_mode": decay_mode,
        "seed": seed,
        "tag_path": str(tag_path),
        "tag_count": sum(int(tag["indices"].numel()) for tag in tags),
    }
    return rows, metadata


def sanity_optimizer(model: nn.Module, arm: str, k: float):
    if arm == "adamw":
        return torch.optim.AdamW(
            adamw_parameter_groups(model, WEIGHT_DECAY),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
    if arm == "cap_blend":
        return create_cap_optimizer(model, decay_mode="blend", k=k)
    raise ValueError(f"unknown E5 sanity arm: {arm}")


def run_sanity_cell(
    *,
    arm: str,
    seed: int,
    k: float,
    train_set,
    test_set,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    seed_everything(seed)
    model = MLP300(input_units=784, output_units=10).to(device)
    optimizer = sanity_optimizer(model, arm, k)
    train_loader = make_loader(
        train_set,
        BATCH_SIZE,
        device,
        num_workers,
        shuffle=True,
        seed=seed,
    )
    test_loader = make_loader(
        test_set,
        EVALUATION_BATCH_SIZE,
        device,
        num_workers,
        shuffle=False,
    )
    step = 0
    model.train()
    for _epoch in range(SANITY_EPOCHS):
        for inputs, targets in train_loader:
            step += 1
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            if isinstance(optimizer, CAPAdam) and step % C_REFRESH_STEPS == 0:
                refresh_global_c(optimizer)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"non-finite E5 sanity loss {arm}/{seed}/{step}"
                )
    if step != SANITY_EXPECTED_STEPS:
        raise RuntimeError(
            f"Fashion-MNIST sanity produced {step} steps, expected {SANITY_EXPECTED_STEPS}"
        )
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    return {
        "arm": arm,
        "seed": seed,
        "k": k if arm == "cap_blend" else "",
        "steps": step,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def pooled_standard_deviation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    numerator = (len(left) - 1) * stdev(left) ** 2 + (len(right) - 1) * stdev(right) ** 2
    return math.sqrt(numerator / (len(left) + len(right) - 2))


def grade_sanity(
    rows: Sequence[Mapping[str, Any]],
    k: float,
    seeds: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    adam_rows = [row for row in rows if str(row["arm"]) == "adamw"]
    cap_rows = [
        row
        for row in rows
        if str(row["arm"]) == "cap_blend" and math.isclose(float(row["k"]), k)
    ]
    adam_by_seed = {
        int(float(row["seed"])): float(row["test_accuracy"]) for row in adam_rows
    }
    cap_by_seed = {
        int(float(row["seed"])): float(row["test_accuracy"]) for row in cap_rows
    }
    required = (
        set(int(seed) for seed in seeds) if seeds is not None else set(adam_by_seed)
    )
    complete = (
        bool(required)
        and set(adam_by_seed) == required
        and set(cap_by_seed) == required
        and len(adam_rows) == len(required)
        and len(cap_rows) == len(required)
    )
    if not complete:
        return {"complete": False, "passes": False, "k": k}
    adam = [adam_by_seed[seed] for seed in sorted(required)]
    cap = [cap_by_seed[seed] for seed in sorted(required)]
    pooled = pooled_standard_deviation(adam, cap)
    difference = mean(cap) - mean(adam)
    return {
        "complete": True,
        "k": k,
        "adamw_mean": mean(adam),
        "cap_mean": mean(cap),
        "difference": difference,
        "pooled_std": pooled,
        "passes": abs(difference) <= pooled,
    }


def grade_drain(
    rows: Sequence[Mapping[str, Any]], seeds: Optional[Sequence[int]] = None
) -> dict[str, Any]:
    final_rows = {
        mode: [
            row
            for row in rows
            if str(row["decay_mode"]) == mode
            and int(float(row["step"])) == DRAIN_STEPS
        ]
        for mode in DRAIN_MODES
    }
    final_by_seed = {
        mode: {
            int(float(row["seed"])): float(row["relative_drift"])
            for row in final_rows[mode]
        }
        for mode in DRAIN_MODES
    }
    required = (
        set(int(seed) for seed in seeds)
        if seeds is not None
        else set(final_by_seed["zero"])
    )
    if not required or any(
        set(final_by_seed[mode]) != required
        or len(final_rows[mode]) != len(required)
        for mode in DRAIN_MODES
    ):
        return {"complete": False, "control_ok": False, "treatment_ok": False}
    means = {
        mode: mean(values.values()) for mode, values in final_by_seed.items()
    }
    control_ok = means["zero"] >= DRAIN_CONTROL_THRESHOLD
    treatment_ok = means["blend"] <= DRAIN_TREATMENT_THRESHOLD
    return {
        "complete": True,
        "final_relative_drift": means,
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "verdict": (
            "DRAIN-CONTROL-ABSENT"
            if not control_ok
            else "PASS" if treatment_ok else "FAIL"
        ),
    }


def grade_norm(path: Path, seeds: Sequence[int]) -> dict[str, Any]:
    if not path.exists():
        return {"complete": False, "passes": False}
    rows = read_csv(path)
    required = set(int(seed) for seed in seeds)
    values = {}
    for step in (100_000, 1_000_000):
        selected = [
            row
            for row in rows
            if str(row["arm"]) == "cap_release_10000"
            and int(float(row["seed"])) in required
            and int(float(row["step"])) == step
        ]
        by_seed = {
            int(float(row["seed"])): float(row["theta_norm"])
            for row in selected
        }
        if len(selected) != len(required) or set(by_seed) != required:
            return {"complete": False, "passes": False}
        values[step] = mean(by_seed.values())
    limit = 1.5 * values[100_000]
    return {
        "complete": True,
        "baseline_mean": values[100_000],
        "final_mean": values[1_000_000],
        "limit": limit,
        "ratio": (
            values[1_000_000] / values[100_000]
            if values[100_000] != 0.0
            else math.inf
        ),
        "passes": values[1_000_000] <= limit,
    }


def preliminary(verdict: str, seeds: Sequence[int]) -> str:
    return verdict if set(seeds) == set(FULL_SEEDS) and len(seeds) == len(FULL_SEEDS) else "PRELIMINARY-" + verdict


def build_report(
    *,
    drain: Mapping[str, Any],
    sanity_primary: Mapping[str, Any],
    sanity_retry: Mapping[str, Any],
    norm: Mapping[str, Any],
    seeds: Sequence[int],
    runtime_seconds: float,
) -> tuple[str, str]:
    retry_required = bool(
        sanity_primary.get("complete") and not sanity_primary.get("passes")
    )
    sanity_selected = sanity_retry if retry_required else sanity_primary
    complete = bool(drain.get("complete") and sanity_selected.get("complete") and norm.get("complete"))
    if not complete:
        raw_verdict = "PENDING"
    elif drain["control_ok"] and drain["treatment_ok"] and sanity_selected["passes"] and norm["passes"]:
        raw_verdict = "PASS"
    elif not drain["control_ok"]:
        raw_verdict = "DRAIN-CONTROL-ABSENT"
    else:
        raw_verdict = "FAIL"
    verdict = preliminary(raw_verdict, seeds)
    lines = [
        "# E5 report — anchored-decay health",
        "",
        "## Protocol header",
        "",
        f"Seed set run: {tuple(seeds)}.",
        f"Forced deviation: {SANITY_DEVIATION}.",
        (
            "Settings: MLP-300, batch=128, lr=1e-3, betas=(0.9,0.999), "
            "eps=1e-8, matrix-weight decay=0.01; CAP release, T_m=10000, "
            "k=3e-5, T_s=T_a=2000, box protection, c refreshed every 100 "
            "steps; drain modes zero/anchor/blend; 5,000 MNIST training "
            "steps then 100,000 zero-gradient steps; Fashion-MNIST sanity "
            "for 20 epochs."
        ),
        "",
        "## Drain criterion",
        "",
    ]
    if drain.get("complete"):
        values = drain["final_relative_drift"]
        lines.extend(
            [
                f"- zero final relative drift={values['zero']:.6f}; control threshold >= {DRAIN_CONTROL_THRESHOLD:.2f}; passes={drain['control_ok']}.",
                f"- blend final relative drift={values['blend']:.6f}; treatment threshold <= {DRAIN_TREATMENT_THRESHOLD:.2f}; passes={drain['treatment_ok']}.",
                f"- anchor final relative drift={values['anchor']:.6f} (descriptive).",
                f"- Drain verdict: **{preliminary(str(drain['verdict']), seeds)}**.",
            ]
        )
    else:
        lines.append("Drain run incomplete.")
    lines.extend(["", "## Stream-A norm criterion", ""])
    if norm.get("complete"):
        lines.append(
            f"blend theta_norm(1,000,000)={norm['final_mean']:.6g}; limit=1.5 x theta_norm(100,000)={norm['limit']:.6g}; ratio={norm['ratio']:.6f}; passes={norm['passes']}."
        )
    else:
        lines.append("Banked Stream-A blend norm rows are incomplete.")
    lines.extend(["", "## Single-task sanity", "", SANITY_DEVIATION + "."])
    for label, grade in (("k=3e-5", sanity_primary), ("k=7e-5 retry", sanity_retry)):
        if grade.get("complete"):
            lines.append(
                f"- {label}: CAP-AdamW accuracy difference={grade['difference']:+.6f}; pooled SD={grade['pooled_std']:.6f}; |difference| <= pooled SD passes={grade['passes']}."
            )
    if sanity_primary.get("complete") and not sanity_primary.get("passes"):
        lines.append("The first-failure response reports the measured cost and invokes exactly one k=7e-5 rerun.")
    lines.extend(["", f"Overall E5 verdict: **{verdict}**.", f"Runtime: {runtime_seconds:.1f} seconds.", ""])
    return "\n".join(lines), verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FULL_SEEDS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e5"))
    parser.add_argument(
        "--stream-a-scalars",
        type=Path,
        default=Path("results/stream_a/stream_scalars.csv"),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--drain-only", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonempty and unique")
    if args.drain_only and args.sanity_only:
        parser.error("--drain-only and --sanity-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    verify_preconditions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drain_path = args.output_dir / "e5_drain.csv"
    sanity_path = args.output_dir / "e5_sanity.csv"
    config_path = args.output_dir / "run_config.json"
    report_path = args.output_dir / "report.md"
    existing_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if args.summarize_only and existing_config:
        args.seeds = tuple(int(seed) for seed in existing_config["seeds"])
    elif args.resume and existing_config:
        configured_seeds = tuple(int(seed) for seed in existing_config["seeds"])
        if set(configured_seeds) != set(args.seeds):
            raise ValueError("E5 resume seeds do not match run_config.json")
    prior_runtime = float(existing_config.get("runtime_seconds", 0.0))
    started = time.perf_counter()
    drain_rows = read_csv(drain_path) if drain_path.exists() else []
    sanity_rows = read_csv(sanity_path) if sanity_path.exists() else []
    tag_records = list(existing_config.get("tag_records", []))
    if not args.resume and not args.summarize_only:
        if not args.sanity_only:
            drain_rows = []
            tag_records = []
            write_csv(drain_path, drain_rows, DRAIN_FIELDS)
        if not args.drain_only:
            sanity_rows = []
            write_csv(sanity_path, sanity_rows, SANITY_FIELDS)

    if not args.summarize_only:
        device = torch.device(args.device)
        if not args.sanity_only:
            mnist_train, _mnist_test = load_mnist(args.data_dir)
            completed = {
                (str(row["decay_mode"]), int(float(row["seed"])))
                for row in drain_rows
                if int(float(row["step"])) == DRAIN_STEPS
            }
            for mode in DRAIN_MODES:
                for seed in args.seeds:
                    if (mode, seed) in completed and args.resume:
                        print(f"Skipping completed E5 drain cell {(mode, seed)}")
                        continue
                    drain_rows = [
                        row for row in drain_rows
                        if not (str(row["decay_mode"]) == mode and int(float(row["seed"])) == seed)
                    ]
                    rows, metadata = run_drain_cell(
                        decay_mode=mode,
                        seed=seed,
                        mnist_train=mnist_train,
                        device=device,
                        num_workers=args.num_workers,
                        tag_dir=args.output_dir / "tags",
                    )
                    drain_rows.extend(rows)
                    tag_records = [
                        record for record in tag_records
                        if not (record["decay_mode"] == mode and int(record["seed"]) == seed)
                    ]
                    tag_records.append(metadata)
                    write_csv(drain_path, drain_rows, DRAIN_FIELDS)

        if not args.drain_only:
            fashion_train, fashion_test = load_fashion_mnist(args.data_dir)
            required = [("adamw", seed, 0.0) for seed in args.seeds] + [
                ("cap_blend", seed, SANITY_PRIMARY_K) for seed in args.seeds
            ]
            for arm, seed, k in required:
                exists = any(
                    str(row["arm"]) == arm
                    and int(float(row["seed"])) == seed
                    and (arm == "adamw" or math.isclose(float(row["k"]), k))
                    for row in sanity_rows
                )
                if exists and args.resume:
                    continue
                sanity_rows.append(
                    run_sanity_cell(
                        arm=arm,
                        seed=seed,
                        k=k,
                        train_set=fashion_train,
                        test_set=fashion_test,
                        device=device,
                        num_workers=args.num_workers,
                    )
                )
                write_csv(sanity_path, sanity_rows, SANITY_FIELDS)
            primary = grade_sanity(sanity_rows, SANITY_PRIMARY_K, args.seeds)
            if primary.get("complete") and not primary.get("passes"):
                for seed in args.seeds:
                    exists = any(
                        str(row["arm"]) == "cap_blend"
                        and int(float(row["seed"])) == seed
                        and math.isclose(float(row["k"]), SANITY_RETRY_K)
                        for row in sanity_rows
                    )
                    if exists and args.resume:
                        continue
                    sanity_rows.append(
                        run_sanity_cell(
                            arm="cap_blend",
                            seed=seed,
                            k=SANITY_RETRY_K,
                            train_set=fashion_train,
                            test_set=fashion_test,
                            device=device,
                            num_workers=args.num_workers,
                        )
                    )
                    write_csv(sanity_path, sanity_rows, SANITY_FIELDS)

    drain_grade = grade_drain(drain_rows, args.seeds)
    sanity_primary = grade_sanity(sanity_rows, SANITY_PRIMARY_K, args.seeds)
    sanity_retry = grade_sanity(sanity_rows, SANITY_RETRY_K, args.seeds)
    norm_grade = grade_norm(args.stream_a_scalars, args.seeds)
    runtime = prior_runtime + time.perf_counter() - started
    report, verdict = build_report(
        drain=drain_grade,
        sanity_primary=sanity_primary,
        sanity_retry=sanity_retry,
        norm=norm_grade,
        seeds=args.seeds,
        runtime_seconds=runtime,
    )
    report_path.write_text(report, encoding="utf-8")
    atomic_write_json(
        config_path,
        {
            "experiment": "E5",
            "created_at_utc": existing_config.get("created_at_utc", utc_now()),
            "updated_at_utc": utc_now(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "seeds": list(args.seeds),
            "forced_deviations": [SANITY_DEVIATION],
            "scientific_settings": {
                "drain_modes": list(DRAIN_MODES),
                "train_steps": DRAIN_TRAIN_STEPS,
                "drain_steps": DRAIN_STEPS,
                "tag_fraction": TAG_FRACTION,
                "sanity_epochs": SANITY_EPOCHS,
                "primary_k": SANITY_PRIMARY_K,
                "retry_k": SANITY_RETRY_K,
            },
            "tag_records": tag_records,
            "drain_grade": drain_grade,
            "norm_grade": norm_grade,
            "sanity_primary_grade": sanity_primary,
            "sanity_retry_grade": sanity_retry,
            "verdict": verdict,
            "runtime_seconds": runtime,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
