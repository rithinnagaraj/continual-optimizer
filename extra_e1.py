"""Finish E1: corrected group test, frozen-model autopsy, final verdict.

Reuses saved artifacts (models + signals) and correlations_by_seed.csv, so
nothing is retrained and Fisher is never recomputed.

Part 1  Corrected group test: the top group's K-trial average is compared
        against the distribution of K-trial averages of G layer-matched
        random groups (averages vs averages -- the fair version).
Part 2  Frozen-model autopsy: recompute imp's estimator on the frozen final
        model at several window lengths, against the artifact's Fisher.
        Built-in control: a frozen raw squared-gradient average must agree
        with Fisher at ~0.99, or the harness itself is suspect.
Part 3  Final criteria-v2.1 verdict combining the corrected group numbers
        with the individual-probe criteria from correlations_by_seed.csv.

The acceptance rule is pre-committed: if the corrected group test still
fails at both budgets in most seeds, the weakness is real and the fix loop
escalates (per-neuron aggregation, then the relative-usage fallback).
No further criterion amendments.

Usage:
    python finish_e1.py --results-dir results/e1 --data-dir data

Place next to experiment_e1.py (the criteria-v2 patched version),
model.py, and cap_adam_experimental.py.  If your patched script has a different file
name, edit the import line below.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from model import MLP300
from patched_e1 import (  # the criteria-v2 patched script
    GROUP_FRACTIONS,
    collect_held_out,
    group_trial_damage,
    layer_matched_random_indices,
    load_dataset,
    parameter_offsets,
    read_csv,
    spearman,
    test_loss,
    weight_parameters,
    write_csv,
)

SEEDS_DEFAULT = (42, 123, 2026, 451, 789)
CRITERIA_FRACTIONS = (0.05, 0.20)
AUTOPSY_WINDOWS = (500, 2000, 8000)
AUTOPSY_SNAPSHOTS = (500, 1000, 2000, 4000, 8000)


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------


def load_artifact(path: Path, device: torch.device):
    """Rebuild the frozen model and per-weight signals from one artifact."""
    artifact = torch.load(path, map_location="cpu")
    model = MLP300(input_units=28 * 28, output_units=10)
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device).eval()
    weights = weight_parameters(model)
    saved_names = list(artifact["weight_names"])
    current_names = [name for name, _ in weights]
    if saved_names != current_names:
        raise RuntimeError(
            f"Weight order mismatch in {path}: {saved_names} vs {current_names}"
        )
    return model, weights, artifact["signals"]


# ---------------------------------------------------------------------------
# Part 1 - corrected group test (averages vs averages)
# ---------------------------------------------------------------------------


def build_noise_scale(
    weights: Sequence[Tuple[str, nn.Parameter]],
    noise_fraction: float,
    device: torch.device,
) -> Tensor:
    parts = []
    for _, parameter in weights:
        layer_std = parameter.detach().std(unbiased=False).clamp_min(1e-12)
        parts.append(
            torch.full(
                (parameter.numel(),),
                noise_fraction * float(layer_std.item()),
                device=device,
            )
        )
    return torch.cat(parts)


def group_average_damage(
    model,
    weights,
    offsets,
    indices: Tensor,
    noise_scale: Tensor,
    inputs: Tensor,
    targets: Tensor,
    baseline: Tensor,
    batch_size: int,
    generator: torch.Generator,
    trials: int,
) -> float:
    """Average antithetic damage over `trials` noise draws for ONE group."""
    return mean(
        group_trial_damage(
            model,
            weights,
            offsets,
            indices,
            noise_scale,
            inputs,
            targets,
            baseline,
            batch_size,
            generator,
        )
        for _ in range(trials)
    )


def corrected_group_probe(
    model,
    weights,
    signals: Dict[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    seed: int,
    args,
) -> List[dict]:
    offsets = parameter_offsets(weights)
    device = weights[0][1].device
    noise_scale = build_noise_scale(weights, args.group_noise_fraction, device)
    baseline_vector = test_loss(model, inputs, targets, args.evaluation_batch_size)
    cpu_generator = torch.Generator().manual_seed(seed + 50_000)
    device_generator = torch.Generator(device=device).manual_seed(seed + 60_000)
    total = offsets[-1]
    rows = []

    for fraction in GROUP_FRACTIONS:
        group_size = max(1, round(total * fraction))
        for signal_name in args.signals:
            top_indices = torch.topk(
                signals[signal_name], group_size, largest=True, sorted=False
            ).indices
            top_average = group_average_damage(
                model, weights, offsets, top_indices, noise_scale,
                inputs, targets, baseline_vector,
                args.evaluation_batch_size, device_generator,
                args.trials_per_group,
            )
            random_averages = []
            for _ in range(args.random_groups):
                random_indices = layer_matched_random_indices(
                    top_indices, offsets, cpu_generator
                )
                random_averages.append(
                    group_average_damage(
                        model, weights, offsets, random_indices, noise_scale,
                        inputs, targets, baseline_vector,
                        args.evaluation_batch_size, device_generator,
                        args.trials_per_group,
                    )
                )
            random_mean = mean(random_averages)
            random_spread = stdev(random_averages) if len(random_averages) > 1 else 0.0
            percentile = 100.0 * sum(
                value <= top_average for value in random_averages
            ) / len(random_averages)
            z_score = (
                (top_average - random_mean) / random_spread
                if random_spread > 0
                else float("nan")
            )
            rows.append(
                {
                    "signal": signal_name,
                    "fraction": fraction,
                    "group_size": group_size,
                    "top_average": top_average,
                    "random_average_mean": random_mean,
                    "random_average_std": random_spread,
                    "percentile": percentile,
                    "z_score": z_score,
                }
            )
            print(
                f"    {signal_name:>16} top-{100 * fraction:g}%: "
                f"percentile {percentile:5.1f}, z={z_score:+.2f}"
            )
    return rows


# ---------------------------------------------------------------------------
# Part 2 - frozen-model autopsy
# ---------------------------------------------------------------------------


def frozen_autopsy(
    model,
    weights,
    signals: Dict[str, Tensor],
    train_set,
    seed: int,
    args,
    device: torch.device,
) -> List[dict]:
    """Re-run imp's estimator on the frozen model; compare against Fisher."""
    fisher = signals["fisher"].numpy()
    trained_imp = signals["imp"].numpy()
    baseline_gap_rho = spearman(trained_imp, fisher)

    parameters = [parameter for _, parameter in weights]
    beta1 = 0.9
    momentum = [torch.zeros_like(parameter) for parameter in parameters]
    centered_ema = {
        window: [torch.zeros_like(parameter) for parameter in parameters]
        for window in AUTOPSY_WINDOWS
    }
    ratchet = [torch.zeros_like(parameter) for parameter in parameters]
    raw_ema = [torch.zeros_like(parameter) for parameter in parameters]
    raw_window = 2000

    loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 70_000),
        num_workers=args.num_workers,
        drop_last=True,
    )

    def flatten(stat: List[Tensor]) -> np.ndarray:
        return torch.cat([tensor.flatten() for tensor in stat]).cpu().numpy()

    rows = []
    step = 0
    print(f"    autopsy: {args.autopsy_steps} frozen steps")
    while step < args.autopsy_steps:
        for inputs, targets in loader:
            if step >= args.autopsy_steps:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = F.cross_entropy(model(inputs), targets)
            gradients = torch.autograd.grad(loss, parameters)
            step += 1
            with torch.no_grad():
                correction = 1.0 - beta1**step
                for index, gradient in enumerate(gradients):
                    momentum[index].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    m_hat = momentum[index] / correction
                    centered = (gradient - m_hat).square()
                    for window in AUTOPSY_WINDOWS:
                        centered_ema[window][index].lerp_(centered, 1.0 / window)
                    candidate = ratchet[index].lerp(centered, 1.0 / 2000)
                    torch.maximum(ratchet[index], candidate, out=ratchet[index])
                    raw_ema[index].lerp_(gradient.square(), 1.0 / raw_window)
            if step in AUTOPSY_SNAPSHOTS:
                row = {"step": step, "trained_imp_vs_fisher": baseline_gap_rho}
                for window in AUTOPSY_WINDOWS:
                    row[f"centered_T{window}_vs_fisher"] = spearman(
                        flatten(centered_ema[window]), fisher
                    )
                row["ratchet_T2000_vs_fisher"] = spearman(flatten(ratchet), fisher)
                row["raw_T2000_vs_fisher_CONTROL"] = spearman(
                    flatten(raw_ema), fisher
                )
                row["centered_T2000_vs_trained_imp"] = spearman(
                    flatten(centered_ema[2000]), trained_imp
                )
                rows.append(row)
                print(
                    f"      step {step}: centered@2000 vs Fisher "
                    f"rho={row['centered_T2000_vs_fisher']:.3f}, "
                    f"control={row['raw_T2000_vs_fisher_CONTROL']:.3f}"
                )
    return rows


def autopsy_verdict(final_row: dict, baseline_gap: float) -> str:
    control = final_row["raw_T2000_vs_fisher_CONTROL"]
    frozen = final_row["centered_T2000_vs_fisher"]
    longest = final_row[f"centered_T{max(AUTOPSY_WINDOWS)}_vs_fisher"]
    if not math.isfinite(control) or control < 0.95:
        return (
            "CONTROL FAILED: the frozen raw average should match Fisher at "
            "~0.99. Investigate the harness before interpreting anything else."
        )
    if frozen >= 0.90:
        return (
            "STALENESS CONFIRMED: on a motionless model the same formula "
            f"reaches {frozen:.3f} against Fisher (trained value was "
            f"{baseline_gap:.3f}). The gap came from accumulating while the "
            "weights were still moving. Document as definitional -- in the "
            "real optimizer imp trails, but the anchor it protects trails "
            "with it. No knob change required."
        )
    if longest > frozen + 0.05:
        return (
            "WINDOW-LIMITED: agreement keeps improving with longer windows "
            f"({frozen:.3f} at T=2000 vs {longest:.3f} at T={max(AUTOPSY_WINDOWS)}). "
            "Recommend a longer T_s, noting that protection then builds more "
            "slowly -- feed this into E3/E4 configs."
        )
    return (
        "UNEXPLAINED: the frozen centered estimator still disagrees with "
        "Fisher while the raw control matches it. The centering term is the "
        "remaining suspect; inspect momentum magnitudes before proceeding."
    )


# ---------------------------------------------------------------------------
# Part 3 - final verdict
# ---------------------------------------------------------------------------


def individual_criteria(correlation_rows: Sequence[dict], dataset: str) -> dict:
    by_seed: Dict[int, Dict[str, float]] = defaultdict(dict)
    for row in correlation_rows:
        if row["dataset"] == dataset and row.get("probe_split", "test") == "test":
            by_seed[int(row["seed"])][row["signal"]] = float(row["rho_delta"])
    ratios, advantages = [], []
    for values in by_seed.values():
        if all(
            signal in values and math.isfinite(values[signal])
            for signal in ("imp", "fisher", "weight_magnitude")
        ):
            if values["fisher"] > 0:
                ratios.append(values["imp"] / values["fisher"])
            advantages.append(values["imp"] - values["weight_magnitude"])
    return {
        "ratio_mean": mean(ratios) if ratios else float("nan"),
        "ratio_values": ratios,
        "advantage_mean": mean(advantages) if advantages else float("nan"),
    }


def final_verdict(individual: dict, group_rows: Sequence[dict]) -> Tuple[str, str]:
    by_seed: Dict[int, Dict[float, float]] = defaultdict(dict)
    for row in group_rows:
        if row["signal"] == "imp":
            by_seed[int(row["seed"])][float(row["fraction"])] = float(
                row["percentile"]
            )
    seed_count = len(by_seed)
    required = math.ceil(0.8 * seed_count)
    successes = {
        fraction: sum(
            values.get(fraction, -math.inf) >= 95.0 for values in by_seed.values()
        )
        for fraction in CRITERIA_FRACTIONS
    }
    group_passes = seed_count > 0 and all(
        count >= required for count in successes.values()
    )
    both_fail = sum(
        all(values.get(fraction, -math.inf) < 95.0 for fraction in CRITERIA_FRACTIONS)
        for values in by_seed.values()
    )
    group_hard_fails = seed_count > 0 and both_fail > seed_count / 2

    ratio = individual["ratio_mean"]
    detail = (
        f"paired imp/Fisher ratio={ratio:.3f}; "
        f"imp-minus-|theta| advantage={individual['advantage_mean']:.3f}; "
        + "; ".join(
            f"top-{100 * fraction:g}% above 95th in {successes[fraction]}/{seed_count} seeds"
            for fraction in CRITERIA_FRACTIONS
        )
    )
    if (math.isfinite(ratio) and ratio < 0.6) or group_hard_fails:
        return "HARD FAIL", detail
    if (
        math.isfinite(ratio)
        and ratio >= 0.8
        and group_passes
        and individual["advantage_mean"] > 0
    ):
        return "PASS", detail
    return "INCONCLUSIVE", detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/e1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets", nargs="+", default=("fashion_mnist", "mnist")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--held-out", type=int, default=10_000)
    parser.add_argument("--evaluation-batch-size", type=int, default=1_000)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--group-noise-fraction", type=float, default=0.1)
    parser.add_argument("--random-groups", type=int, default=40)
    parser.add_argument("--trials-per-group", type=int, default=5)
    parser.add_argument(
        "--signals", nargs="+", default=("imp", "fisher", "weight_magnitude")
    )
    parser.add_argument("--autopsy-steps", type=int, default=8_000)
    parser.add_argument("--skip-group", action="store_true")
    parser.add_argument("--skip-autopsy", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = args.results_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation_rows = read_csv(args.results_dir / "correlations_by_seed.csv")
    group_rows_all: List[dict] = []
    autopsy_rows_all: List[dict] = []
    report: List[str] = [
        "# E1 final report — criteria v2.1",
        "",
        "v2.1 = v2 with one measurement correction: the group test now compares",
        "the top group's trial average against the distribution of random-group",
        "trial averages (averages vs averages). Acceptance is pre-committed:",
        "if the corrected group test fails at both budgets in most seeds, the",
        "weakness is real and the fix loop escalates. No further amendments.",
        "",
    ]

    for dataset_name in args.datasets:
        train_set, test_set = load_dataset(dataset_name, args.data_dir)
        dataset_groups: List[dict] = []
        report.extend([f"## {dataset_name}", ""])
        for seed in args.seeds:
            run_name = f"{dataset_name}_seed{seed}"
            print(f"\n[{run_name}]")
            artifact_path = args.results_dir / "artifacts" / f"{run_name}.pt"
            model, weights, signals = load_artifact(artifact_path, device)

            if not args.skip_group:
                inputs, targets = collect_held_out(
                    test_set, args.held_out, seed + 10_000, device
                )
                rows = corrected_group_probe(
                    model, weights, signals, inputs, targets, seed, args
                )
                for row in rows:
                    row.update({"dataset": dataset_name, "seed": seed})
                dataset_groups.extend(rows)
                group_rows_all.extend(rows)

            if not args.skip_autopsy:
                rows = frozen_autopsy(
                    model, weights, signals, train_set, seed, args, device
                )
                baseline_gap = rows[-1]["trained_imp_vs_fisher"]
                verdict_text = autopsy_verdict(rows[-1], baseline_gap)
                for row in rows:
                    row.update({"dataset": dataset_name, "seed": seed})
                autopsy_rows_all.extend(rows)
                report.append(f"- **{run_name} autopsy:** {verdict_text}")

        if not args.skip_group:
            individual = individual_criteria(correlation_rows, dataset_name)
            verdict, detail = final_verdict(individual, dataset_groups)
            report.extend(
                ["", f"**Final verdict: {verdict}.** {detail}", ""]
            )

    if group_rows_all:
        write_csv(output_dir / "corrected_group_by_seed.csv", group_rows_all)
    if autopsy_rows_all:
        write_csv(output_dir / "autopsy_trajectories.csv", autopsy_rows_all)
    report_text = "\n".join(report) + "\n"
    (output_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"Written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
