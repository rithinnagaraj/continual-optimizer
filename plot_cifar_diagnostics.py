"""Standalone scientific diagnostic figure from verified priority-4 evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "cifar-diagnostics-v1"
import matplotlib.pyplot as plt
from report_cifar_diagnostics import diagnostics, estimate
from experiment_cifar_diagnostics import SEEDS, AGES


def plot(root, output):
    report = json.loads((root/"verified_report.json").read_text())
    assert report["complete_jobs"] == 160
    summaries = {(r["architecture"], r["age"], r["branch"]): r for r in report["summaries"]}
    effects = {(r["architecture"], r["age"], r["branch"]): r for r in report["paired_effects"]}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    colors = {"box_off": "#D55E00", "importance_reset": "#0072B2", "recenter_only": "#7B3294", "baseline": "#333333"}
    names = {"box_off": "Box off", "importance_reset": "Importance reset", "recenter_only": "Recenter anchors", "baseline": "Baseline"}

    def band(ax, x, estimates, color, label, scale=1):
        mean = np.array([r["mean"] for r in estimates])*scale
        lo = np.array([r["ci95_seed_bootstrap"][0] for r in estimates])*scale
        hi = np.array([r["ci95_seed_bootstrap"][1] for r in estimates])*scale
        ax.plot(x, mean, color=color, marker="o", markersize=3, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=.15, linewidth=0)

    for i, arch in enumerate(["mlp", "cnn"]):
        a, b, c = axes[i]
        band(a, AGES, [summaries[(arch, age, "baseline")]["early_removed_over_proposed"] for age in AGES], colors["baseline"], "Baseline")
        a.set(xlabel="Checkpoint age (tasks)", ylabel="Removed / proposed update norm", title=arch.upper()+": early projection cost")
        for branch in ["box_off", "importance_reset", "recenter_only"]:
            band(b, AGES, [effects[(arch, age, branch)]["early_auc"] for age in AGES], colors[branch], names[branch], 100)
        b.axhline(0, color="gray", linewidth=.7)
        b.set(xlabel="Checkpoint age (tasks)", ylabel="Early accuracy change (pp)", title=arch.upper()+": paired intervention effect")
        steps = list(range(1, 21))+list(range(100, 1001, 100))
        for branch in ["baseline", "recenter_only"]:
            records = [json.loads((root/"jobs"/f"{arch}-{seed}-150-{branch}"/"result.json").read_text()) for seed in SEEDS]
            values = [estimate([diagnostics(r["traces"], step, step)["removed_over_proposed"] for r in records]) for step in steps]
            band(c, steps, values, colors[branch], names[branch])
        c.set(xscale="log", xlabel="Updates after intervention (log scale)", ylabel="Removed / proposed update norm", title=arch.upper()+": recentering at age 150")
        a.set_ylim(0, .55)
        b.set_ylim(-.1, 4.2)
        c.set_ylim(-.01, .55)
        for ax in [a,b,c]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.15)
        b.legend(frameon=False, fontsize=8)
        c.legend(frameon=False, fontsize=8)
    fig.suptitle("Matched CIFAR continuations: five seeds; shaded 95% seed-bootstrap intervals", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n").encode())
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/cifar_diagnostics"))
    p.add_argument("--output", type=Path, default=Path("results/cifar_diagnostics/diagnostics"))
    args = p.parse_args()
    plot(args.root, args.output)
