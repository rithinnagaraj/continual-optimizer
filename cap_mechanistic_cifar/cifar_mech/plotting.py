from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "figure.constrained_layout.use": True})

def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def seed_lines(frame, x, y, path, label=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if label:
        for name, group in frame.groupby(label, sort=False):
            for seed, values in group.groupby("seed"):
                values = values.groupby(x, as_index=False)[y].mean().sort_values(x)
                ax.plot(values[x], values[y], alpha=.2, lw=.8)
            values = group.groupby(x, as_index=False)[y].mean().sort_values(x)
            ax.plot(values[x], values[y], "o-", label=str(name))
        ax.legend(fontsize=8)
    else:
        for seed, values in frame.groupby("seed"):
            values = values.groupby(x, as_index=False)[y].mean().sort_values(x)
            ax.plot(values[x], values[y], "o-", alpha=.6, label=str(seed))
        ax.legend(title="Seed", fontsize=8)
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    save(fig, path)

def branch_plot(frame, outcomes, path):
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6*len(outcomes), 4.5), squeeze=False)
    branches = list(dict.fromkeys(frame.branch))
    for ax, metric in zip(axes[0], outcomes):
        for seed, g in frame.groupby("seed"):
            vals = g.groupby("branch")[metric].mean()
            ax.plot(range(len(branches)), [vals.get(b, np.nan) for b in branches], "o-", alpha=.55, label=str(seed))
        ax.set_xticks(range(len(branches)), branches, rotation=35, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
    axes[0][0].legend(title="Seed")
    save(fig, path)

def frontier(frame, path):
    ages = sorted(frame.checkpoint_task.unique())
    fig, axes = plt.subplots(1, len(ages), figsize=(4.5*len(ages), 4), squeeze=False)
    for ax, age in zip(axes[0], ages):
        g = frame[frame.checkpoint_task == age]
        finite = g[g.branch.str.startswith("k_")].copy()
        finite["multiplier"] = finite.branch.str[2:].astype(float)
        for seed, values in finite.groupby("seed"):
            values = values.sort_values("multiplier")
            ax.plot(values.retention, values.early_auc, "o-", alpha=.3)
        for multiplier, vals in finite.groupby("multiplier"):
            ax.annotate(f"{multiplier:g}×", (vals.retention.mean(), vals.early_auc.mean()))
        off = g[g.branch == "box_off"]
        ax.scatter(off.retention, off.early_auc, marker="x", color="black", label="box off")
        ax.set(title=f"After task {age}", xlabel="Old-task retention", ylabel="Early plasticity AUC")
        ax.legend()
    save(fig, path)

def faithfulness_plot(frame, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for ax, fraction in zip(axes[0], sorted(frame.fraction.unique())):
        g = frame[(frame.fraction == fraction) & (frame.comparator == "importance")]
        for seed, block in g.groupby("seed"):
            ax.plot([0, 1, 2], [block[c].mean() for c in ("bottom_imp_damage", "random_damage_mean", "top_imp_damage")], "o-", alpha=.65, label=str(seed))
        ax.set_xticks([0, 1, 2], ["Low I", "Random", "High I"])
        ax.set(title=f"{100*fraction:g}% of layer", ylabel="Antithetic retained-loss damage")
    axes[0][0].legend(title="Seed")
    save(fig, path)

def prediction_plot(frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ("baseline", "combined")):
        g = frame[(frame.model == name) & (frame.outcome == "early_auc")]
        for seed, rows in g.groupby("seed"):
            ax.scatter(rows.observed, rows.predicted, s=9, alpha=.5, label=str(seed))
        lo = min(g.observed.min(), g.predicted.min())
        hi = max(g.observed.max(), g.predicted.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax.set(title=name, xlabel="Observed next-task early AUC", ylabel="Held-out prediction")
    axes[0].legend(title="Seed")
    save(fig, path)
