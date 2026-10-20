"""Render selected paper figures from fresh local measurements, or the update schematic."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def schematic(output):
    # The diagram illustrates the algorithm; it contains no experiment values.
    output.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="240" viewBox="0 0 900 240">
<rect width="900" height="240" fill="white"/>
<g font-family="sans-serif" font-size="18" fill="#162330">
<text x="32" y="32">CAP-Adam: one protected coordinate</text>
<rect x="260" y="65" width="340" height="75" fill="#dbeafe"/>
<path d="M60 105 H830" stroke="#162330" stroke-width="2"/>
<path d="M260 64 V146 M430 64 V146 M600 64 V146" stroke="#162330" stroke-width="2"/>
<circle cx="740" cy="105" r="8" fill="#c2410c"/>
<circle cx="600" cy="105" r="8" fill="#0369a1"/>
<path d="M724 84 H619 l12 -7 m-12 7 l12 7" fill="none" stroke="#c2410c" stroke-width="2"/>
<text x="240" y="170">a-r</text><text x="424" y="170">a</text><text x="582" y="170">a+r</text>
<text x="690" y="66">Adam proposal z</text><text x="560" y="198">clipped weight</text>
<text x="32" y="224">r = k/(sqrt(I)+eps); crossing: |z-a| &gt; r; bound I decays; free I ratchets.</text>
</g></svg>''', encoding="utf-8")


def render(target):
    output = ROOT/"results/paper"/target
    output.mkdir(parents=True, exist_ok=True)
    if target == "figure-01":
        schematic(output/"figure.svg")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if target == "figure-03":
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
        for ax, filename, xkey, ykey in (
            (axes[0], "horizon_plasticity.csv", "horizon", "plasticity"),
            (axes[1], "retention_by_gap.csv", "gap_bucket", "retention")):
            rows = read(ROOT/"results/paper_benchmark/analysis"/filename)
            for method in dict.fromkeys(r.get("method", r.get("variant")) for r in rows):
                grouped = defaultdict(list)
                for row in rows:
                    if row.get("method", row.get("variant")) == method:
                        grouped[row[xkey]].append(float(row[ykey]))
                keys = sorted(grouped, key=float) if xkey=="horizon" else [k for k in ("1-5","6-10","11-20","21-40","41-80") if k in grouped]
                x = [int(k)/1000 for k in keys] if xkey=="horizon" else keys
                ax.plot(x, [mean(grouped[k]) for k in keys], marker="o", label=method,
                        linestyle="--" if method=="adam_replay" else "-")
            ax.set(xlabel="Horizon (thousand updates)" if xkey=="horizon" else "Revisit gap (tasks)", ylabel=ykey.capitalize())
        axes[0].axvspan(500,750,alpha=.1,color="gray")
        axes[1].legend(fontsize=8)
    elif target == "figure-07":
        payload=json.loads((ROOT/"results/mechanistic_mnist/reproduction/reproduction_gate.json").read_text())
        rows=payload["rows"]
        if any(r["absolute_error"] is None for r in rows):
            raise ValueError("Endpoint-error figure requires complete observations")
        fig, ax=plt.subplots(figsize=(8,3.7))
        for metric in dict.fromkeys(r["metric"] for r in rows):
            subset=[r for r in rows if r["metric"]==metric]
            ax.plot([str(r["seed"]) for r in subset],[100*r["absolute_error"] for r in subset],marker="o",label=metric)
        ax.axhline(100*payload["tolerance"],linestyle="--",color="black",label="Gate tolerance")
        ax.set(xlabel="Seed",ylabel="Absolute endpoint error (percentage points)")
        ax.legend(fontsize=8)
    else:
        scalars=read(ROOT/"results/stream_a/stream_scalars.csv")
        tasks=read(ROOT/"results/stream_a/stream_tasks.csv")
        arms=("adamw","cap_leak","cap_ratchet","cap_release_10000")
        fig, axes=plt.subplots(1,2,figsize=(10,3.7))
        for arm in arms:
            blocks=defaultdict(list)
            for row in scalars:
                if row["arm"]==arm and row["frozen_fraction"]!="":
                    block=(int(float(row["step"]))-1)//100000
                    blocks[block].append(100*float(row["frozen_fraction"]))
            if blocks:
                axes[0].plot([20*(b+1) for b in sorted(blocks)], [mean(blocks[b]) for b in sorted(blocks)],label=arm)
            values=[]
            for low,high in ((1,20),(181,200)):
                selected=[r for r in tasks if r["arm"]==arm and low<=int(float(r["task"]))<=high]
                if len(selected)!=100:
                    raise ValueError(f"Expected 5 x 20 task measurements for {arm}")
                values.append(mean(float(r["steps_to_085"]) if r["steps_to_085"]!="" else 5000 for r in selected))
            axes[1].plot(["First 20 tasks","Last 20 tasks"],values,marker="o",label=arm)
        axes[0].set(xlabel="Task-window end",ylabel="Frozen weights (%)")
        axes[1].set(ylabel="Updates to 85% accuracy (5000 if unreached)")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png","pdf"):
        fig.savefig(output/("figure."+suffix),dpi=180)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target",choices=("figure-01","figure-03","figure-07","figure-09"))
    args=parser.parse_args(argv)
    render(args.target)


if __name__ == "__main__":
    main()
