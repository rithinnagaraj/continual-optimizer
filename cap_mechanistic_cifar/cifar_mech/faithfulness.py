from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import torch
from .checkpointing import require_gate, prepare_run, file_hash, load, restore, atomic_save, atomic_json, digest
from .runner import manifest, bank_path, provenance, new_state, NAMES
from .constants import MECHANISMS, STREAMS
from .data import Data
from .metrics import evaluate
from .interventions import protected
from .probes.fisher import empirical_fisher
from .probes.functional_damage import antithetic_damage
from .records import Journal, read_table, write_csv
from .statistics import cluster_bootstrap

def run(args):
    require_gate(args.output_dir, args.allow_unverified_reference)
    tasks = [2] if args.smoke else MECHANISMS["m1_checkpoints"]
    sources = {f"{s}_{t}": bank_path(args.output_dir, "b", s, t) for s in args.seeds for t in tasks}
    config = manifest(args, "m1", {k: file_hash(v) for k, v in sources.items()})
    root = Path(args.output_dir)/NAMES["m1"]
    prepare_run(root, config, args.resume)
    data = Data(args.data_dir, args.smoke)
    for seed in args.seeds:
        for task in tasks:
            source = sources[f"{seed}_{task}"]
            model, optimizer, sampler = new_state(seed, data, args.device, args.smoke)
            restore(load(source), model, optimizer, sampler)
            cell = root/"raw"/f"seed{seed}_task{task}"
            cell.mkdir(parents=True, exist_ok=True)
            progress = cell/"progress.pt"
            state = load(progress) if progress.exists() else {"done": [], "row_counts": {}}
            journal = Journal(cell, state["row_counts"])
            common = {**provenance(file_hash(source), config), "seed": seed, "checkpoint_task": task}
            fisher_path = cell/"fisher.pt"
            if not fisher_path.exists():
                fisher = empirical_fisher(model, data, args.device,
                                          10 if args.smoke else MECHANISMS["fisher_examples_per_permutation"],
                                          MECHANISMS["fisher_subset_seed"])
                atomic_save(fisher_path, {k: v.cpu() for k, v in fisher.items()})
            fisher = load(fisher_path)
            def retained_loss():
                return float(np.mean([r["loss"] for r in evaluate(model, data, "b", range(STREAMS["pool_size"]), args.device)]))
            baseline = retained_loss()
            for layer_id, (group, p, s) in enumerate(protected(optimizer)):
                layer = group["param_name"]
                scores = {"importance": s["imp"].flatten(), "fisher": fisher[layer].flatten().to(p.device),
                          "magnitude": p.detach().abs().flatten()}
                rankings = {name: torch.argsort(value, stable=True) for name, value in scores.items()}
                rank_path = cell/f"{layer}_rankings.csv"
                if not rank_path.exists():
                    ranks = {name: torch.argsort(order, stable=True).cpu().numpy() for name, order in rankings.items()}
                    write_csv(rank_path, pd.DataFrame({**common, "layer": layer, "coordinate": np.arange(p.numel()),
                                                      **{f"{n}_rank": v for n, v in ranks.items()}}))
                for fraction_id, fraction in enumerate(MECHANISMS["fractions"]):
                    count = max(1, int(p.numel()*fraction))
                    fixed_seed = MECHANISMS["probe_seed"]+seed*100000+task*1000+layer_id*100+fraction_id
                    selections = []
                    for name, order in rankings.items():
                        selections.extend([(name, "top", 0, order[-count:]), (name, "bottom", 0, order[:count])])
                    for rep in range(2 if args.smoke else MECHANISMS["random_groups"]):
                        generator = torch.Generator().manual_seed(fixed_seed+rep*10000)
                        selections.append(("random", "random", rep, torch.randperm(p.numel(), generator=generator)[:count].to(p.device)))
                    for comparator, kind, rep, indices in selections:
                        key = f"{layer}/{fraction}/{comparator}/{kind}/{rep}"
                        if key in state["done"]:
                            continue
                        damage, draws = antithetic_damage(p, indices, retained_loss, baseline, fixed_seed+rep*10000,
                                                         1 if args.smoke else MECHANISMS["noise_draws"],
                                                         MECHANISMS["noise_scale"])
                        journal.add("group_damage", {**common, "layer": layer, "fraction": fraction,
                                    "coordinate_count": count, "comparator": comparator, "group": kind,
                                    "replicate": rep, "damage": damage, "draw_damages": draws,
                                    "baseline_loss": baseline, "random_seed": fixed_seed+rep*10000})
                        state["done"].append(key)
                        state["row_counts"] = dict(journal.counts)
                        atomic_save(progress, state)
                    print(f"M1 seed={seed} task={task} {layer} fraction={fraction} complete", flush=True)
    damage = read_table(root, "group_damage")
    summaries = []
    keys = ["seed", "checkpoint_task", "layer", "fraction"]
    for block, group in damage.groupby(keys, sort=False):
        random = group[group.comparator == "random"].damage.to_numpy()
        for comparator in ("importance", "fisher", "magnitude"):
            subset = group[group.comparator == comparator]
            top = float(subset[subset.group == "top"].damage.iloc[0])
            bottom = float(subset[subset.group == "bottom"].damage.iloc[0])
            summaries.append({**dict(zip(keys, block)), **{k: group.iloc[0][k] for k in provenance("", config)},
                              "comparator": comparator, "top_imp_damage": top, "bottom_imp_damage": bottom,
                              "random_damage_mean": float(random.mean()),
                              "top_imp_percentile_among_random": float(100*((random < top).mean()+.5*(random == top).mean())),
                              "top_minus_bottom_damage": top-bottom, "top_minus_random_mean": top-float(random.mean())})
    summary = pd.DataFrame(summaries)
    write_csv(root/"raw"/"m1_group_damage.csv", damage)
    write_csv(root/"summary"/"m1_summary.csv", summary)
    # Stream the full coordinate table; concatenating all blocks in pandas would
    # duplicate millions of long provenance strings in memory.
    output = root/"raw"/"m1_fisher_rankings.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        for index, path in enumerate(sorted((root/"raw").glob("*/*_rankings.csv"))):
            with path.open(encoding="utf-8", newline="") as source:
                header = source.readline()
                if index == 0:
                    destination.write(header)
                shutil.copyfileobj(source, destination)
    os.replace(temporary, output)
    ci = []
    for (comp, fraction), g in summary.groupby(["comparator", "fraction"]):
        blocks = g.groupby(["seed", "checkpoint_task"], as_index=False).mean(numeric_only=True)
        for metric in ("top_minus_random_mean", "top_minus_bottom_damage", "top_imp_percentile_among_random"):
            ci.append({"comparator": comp, "fraction": fraction, "metric": metric,
                       "median": float(blocks[metric].median()), **cluster_bootstrap(blocks, metric)})
    write_csv(root/"summary"/"m1_cluster_intervals.csv", ci)
    from .plotting import faithfulness_plot
    faithfulness_plot(summary, root/"summary"/"m1_importance_faithfulness.png")
    atomic_json(root/"complete.json", {"config_hash": digest(config), "scientifically_verified": not args.allow_unverified_reference and not args.smoke})
