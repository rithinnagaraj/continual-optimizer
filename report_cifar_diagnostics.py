"""Verify completed priority-4 evidence and summarize paired seed effects."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiment_cifar_diagnostics import jobs, job_id, SEEDS, AGES, BRANCHES, identity, reference
from cap_mechanistic_cifar.cifar_mech.checkpointing import load, file_hash, digest, atomic_json
from cap_mechanistic_cifar.cifar_mech.metrics import task_metrics
from cap_mechanistic_cifar.cifar_mech.streams import old_tasks


def estimate(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (5,) and np.isfinite(a).all()
    samples = np.random.default_rng(610000).integers(0, 5, size=(10000, 5))
    lo, hi = np.quantile(a[samples].mean(axis=1), [.025, .975])
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "ci95_seed_bootstrap": [float(lo), float(hi)],
            "values_in_seed_order": a.tolist()}


def diagnostics(traces, low, high):
    by_step = {}
    for r in traces:
        if low <= r["step"] <= high:
            by_step.setdefault(r["step"], []).append(r)
    totals = []
    for rows in by_step.values():
        n = sum(r["coordinates"] for r in rows)
        proposed2 = sum(r["proposed_norm"]**2 for r in rows)
        removed2 = sum(r["removed_norm"]**2 for r in rows)
        totals.append({"proposed_rms": math.sqrt(proposed2/n),
                       "applied_rms": math.sqrt(sum(r["applied_norm"]**2 for r in rows)/n),
                       "removed_rms": math.sqrt(removed2/n),
                       "removed_over_proposed": math.sqrt(removed2/max(proposed2, 1e-60)),
                       **{key: sum(r[key]*r["coordinates"] for r in rows)/n for key in
                          ["bind_rate", "anchor_distance_over_radius_mean", "near_edge_fraction", "radius_mean", "importance_mean"]}})
    return {k: float(np.mean([r[k] for r in totals])) for k in totals[0]}


def report(root, verify_checkpoints=True):
    campaign = json.loads((root/"campaign.json").read_text())
    assert campaign["identity"] == identity()
    records, hashes, rows = {}, {}, []
    expected_steps = set(range(1, 21)) | set(range(100, 5001, 100))
    for j in jobs():
        name = job_id(j)
        directory = root/"jobs"/name
        p = directory/"result.json"
        r = json.loads(p.read_text())
        assert r["complete"] and r["step"] == 5000 and len(r["online"]) == 5000
        assert r["key"]["job"] == list(j) and r["key"]["campaign"] == digest(campaign)
        assert r["key"]["source_sha256"] == reference()[j[0]]["reproduction_gate.json"]["content"]["checkpoint_hashes"][f"a_seed{j[1]}_task{j[2]}.pt"]
        assert [x["step"] for x in r["online"]] == list(range(1, 5001))
        assert r["accuracies"] == [x["accuracy"] for x in r["online"]]
        assert task_metrics(r["accuracies"]) == r["metrics"]
        assert all(math.isfinite(x["loss"]) and 0 <= x["accuracy"] <= 1 and 0 <= x["bind_rate"] <= 1 for x in r["online"])
        layers = {x["layer"] for x in r["traces"]}
        assert len(layers) == (3 if j[0] == "mlp" else 5)
        assert len(r["traces"]) == len(layers)*70
        assert {(x["layer"], x["step"]) for x in r["traces"]} == {(l, s) for l in layers for s in expected_steps}
        assert all(math.isfinite(v) for x in r["traces"] for k, v in x.items() if k != "layer")
        assert set(r["retention"]) == {"0", "1000", "5000"}
        for rows_at_step in r["retention"].values():
            assert [x["domain"] for x in rows_at_step] == old_tasks(j[2])
            assert all(0 <= x["accuracy"] <= 1 and math.isfinite(x["loss"]) for x in rows_at_step)
        assert r["current_test"][0]["domain"] == j[2]+1
        if j[3] == "box_off":
            assert all(x["removed_norm"] == 0 for x in r["traces"])
        if verify_checkpoints:
            cp = load(directory/"resume.pt")
            assert cp["key"] == r["key"] and cp["record"] == {k: v for k, v in r.items() if k != "key"}
            assert all(torch.isfinite(t).all().item() for t in cp["model"].values())
            assert all(s["step"] == (j[2]+1)*5000 for s in cp["optimizer"]["state"].values())
            assert all(torch.isfinite(v).all().item() for s in cp["optimizer"]["state"].values()
                       for v in s.values() if torch.is_tensor(v))
            hashes[str((directory/"resume.pt").relative_to(root))] = file_hash(directory/"resume.pt")
        hashes[str(p.relative_to(root))] = file_hash(p)
        records[j] = r
        row = {"architecture": j[0], "seed": j[1], "age": j[2], "branch": j[3], **r["metrics"],
               "elapsed_seconds": r["elapsed_seconds"],
               "retention_1000": float(np.mean([x["accuracy"] for x in r["retention"]["1000"]])),
               "retention_5000": float(np.mean([x["accuracy"] for x in r["retention"]["5000"]]))}
        for window, low, high in [("transient", 1, 20), ("early", 100, 1000), ("late", 1100, 5000)]:
            row.update({window+"_"+k: v for k, v in diagnostics(r["traces"], low, high).items()})
        rows.append(row)
    for arch in ["mlp", "cnn"]:
        for seed in SEEDS:
            for age in AGES:
                base = records[(arch, seed, age, "baseline")]
                for branch in BRANCHES:
                    r = records[(arch, seed, age, branch)]
                    assert r["retention"]["0"] == base["retention"]["0"]
                    assert [x["batch_sha256"] for x in r["online"]] == [x["batch_sha256"] for x in base["online"]]
    index = {(r["architecture"], r["seed"], r["age"], r["branch"]): r for r in rows}
    metrics = ["early_auc", "early_200", "plasticity", "retention_1000", "retention_5000",
               "transient_removed_over_proposed", "early_removed_over_proposed", "late_removed_over_proposed",
               "early_bind_rate", "late_bind_rate", "early_anchor_distance_over_radius_mean"]
    summaries, effects, trends = [], [], []
    for arch in ["mlp", "cnn"]:
        for age in AGES:
            for branch in BRANCHES:
                summaries.append({"architecture": arch, "age": age, "branch": branch,
                                  **{m: estimate([index[(arch,s,age,branch)][m] for s in SEEDS]) for m in metrics}})
                if branch != "baseline":
                    effects.append({"architecture": arch, "age": age, "branch": branch,
                                    **{m: estimate([index[(arch,s,age,branch)][m]-index[(arch,s,age,"baseline")][m] for s in SEEDS]) for m in metrics}})
        for branch in BRANCHES:
            trends.append({"architecture": arch, "branch": branch, "contrast": "age150-minus-age20",
                           **{m: estimate([index[(arch,s,150,branch)][m]-index[(arch,s,20,branch)][m] for s in SEEDS]) for m in metrics}})
    output = {"complete_jobs": len(rows), "seeds": SEEDS, "elapsed_job_hours": sum(r["elapsed_seconds"] for r in rows)/3600,
              "reporter_sha256": file_hash(Path(__file__)),
              "checkpoint_payloads_verified": verify_checkpoints, "campaign_sha256": file_hash(root/"campaign.json"),
              "artifact_hashes": hashes, "per_job": rows, "summaries": summaries, "paired_effects": effects,
              "age_contrasts": trends, "uncertainty": "Percentile bootstrap over 5 seeds, 10000 resamples; descriptive, no multiplicity adjustment"}
    atomic_json(root/"verified_report.json", output)
    print(json.dumps({k: output[k] for k in ["complete_jobs", "elapsed_job_hours", "checkpoint_payloads_verified"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/cifar_diagnostics"))
    p.add_argument("--without-checkpoints", action="store_true", help="For independently checking downloaded nonbinary evidence")
    a = p.parse_args()
    report(a.output, not a.without_checkpoints)
