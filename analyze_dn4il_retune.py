"""Combine new CAP/corrected-UPGD runs with read-only original AdamW/EWC."""
from __future__ import annotations
import argparse
from pathlib import Path

from dn4il_retune_protocol import BASE_ROOT, OUTPUT_ROOT, VERSION, BASE_VERSION, freeze_protocol, write_json, read_json


def analyze(baseline=BASE_ROOT, results=OUTPUT_ROOT):
    from analyze_frozen_resnet_head import load_runs, analyze_runs
    protocol = freeze_protocol(baseline, results)
    roots = {"adamw": baseline, "ewc": baseline, "cap": results, "upgd": results}
    versions = {"adamw": BASE_VERSION, "ewc": BASE_VERSION, "cap": VERSION, "upgd": VERSION}
    runs = load_runs(results, method_roots=roots, expected_versions=versions)
    for method in ("cap", "upgd"):
        selected = read_json(Path(results) / "selected_configs" / f"{method}.json")
        if selected.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("selection belongs to a different retune protocol")
        for run in runs[method].values():
            config = run["config"]
            if (config.get("study_protocol", {}).get("protocol_hash") != protocol["protocol_hash"]
                or config.get("study_protocol") != selected.get("study_protocol")
                or config.get("config") != selected.get("config")):
                raise ValueError("new reporting run differs from the selected retune configuration/code")
    provenance = {m: {"source_role": "reused original baseline" if m in ("adamw", "ewc") else "new retuned run",
                      "source_experiment": versions[m], "source_results_dir": str(Path(roots[m]).resolve())}
                  for m in roots}
    output = Path(results) / "analysis"
    analyze_runs(runs, output, labels={"adamw": "AdamW", "cap": "CAP", "upgd": "UPGD (corrected)", "ewc": "EWC-oracle"},
        provenance=provenance, disclosure=protocol["disclosure"] +
        " New CAP and corrected-UPGD each use 36 candidates on both development seeds. "
        "All four methods use identical reporting schedules, head initializations, data orders, and test banks. "
        "Runtime measurements are reused for original baselines and may come from a different execution session.")
    write_json(output / "source_provenance.json", {
        "experiment_version": VERSION, "protocol_hash": protocol["protocol_hash"],
        "methods": provenance, "frozen_baseline_sources": protocol["sources"],
    })
    print(f"Combined analysis written to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results-dir", type=Path, default=BASE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    analyze(args.baseline_results_dir, args.results_dir)
