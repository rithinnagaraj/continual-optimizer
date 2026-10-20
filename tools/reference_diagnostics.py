"""Generate endpoint diagnostics even when complete reference endpoints miss tolerance.

This never admits mechanistic experiments or changes their historical gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 451, 789, 2026)
METRICS = ("P_historical_100step", "R")


def source_hash(core):
    digest = hashlib.sha256()
    for path in sorted([*core.rglob("*.py"), *core.rglob("*.json")]):
        digest.update(path.relative_to(core).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_reference(output, core):
    directory = output/"reproduction"
    gate = json.loads((directory/"reproduction_gate.json").read_text(encoding="utf-8"))
    config = json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
    baseline = json.loads((core/"configs/final_cap.json").read_text(encoding="utf-8"))
    streams = json.loads((core/"configs/streams.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(baseline, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    expected_source = source_hash(core)
    if (config.get("smoke") or config.get("seeds") != list(SEEDS)
        or config.get("baseline") != baseline or config.get("streams") != streams
        or config.get("threads") != 1 or config.get("source_version") != expected_source
        or gate.get("source_version") != expected_source
        or not gate.get("valid_reference") or gate.get("baseline_config_sha256") != digest):
        raise ValueError("Reference is incomplete, synthetic or incompatible with this source/configuration")
    rows = gate.get("rows", [])
    expected_rows = {(seed, metric) for seed in SEEDS for metric in METRICS}
    if len(rows) != 10 or {(r["seed"], r["metric"]) for r in rows} != expected_rows:
        raise ValueError("Expected all five seeds and both endpoint metrics")
    tolerance = gate["tolerance"]
    for row in rows:
        values = (row["observed"], row["expected"], row["absolute_error"], tolerance)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Endpoint diagnostics contain missing/nonfinite measurements")
        error = abs(row["observed"]-row["expected"])
        if not math.isclose(error, row["absolute_error"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Endpoint error does not match its measurement")
        if bool(row["passed"]) != (error <= tolerance):
            raise ValueError("Endpoint pass flag is inconsistent")
    if bool(gate["passed"]) != all(r["passed"] for r in rows):
        raise ValueError("Historical gate result is inconsistent")
    expected_bank = {f"{stream}_seed{seed}_task{task}.pt" for seed in SEEDS
                     for stream, tasks in (("a", (20,50,100,150,200)), ("b", (50,100,150,200)))
                     for task in tasks}
    hashes = gate.get("checkpoint_hashes", {})
    if set(hashes) != expected_bank:
        raise ValueError("Reference checkpoint bank is incomplete")
    for name, expected in hashes.items():
        if hashlib.sha256((directory/"checkpoints"/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Reference checkpoint checksum mismatch")
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Validate an existing completed reference without training")
    args = parser.parse_args(argv)
    output = ROOT/"results/mechanistic_mnist"
    code = None
    if not args.collect_only:
        command = [sys.executable, "-u", "-m", "cap_mech.cli", "reproduce",
                   "--data-dir", "../data", "--output-dir", "../results/mechanistic_mnist",
                   "--device", args.device, "--threads", "1"]
        if args.resume:
            command.append("--resume")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        code = subprocess.run(command, cwd=ROOT/"cap_mechanistic", env=env).returncode
    gate = validate_reference(output, ROOT/"cap_mechanistic/cap_mech")
    result = {"diagnostic_inputs_complete": True, "reference_process_returncode": code,
              "historical_endpoint_passed": gate["passed"],
              "mechanisms_admitted": False,
              "note": "Complete endpoint measurements may be reported when tolerance fails; the historical gate is unchanged."}
    (output/"endpoint_diagnostic_status.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
