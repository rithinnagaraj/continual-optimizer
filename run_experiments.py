"""Portable experiment launcher. Planning requires only the Python standard library."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = (
    "e1", "e1-extra", "e2", "e3", "e4a", "e4a-discriminator", "phase1-prepare",
    "e4b", "phase1-stream-b", "e5", "e6", "post-phase1", "final",
    "mechanistic-mnist", "mechanistic-mnist-paper", "cifar-lr-sweep", "mechanistic-cifar", "mechanistic-cifar-selected", "mechanistic-cifar-paper", "horizon-audit",
)


def plan(stage, *, device="cpu", workers=8, resume=False, smoke=False):
    """Return (working directory, argv) pairs; never import the training runtime."""
    common = ["--data-dir", "data", "--device", device, "--num-workers", str(workers)]
    continued = ["--resume"] if resume else []
    jobs = []

    def script(name, args=(), cwd=ROOT):
        jobs.append((cwd, [name + ".py", *args]))

    if smoke and stage not in {"mechanistic-mnist", "mechanistic-cifar"}:
        raise ValueError("Synthetic smoke is available for mechanistic-mnist and mechanistic-cifar only")
    if stage in {"e1", "e1-extra", "e2", "e3", "e4a", "e4a-discriminator", "e4b", "e5"}:
        name = {"e1": "patched_e1", "e1-extra": "extra_e1"}.get(stage, "experiment_" + stage.replace("-", "_"))
        args = common + (continued if stage not in {"e1", "e1-extra", "e3"} else [])
        script(name, args)
    elif stage == "e6":
        script("experiment_e6", ["--device", device, *continued])
    elif stage == "phase1-prepare":
        script("phase1_tuning", [*common, *continued])
        script("phase1_retention_validation", [*common, *continued])
    elif stage == "phase1-stream-b":
        script("experiment_stream_b", [*common, *continued])
        script("phase1_grade")
    elif stage == "post-phase1":
        script("post_phase1_grid", ["preflight", *common])
        # A completed pool decision is immutable; downstream code verifies it.
        if not (resume and (ROOT / "results/post_phase1_v2/pool_size_decision.json").exists()):
            script("post_phase1_pool_preflight", [*common, *continued])
        script("post_phase1_grid", ["corrected-upgd", *common])
        script("post_upgd_corrected", common)
        script("post_phase1_grid", ["retention-grid", *common])
        script("post_phase1_select", ["plan-extension"])
        script("post_phase1_grid", ["edge-extension", *common])
        script("post_phase1_select", ["freeze-winners"])
        script("post_phase1_streams", common)
        script("post_phase1_report")
    elif stage == "final":
        script("cap_release_final", ["--retune", *common, *continued])
        script("cap_release_final", ["--train", "--all-seeds", *common, *continued])
        script("cap_release_final", ["--analyze"])
    elif stage == "horizon-audit":
        script("analyze_stream_horizon_plasticity")
    elif stage == "mechanistic-mnist-paper":
        # A complete discrepant reference is diagnostic evidence, followed by
        # the paper's independently checked bounded admission policy.
        script("tools/reference_diagnostics", ["--device", device, *continued])
        script("cap_mechanistic/diagnostics/run_validated_mechanisms",
               ["--output-dir", "results/mechanistic_mnist", "--data-dir", "data",
                "--device", device, *continued])
    elif stage == "mechanistic-mnist":
        cwd = ROOT / "cap_mechanistic"
        for command in ("reproduce", "m3-state-causality", "m2-k-dial", "m4-importance-assignment",
                        "m1-faithfulness", "m5-predict", "m6-release", "analyze"):
            args = ["-m", "cap_mech.cli", command, "--data-dir", "../data", "--device", device,
                    "--output-dir", "../results/mechanistic_mnist", *continued]
            if smoke:
                args.append("--smoke")
                if command != "reproduce":
                    args.append("--allow-unverified-reference")
            jobs.append((cwd, args))
    elif stage == "cifar-lr-sweep":
        cwd = ROOT / "cap_mechanistic_cifar"
        # The sweep deliberately requires an existing download and never uses the test split.
        script("sweep_learning_rates", ["--device", device, "--data-dir", "../data"], cwd)
    elif stage in {"mechanistic-cifar", "mechanistic-cifar-selected", "mechanistic-cifar-paper"}:
        args = ["-m", "cifar_mech.cli", "all", "--model", "both", "--device", device,
                "--data-dir", "../data", "--output-dir", "../results/mechanistic_cifar", *continued]
        if smoke:
            args.extend(["--smoke", "--seeds", "42"])
        if stage.endswith("-selected"):
            args.extend(["--lr-selection", "selected_lrs.json"])
        if stage.endswith("-paper"):
            args.append("--paper-settings")
        jobs.append((ROOT / "cap_mechanistic_cifar", args))
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", choices=STAGES)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without imports, downloads or writes")
    parser.add_argument("--device", default="cpu", help="Use cuda for full training; E6 requires CUDA")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Synthetic mechanistic validation only")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.list:
        print("\n".join(STAGES))
        return 0
    if not args.stages:
        parser.error("Specify a stage or --list")
    # Validate the entire plan before starting the first command.
    try:
        jobs = [job for stage in args.stages for job in plan(stage, device=args.device,
                workers=args.num_workers, resume=args.resume, smoke=args.smoke)]
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run and any("experiment_e6.py" in command for _, command in jobs) and not args.device.startswith("cuda"):
        parser.error("E6 requires --device cuda")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_HOME"] = str(ROOT / "data" / "torch_cache")
    for cwd, command in jobs:
        print(f"[{cwd.relative_to(ROOT).as_posix()}] python {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, "-u", *command], cwd=cwd, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
