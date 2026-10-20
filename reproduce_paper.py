"""Run or inspect the data-producing pipeline for one paper table or figure.

Targets define experiment dependencies and the generated source artifacts
for each table or figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "paper_targets.json"


def main(argv=None):
    targets = json.loads(CATALOG.read_text(encoding="utf-8"))["targets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=sorted(targets))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute the listed training/analysis prerequisites with resume")
    parser.add_argument("--collect-only", action="store_true", help="Record hashes of already-generated source artifacts")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.list:
        for name, target in targets.items():
            print(f"{name}: {target['label']} [{target['status']}] {target['title']}")
        return 0
    if not args.target:
        parser.error("Specify a target or --list")
    if args.run and args.collect_only:
        parser.error("Choose --run or --collect-only")
    target = targets[args.target]
    print(json.dumps(target, indent=2), flush=True)
    if not (args.run or args.collect_only):
        return 0  # Default: inspect the plan without training or mutation.
    if args.run:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TORCH_HOME"] = str(ROOT / "data/torch_cache")
        for command in target["commands"]:
            command = [part.replace("{device}", args.device) for part in command]
            print("python " + shlex.join(command), flush=True)
            subprocess.run([sys.executable, "-u", *command], cwd=ROOT, env=env, check=True)
    output = ROOT / "results/paper" / args.target
    sources = {}
    missing = []
    for pattern in target["artifacts"]:
        found = sorted(path for path in ROOT.glob(pattern) if path.is_file())
        if not found:
            missing.append(pattern)
        for path in found:
            sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise FileNotFoundError("Required result artifacts missing: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    result = {"paper_target": args.target, "label": target["label"],
              "title": target["title"], "source_sha256": sources,
              "notes": target["notes"], "runtime": target["runtime"]}
    # A protocol table is a specification, not a measured outcome.
    if target["status"] == "protocol":
        result["protocol"] = json.loads((ROOT / "cap_mechanistic/cap_mech/configs/final_cap.json").read_text())
    (output / "source_artifacts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded source artifacts in results/paper/{args.target}/source_artifacts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
