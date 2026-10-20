"""Run every CLI end to end on synthetic inputs; no scientific training."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
output = ROOT/"validation"/(sys.argv[1] if len(sys.argv) > 1 else "cli-smoke")
output.mkdir(parents=True, exist_ok=True)
commands = ["reproduce", "m3-state-causality", "m2-k-dial", "m4-importance-assignment",
            "m1-faithfulness", "m5-predict", "m6-release", "analyze"]
results = []
for command in commands:
    argv = [sys.executable, "-m", "cap_mech.cli", command, "--smoke", "--output-dir", str(output)]
    if command != "reproduce":
        argv.append("--allow-unverified-reference")
    start = time.perf_counter()
    with (output/f"{command}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    row = {"command": command, "exit_code": result.returncode,
           "seconds": round(time.perf_counter()-start, 3), "synthetic": True}
    results.append(row)
    (output/"smoke_results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(row), flush=True)
    if result.returncode:
        print((output/f"{command}.log").read_text(), flush=True)
        raise SystemExit(result.returncode)
