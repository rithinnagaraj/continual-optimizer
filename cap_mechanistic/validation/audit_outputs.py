"""Check generated smoke artifacts, pairing, aggregation and validation labels."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cap_mech.runner import NAMES
from cap_mech.records import read_table

root = ROOT/"validation"/(sys.argv[1] if len(sys.argv) > 1 else "final-smoke")/"smoke"
gate = json.loads((root/"reproduction/reproduction_gate.json").read_text())
assert gate["passed"] is False
assert {r["metric"] for r in gate["rows"]} == {"P_historical_100step", "R"}
assert len(gate["P_dense_all_updates"]) == 5
pool = read_table(root/"reproduction", "pool")
assert pool.groupby(["seed", "task"]).size().eq(40).all()
assert pool[~pool.is_current].groupby(["seed", "task"]).size().eq(39).all()
checks = {"pool40_and_39_noncurrent": True, "smoke_gate_failed_as_required": True,
          "historical_and_dense_P_distinct": True}
for experiment, expected in (("m2", 30), ("m3", 25), ("m4", 50), ("m6", 25)):
    directory = root/NAMES[experiment]
    online = read_table(directory, "online")
    assert online.branch_id.nunique() == expected
    matched = online.groupby(["seed", "checkpoint_task", "task", "within_task_update"])
    assert matched.batch_indices_sha256.nunique().eq(1).all()
    assert matched.source_checkpoint_sha256.nunique().eq(1).all()
    checks[f"{experiment}_matched_batches_and_sources"] = True
replicates = pd.read_csv(root/NAMES["m4"]/"raw/m4_shuffle_replicates.csv")
blocks = pd.read_csv(root/NAMES["m4"]/"summary/m4_block_effects.csv")
assert len(replicates) == 40
shuffled = blocks[blocks.branch == "layer_shuffled"]
assert len(shuffled) == 5
for _, row in shuffled.iterrows():
    group = replicates[(replicates.seed == row.seed) & (replicates.checkpoint_task == row.checkpoint_task)]
    assert abs(row.delta_retention-group.delta_retention.mean()) < 1e-12
checks["shuffle_replicates_averaged_within_block"] = True
features = pd.read_csv(root/NAMES["m5"]/"raw/m5_task_features.csv")
tasks = read_table(root/"reproduction", "tasks")
for _, row in features.iterrows():
    previous = tasks[(tasks.stream == "a") & (tasks.seed == row.seed) & (tasks.task == row.task-1)]
    assert abs(row.previous_task_accuracy-previous.plasticity.iloc[0]) < 1e-12
checks["prospective_previous_accuracy_has_no_future_leakage"] = True
claims = pd.read_csv(root/"analysis/claim_evidence_table.csv")
assert not claims.status.eq("SUPPORTED").any()
assert len(list((root/"analysis/figures").glob("*.png"))) == 9
checks["unverified_report_and_all_nine_figures"] = True
path = root.parent/"output_audit.json"
path.write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
print(json.dumps(checks, indent=2))
