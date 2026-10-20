"""Append-only JSONL tables with checkpoint-authoritative row counts."""
import json
import os
from pathlib import Path
import pandas as pd

class Journal:
    def __init__(self, root, counts=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = dict(counts or {})
        for path in self.root.glob("*.jsonl"):
            keep = self.counts.get(path.stem, 0)
            rows = path.read_text().splitlines()
            if len(rows) < keep:
                raise RuntimeError(f"Committed rows missing: {path}")
            temp = path.with_suffix(".tmp")
            temp.write_text("".join(r+"\n" for r in rows[:keep]), encoding="utf-8")
            os.replace(temp, path)

    def add(self, table, row):
        with (self.root/f"{table}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
        self.counts[table] = self.counts.get(table, 0)+1

    def rows(self, table):
        path = self.root/f"{table}.jsonl"
        return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

def read_table(root, table):
    rows = []
    for path in sorted(Path(root).glob(f"raw/*/{table}.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return pd.DataFrame(rows)

def write_csv(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(data)
    # Aggregates retain a digest of the complete input-checkpoint manifest.
    # Individual rows supplied by runners retain their actual checkpoint SHA.
    from .checkpointing import digest
    from .constants import CAP
    for parent in path.parents:
        manifest = parent/"run_config.json"
        if manifest.exists():
            config = json.loads(manifest.read_text())
            values = {"baseline_config_sha256": digest(CAP), "experiment_config_sha256": digest(config),
                      "source_version": config["source_version"],
                      "source_checkpoint_sha256": digest(config.get("checkpoint_hashes", {}))}
            for key, value in values.items():
                if key not in frame.columns:
                    frame[key] = value
            break
    temp = path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
