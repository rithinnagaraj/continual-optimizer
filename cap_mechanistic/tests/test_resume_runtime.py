from types import SimpleNamespace
import json
import torch
import pytest
from cap_mech.runner import train_cell, manifest
from cap_mech.data import Data
from cap_mech.records import Journal
from cap_mech.checkpointing import load

def test_interrupted_training_replays_exactly(tmp_path, monkeypatch):
    def args(path):
        return SimpleNamespace(smoke=True, seeds=[42], device="cpu", resume=True, threads=1,
                               output_dir=path, allow_unverified_reference=True)
    uninterrupted = args(tmp_path/"uninterrupted")
    interrupted = args(tmp_path/"interrupted")
    data = Data(None, True)
    config = manifest(uninterrupted, "reference")
    run = lambda a: train_cell(a, data, a.output_dir/"reproduction", config, "a", 42, "a_seed42__baseline")
    good = run(uninterrupted)
    original_add = Journal.add
    calls = []
    def fail_after_write(journal, table, row):
        original_add(journal, table, row)
        if table == "online":
            calls.append(row)
            if len(calls) == 2:
                raise RuntimeError("simulated interruption after uncommitted row")
    monkeypatch.setattr(Journal, "add", fail_after_write)
    with pytest.raises(RuntimeError, match="simulated"):
        run(interrupted)
    monkeypatch.setattr(Journal, "add", original_add)
    resumed = run(interrupted)
    for name in good["model"]:
        assert torch.equal(good["model"][name], resumed["model"][name])
    for idx, state in good["optimizer"]["state"].items():
        for name, expected in state.items():
            value = resumed["optimizer"]["state"][idx][name]
            assert torch.equal(expected, value) if isinstance(expected, torch.Tensor) else expected == value
    assert torch.equal(good["sampler"]["generator"], resumed["sampler"]["generator"])
    assert good["row_counts"] == resumed["row_counts"]
    rows = []
    for a in (uninterrupted, interrupted):
        file = a.output_dir/"reproduction/raw/a_seed42__baseline/online.jsonl"
        rows.append([json.loads(line) for line in file.read_text().splitlines()])
    assert [(r["accuracy"], r["batch_indices_sha256"]) for r in rows[0]] == [(r["accuracy"], r["batch_indices_sha256"]) for r in rows[1]]
