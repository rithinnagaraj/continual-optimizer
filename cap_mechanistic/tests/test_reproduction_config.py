import pytest
from cap_mech.constants import CAP, SEEDS, MECHANISMS, STREAMS, EXPECTED
from cap_mech.checkpointing import require_gate, gate_payload, atomic_json, source_version, file_hash

def test_frozen_config():
    assert CAP == {"lr": .0033333333333333335, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
                    "k": 7e-5, "T_a": 2000., "T_s": 2000., "T_m": 1000000., "importance": "raw_g2",
                    "protection": "coordinate_box", "release_update": "contest_blocked"}
    assert tuple(MECHANISMS["seeds"]) == SEEDS
    assert STREAMS["tasks"]*STREAMS["updates_per_task"] == 1000000

def test_gate_blocks_missing_failed_and_smoke(tmp_path):
    with pytest.raises(RuntimeError):
        require_gate(tmp_path)
    assert require_gate(tmp_path, allow=True) is False
    observed = {s: {"P_historical_100step": p, "R": r} for s, (p, r) in EXPECTED.items()}
    path = tmp_path/"reproduction"/"reproduction_gate.json"
    hashes = {}
    for s in SEEDS:
        for stream, tasks in (("a", (20, 50, 100, 150, 200)), ("b", (50, 100, 150, 200))):
            for t in tasks:
                cp = path.parent/"checkpoints"/f"{stream}_seed{s}_task{t}.pt"
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text("test checkpoint hash fixture")
                hashes[cp.name] = file_hash(cp)
    atomic_json(path, gate_payload(observed, False, source_version=source_version()))
    with pytest.raises(RuntimeError):
        require_gate(tmp_path)
    atomic_json(path, gate_payload(observed, True, source_version=source_version(), checkpoint_hashes=hashes))
    assert require_gate(tmp_path)
    cp.write_text("changed checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        require_gate(tmp_path)
    cp.write_text("test checkpoint hash fixture")
    observed[42]["P_historical_100step"] += .001
    atomic_json(path, gate_payload(observed, True, source_version=source_version(), checkpoint_hashes=hashes))
    with pytest.raises(RuntimeError):
        require_gate(tmp_path)

def test_every_command_dry_runs(capsys):
    from cap_mech.cli import COMMANDS, main
    for command in COMMANDS:
        assert main([command, "--dry-run"]) == 0
