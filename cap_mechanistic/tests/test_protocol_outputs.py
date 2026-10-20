import pandas as pd
from cap_mech.runner import branches
from cap_mech.constants import MECHANISMS
from cap_mech.statistics import spearman

def test_preregistered_branch_counts():
    assert len(branches("m2", 42, 20)) == 6
    assert len(branches("m3", 42, 20)) == 5
    assert len(branches("m4", 42, 100)) == 10
    assert len(branches("m6", 42, 100)) == 5
    assert MECHANISMS["horizons"] == {"m2": 5, "m3": 1, "m4": 10, "m6": 10}

def test_spearman_uses_average_ties():
    assert abs(spearman([1, 2, 2, 4], [4, 3, 3, 1])+1) < 1e-12

def test_top_level_configs_match_installed_resources():
    from pathlib import Path
    root = Path(__file__).parents[1]
    for path in (root/"cap_mech/configs").glob("*.json"):
        assert path.read_bytes() == (root/"configs"/path.name).read_bytes()
