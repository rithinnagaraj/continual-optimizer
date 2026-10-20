import numpy as np
import pandas as pd
import pytest
from cap_mech.statistics import cluster_bootstrap, paired_deltas, nested_loso

def test_cluster_bootstrap_repeats_whole_seed():
    df = pd.DataFrame({"seed": [1, 1, 2, 2], "effect": [0., 2., 10., 12.]})
    a = cluster_bootstrap(df, "effect", 10000, 42)
    assert a == cluster_bootstrap(df, "effect", 10000, 42)
    assert a["mean"] == 6
    assert a["ci_low"] == 1 and a["ci_high"] == 11
    assert a["n_seeds"] == 2

def test_pairing_requires_baseline():
    df = pd.DataFrame({"seed": [1], "checkpoint_task": [20], "branch": ["shuffle"], "value": [1.]})
    with pytest.raises(ValueError):
        paired_deltas(df, "baseline", ["value"])

def test_nested_loso_standardizes_without_test_leakage():
    df = pd.DataFrame([{"seed": s, "task": t, "x": t+s, "y": 2*(t+s)+3} for s in range(5) for t in range(1, 5)])
    predictions, metrics, coefs = nested_loso(df, ["x"], "y")
    assert np.allclose(predictions.observed, predictions.predicted)
    for _, row in coefs.iterrows():
        assert row.training_mean == pytest.approx(df[df.seed != row.seed].x.mean())
    assert set(predictions.seed) == set(range(5))
