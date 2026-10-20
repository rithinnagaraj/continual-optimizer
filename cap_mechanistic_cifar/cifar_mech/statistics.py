"""Seed-cluster inference and nested leave-one-seed-out ridge regression."""
import numpy as np
import pandas as pd
from .constants import MECHANISMS

def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    # Pearson correlation of average tied ranks is exactly Spearman's rho.
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])

def cluster_bootstrap(frame, value, resamples=10000, seed=610000):
    """Resample whole seeds, retaining every row of each sampled cluster."""
    frame = frame[["seed", value]].dropna()
    groups = [g[value].to_numpy(float) for _, g in frame.groupby("seed", sort=True)]
    if not groups:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_seeds": 0}
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    draws = np.random.default_rng(seed).integers(0, len(groups), (resamples, len(groups)))
    samples = sums[draws].sum(1)/counts[draws].sum(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {"mean": float(sums.sum()/counts.sum()), "ci_low": float(lo), "ci_high": float(hi), "n_seeds": len(groups)}

def paired_deltas(frame, baseline, outcomes):
    keys = ["seed", "checkpoint_task"]
    base = frame[frame.branch == baseline][keys+outcomes]
    if base.duplicated(keys).any():
        raise ValueError("Baseline must contain one observation per seed/checkpoint")
    merged = frame.merge(base, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    if len(merged) != len(frame):
        raise ValueError("Missing paired baseline")
    for value in outcomes:
        merged[f"delta_{value}"] = merged[value]-merged[f"{value}_baseline"]
    return merged

def ridge_fit(x, y, alpha):
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.
    z, intercept = (x-center)/scale, float(y.mean())
    if alpha == 0:
        coef = np.linalg.lstsq(z, y-intercept, rcond=None)[0]
    else:
        coef = np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]), z.T@(y-intercept))
    return center, scale, coef, intercept

def ridge_predict(fit, x):
    center, scale, coef, intercept = fit
    return (x-center)/scale@coef+intercept

def predictive_metrics(y, pred):
    denominator = np.sum((y-y.mean())**2)
    return {"MAE": float(np.mean(abs(y-pred))),
            "R2": float(1-np.sum((y-pred)**2)/denominator) if denominator > 0 else float("nan"),
            "spearman": spearman(y, pred)}

def nested_loso(frame, features, target, alphas=(0, .1, 1, 10, 100)):
    seeds = sorted(frame.seed.unique())
    if len(seeds) < 3:
        raise ValueError("Nested seed validation requires at least three seeds")
    predictions, metrics, coefficients = [], [], []
    for heldout in seeds:
        train, test = frame[frame.seed != heldout], frame[frame.seed == heldout]
        scores = []
        for alpha in alphas:
            errors = []
            for inner in sorted(train.seed.unique()):
                inner_train, inner_test = train[train.seed != inner], train[train.seed == inner]
                fit = ridge_fit(inner_train[features].to_numpy(float), inner_train[target].to_numpy(float), alpha)
                p = ridge_predict(fit, inner_test[features].to_numpy(float))
                errors.append(float(np.mean((p-inner_test[target].to_numpy(float))**2)))
            scores.append(np.mean(errors))
        alpha = alphas[int(np.argmin(scores))]
        fit = ridge_fit(train[features].to_numpy(float), train[target].to_numpy(float), alpha)
        p = ridge_predict(fit, test[features].to_numpy(float))
        for (_, row), value in zip(test.iterrows(), p):
            predictions.append({"seed": int(heldout), "task": int(row.task), "observed": float(row[target]),
                                "predicted": float(value), "alpha": alpha})
        metrics.append({"seed": int(heldout), "alpha": alpha, **predictive_metrics(test[target].to_numpy(float), p)})
        for name, center, scale, coef in zip(features, fit[0], fit[1], fit[2]):
            coefficients.append({"seed": int(heldout), "feature": name, "coefficient": float(coef),
                                 "training_mean": float(center), "training_scale": float(scale),
                                 "intercept": fit[3], "alpha": alpha})
    pred = pd.DataFrame(predictions)
    metrics.append({"seed": "overall", "alpha": None, **predictive_metrics(pred.observed.to_numpy(), pred.predicted.to_numpy())})
    return pred, pd.DataFrame(metrics), pd.DataFrame(coefficients)
