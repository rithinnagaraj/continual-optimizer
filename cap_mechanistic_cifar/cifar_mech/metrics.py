import numpy as np
from collections import deque
import torch
from torch.nn import functional as F
from .streams import permutation
from .constants import STREAMS

@torch.no_grad()
def evaluate(model, data, stream, domains, device, indices=None):
    was_training = model.training
    model.eval()
    x, y = data.test_x, data.test_y
    if indices is not None:
        x, y = x[indices], y[indices]
    results = []
    try:
        for domain in domains:
            correct, loss = 0, 0.
            perm = permutation(stream, domain)
            for start in range(0, len(y), STREAMS["eval_batch_size"]):
                xb, yb = x[start:start+STREAMS["eval_batch_size"]][:, perm].to(device), y[start:start+STREAMS["eval_batch_size"]].to(device)
                logits = model(xb)
                loss += F.cross_entropy(logits, yb, reduction="sum").item()
                correct += (logits.argmax(1) == yb).sum().item()
            results.append({"domain": int(domain), "accuracy": correct/len(y), "loss": loss/len(y)})
    finally:
        model.train(was_training)
    return results

def task_metrics(accuracies, horizon=5000):
    a = np.asarray(accuracies)
    if len(a) == 0:
        raise ValueError("Empty task")
    steps, event = horizon, False
    window, total = deque(maxlen=200), 0.
    for index, value in enumerate(a, 1):
        if len(window) == 200:
            total -= window[0]
        window.append(value)
        total += value
        if len(window) == 200 and total/200 >= .85:
            steps, event = index, True
            break
    return {"early_200": float(a[:200].mean()), "early_auc": float(a[:1000].mean()),
            "plasticity": float(a.mean()), "steps_to_85_percent": steps,
            "threshold_reached": event}

def end_retention(pool_rows, final_task=200, pool_size=STREAMS["pool_size"]):
    rows = [r for r in pool_rows if r["task"] == final_task and not r["is_current"]]
    if len(rows) != pool_size-1 or len({r["domain"] for r in rows}) != pool_size-1:
        raise ValueError(f"Expected {pool_size-1} non-current domains at the final boundary")
    return float(np.mean([r["accuracy"] for r in rows]))
