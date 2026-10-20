"""Configuration commits and complete-output checks; no scientific imports."""
from __future__ import annotations

import json
from pathlib import Path


def atomic_configuration(path: Path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def lock_e2_configuration(path: Path, proposed, *, require_existing=False):
    """Commit the scientific identity before the first resumable result exists."""
    # JSON-normalize tuples and other standard JSON sequences before comparison.
    proposed = json.loads(json.dumps(proposed))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("scientific_settings", "seeds"):
            if existing.get(key) != proposed[key]:
                raise RuntimeError("E2 scientific configuration/source changed; use a fresh output directory")
        return existing
    if require_existing:
        raise RuntimeError("E2 partial results have no compatible committed configuration; use a fresh output directory")
    atomic_configuration(path, proposed)
    return proposed


def require_complete_cell(scalars, tasks, pool, *, arm, seed, task_count,
                          total_steps, scalar_every, pool_size, committed_counts):
    """Require exact per-cell coverage before accepting a terminal checkpoint."""
    def select(rows):
        return [r for r in rows if str(r["arm"]) == arm and int(float(r["seed"])) == seed]

    scalar_rows, task_rows, pool_rows = map(select, (scalars, tasks, pool))
    expected_steps = set(range(scalar_every, total_steps + 1, scalar_every))
    expected_tasks = set(range(1, task_count + 1))
    if (len(scalar_rows) != len(expected_steps)
        or {int(float(r["step"])) for r in scalar_rows} != expected_steps
        or len(task_rows) != task_count
        or {int(float(r["task"])) for r in task_rows} != expected_tasks):
        raise RuntimeError("Terminal checkpoint has incomplete/duplicate scalar or task output")
    counts = {"scalars": len(scalar_rows), "tasks": len(task_rows)}
    if pool_size:
        expected_pool = {(t, d) for t in expected_tasks for d in range(pool_size)}
        observed_pool = {(int(float(r["task"])), int(float(r["pool_index"]))) for r in pool_rows}
        if len(pool_rows) != len(expected_pool) or observed_pool != expected_pool:
            raise RuntimeError("Terminal checkpoint has incomplete/duplicate pool output")
        for task in expected_tasks:
            current = [r for r in pool_rows if int(float(r["task"])) == task
                       and str(r["is_current"]).lower() in {"true", "1", "1.0"}]
            if len(current) != 1:
                raise RuntimeError("Pool output must contain exactly one current domain per task")
        counts["pool"] = len(pool_rows)
    if any(committed_counts.get(name) != count for name, count in counts.items()):
        raise RuntimeError("Terminal checkpoint output counts do not match committed rows")
