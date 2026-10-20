"""Plan one edge-extension round and freeze retention-optimal winners."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from experiment_e2 import read_csv
from post_phase1_common import (
    FREEZE_RECORD,
    GRID_RESULTS,
    METHODS,
    V2_ISOLATION,
    WINNER_ARTIFACT,
    atomic_write_json,
    canonical_json,
    post_guard,
    grid_knobs,
    load_json,
    load_pool_decision,
    normalize_config,
    retention_grids,
    sha256_file,
    update_freeze_record,
)


EDGE_PLAN = GRID_RESULTS.parent / "post_edge_extension_plan.json"
GRID_RUN_CONFIG = GRID_RESULTS.parent / "post_run_config.json"
def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else -math.inf


def _valid(row: Mapping[str, Any]) -> bool:
    diverged = str(row.get("diverged", "")).lower() in {"true", "1"}
    return (
        not diverged
        and str(row.get("status", "OK")) == "OK"
        and math.isfinite(_number(row, "validation_retention_metric"))
        and math.isfinite(_number(row, "validation_plasticity"))
    )


def _config_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["config_json"]))


def selection_pools() -> dict[str, list[dict[str, Any]]]:
    run_config = load_json(GRID_RUN_CONFIG)
    if run_config.get("isolation") != V2_ISOLATION:
        raise RuntimeError(
            "retention grid predates PP1-A1(C) isolation"
        )
    rows = read_csv(GRID_RESULTS)
    grids = retention_grids()
    expected = sum(len(value) for value in grids.values())
    base_rows = [
        row for row in rows if str(row.get("provenance")) == "registered_grid"
    ]
    if len(base_rows) != expected:
        raise RuntimeError(
            f"base retention grid has {len(base_rows)} rows; expected {expected}"
        )
    expected_configs = {
        (method, canonical_json(config))
        for method, cells in grids.items()
        for config in cells
    }
    observed_configs = {
        (str(row["method"]), str(row["config_json"])) for row in base_rows
    }
    if observed_configs != expected_configs:
        raise RuntimeError(
            "base retention grid contains a duplicate, omission, or "
            "unregistered configuration"
        )
    pools: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for order, row in enumerate(rows):
        provenance = str(row.get("provenance"))
        if provenance not in {"registered_grid", "edge_extension"}:
            raise RuntimeError(
                "PP1-A1(C) selection input is not a v2 grid cell: "
                f"{provenance}"
            )
        method = str(row["method"])
        if method not in pools:
            raise RuntimeError(f"unknown method in grid results: {method}")
        candidate = dict(row)
        candidate["selection_order"] = order
        candidate["config"] = _config_from_row(row)
        candidate["validation_scale"] = "pp1_a1_v2"
        pools[method].append(candidate)
    return pools


def choose(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [dict(item) for item in pool if _valid(item)]
    if not eligible:
        raise RuntimeError("selection pool contains no valid completed cell")
    return max(
        eligible,
        key=lambda item: (
            min(
                _number(item, "validation_retention_metric"),
                _number(item, "validation_plasticity"),
            ),
            mean(
                (
                    _number(item, "validation_retention_metric"),
                    _number(item, "validation_plasticity"),
                )
            ),
            -int(item["selection_order"]),
        ),
    )


def _extend_value(
    method: str,
    knob: str,
    values: Sequence[float],
    direction: str,
    rows: Sequence[Mapping[str, Any]],
) -> float | int | None:
    numeric = sorted(float(value) for value in values)
    if direction == "low":
        if knob == "weight_decay" and numeric[0] == 0:
            return None
        ratio = numeric[1] / numeric[0] if numeric[0] != 0 else 1.0
        candidate = numeric[0] / ratio
    else:
        if knob == "lambda" and any(
            str(row.get("method")) == "ewc_oracle"
            and math.isclose(float(_config_from_row(row).get("lambda", -1)), 3e5)
            and str(row.get("status")) == "EWC-DIVERGED"
            for row in rows
        ):
            return None
        if knob == "replay_count":
            return 127
        ratio = numeric[-1] / numeric[-2] if numeric[-2] != 0 else 1.0
        candidate = numeric[-1] * ratio
    lower = {
        "lr": 1e-5,
        "weight_decay": 0.0,
        "sigma": 1e-4,
        "k": 1e-7,
        "T_a": 100.0,
        "T_s": 100.0,
        "T_m": 100.0,
    }.get(knob, -math.inf)
    upper = {
        "lr": 1e-1,
        "replay_count": 127.0,
        "lambda": 1e6,
        "gamma": 1.0,
        "T_a": 1e6,
        "T_s": 1e6,
        "T_m": 1e6,
    }.get(knob, math.inf)
    clipped = min(max(candidate, lower), upper)
    edge = numeric[0] if direction == "low" else numeric[-1]
    if math.isclose(clipped, edge):
        return None
    if knob in {"replay_count", "T_a", "T_s", "T_m"}:
        return int(round(clipped))
    return clipped


def build_edge_plan() -> dict[str, Any]:
    if EDGE_PLAN.exists():
        existing = load_json(EDGE_PLAN)
        if (
            existing.get("schema_version") != 3
            or existing.get("isolation") != V2_ISOLATION
            or set(
            existing.get("methods", {})
            ) != set(METHODS)
        ):
            raise RuntimeError("existing edge-extension plan is invalid")
        print("The frozen one-round edge-extension plan already exists; reusing it")
        return existing
    pools = selection_pools()
    grids = retention_grids()
    all_rows = read_csv(GRID_RESULTS)
    plan: dict[str, Any] = {
        "schema_version": 3,
        "isolation": V2_ISOLATION,
        "methods": {},
    }
    for method in METHODS:
        winner = choose(pools[method])
        winner_config = normalize_config(method, winner["config"])
        dimensions = {
            knob: list(dict.fromkeys(config[knob] for config in grids[method]))
            for knob in grid_knobs(method)
        }
        additions: dict[str, Any] = {}
        edge_outcomes = []
        for knob, values in dimensions.items():
            low, high = values[0], values[-1]
            direction = None
            if winner_config[knob] == low:
                direction = "low"
            elif winner_config[knob] == high:
                direction = "high"
            if direction is None:
                continue
            added = _extend_value(method, knob, values, direction, all_rows)
            edge_outcomes.append(
                {
                    "knob": knob,
                    "direction": direction,
                    "new_value": added,
                    "cap_prevented_extension": added is None,
                }
            )
            if added is not None:
                additions[knob] = added

        expanded = {key: list(value) for key, value in dimensions.items()}
        for knob, value in additions.items():
            expanded[knob].append(value)
        new_configs = []
        existing = {canonical_json(config) for config in grids[method]}
        for combination in itertools.product(
            *(expanded[knob] for knob in grid_knobs(method))
        ):
            config = dict(grids[method][0])
            for knob, value in zip(grid_knobs(method), combination):
                config[knob] = value
            encoded = canonical_json(config)
            if encoded not in existing:
                new_configs.append(config)
                existing.add(encoded)
        plan["methods"][method] = {
            "pre_extension_winner": winner_config,
            "edge_outcomes": edge_outcomes,
            "new_configs": new_configs,
        }
    atomic_write_json(EDGE_PLAN, plan)
    return plan


def freeze_winners() -> dict[str, Any]:
    if not EDGE_PLAN.exists():
        raise RuntimeError("the one edge-extension round has not been planned")
    plan = load_json(EDGE_PLAN)
    if (
        plan.get("schema_version") != 3
        or plan.get("isolation") != V2_ISOLATION
        or set(plan.get("methods", {})) != set(METHODS)
    ):
        raise RuntimeError("edge-extension plan predates PP1-A1(C) isolation")
    planned = sum(
        len(entry.get("new_configs", []))
        for entry in plan.get("methods", {}).values()
    )
    extension_rows = [
        row
        for row in read_csv(GRID_RESULTS)
        if str(row.get("provenance")) == "edge_extension"
    ]
    if len(extension_rows) != planned:
        raise RuntimeError(
            f"edge extension has {len(extension_rows)} rows; expected {planned}"
        )
    expected_extensions = {
        (method, canonical_json(config))
        for method, entry in plan["methods"].items()
        for config in entry.get("new_configs", [])
    }
    observed_extensions = {
        (str(row["method"]), str(row["config_json"]))
        for row in extension_rows
    }
    if observed_extensions != expected_extensions:
        raise RuntimeError(
            "edge-extension output differs from the isolated frozen plan"
        )
    pools = selection_pools()
    decision = load_pool_decision()
    artifact: dict[str, Any] = {}
    for method in METHODS:
        winner = choose(pools[method])
        config = normalize_config(method, winner["config"])
        edge_outcomes = plan["methods"][method]["edge_outcomes"]
        still = []
        dimensions = {
            knob: list(
                dict.fromkeys(
                    config[knob] for config in retention_grids()[method]
                )
            )
            for knob in grid_knobs(method)
        }
        for outcome in edge_outcomes:
            if outcome["new_value"] is not None:
                dimensions[outcome["knob"]].append(outcome["new_value"])
        dimensions = {
            knob: sorted(values) for knob, values in dimensions.items()
        }
        for knob, values in dimensions.items():
            if config[knob] == values[0]:
                still.append({"knob": knob, "direction": "low"})
            if config[knob] == values[-1]:
                still.append({"knob": knob, "direction": "high"})
        tried = []
        for item in pools[method]:
            tried.append(
                {
                    "config": normalize_config(method, item["config"]),
                    "validation_retention_metric": (
                        None
                        if not math.isfinite(
                            _number(item, "validation_retention_metric")
                        )
                        else _number(item, "validation_retention_metric")
                    ),
                    "validation_plasticity": (
                        None
                        if not math.isfinite(
                            _number(item, "validation_plasticity")
                        )
                        else _number(item, "validation_plasticity")
                    ),
                    "diverged": not _valid(item),
                    "status": item.get("status"),
                    "provenance": item.get("provenance"),
                    "selection_order": int(item["selection_order"]),
                    "validation_scale": item.get(
                        "validation_scale", "pp1_a1_v2"
                    ),
                    "selection_min_score": (
                        None
                        if not _valid(item)
                        else min(
                            _number(item, "validation_retention_metric"),
                            _number(item, "validation_plasticity"),
                        )
                    ),
                }
            )
        artifact[method] = {
            "retention_config": config,
            "retention_metric": _number(
                winner, "validation_retention_metric"
            ),
            "validation_plasticity": _number(
                winner, "validation_plasticity"
            ),
            "selection_min_score": min(
                _number(winner, "validation_retention_metric"),
                _number(winner, "validation_plasticity"),
            ),
            "selection_mean_tiebreak": mean(
                (
                    _number(winner, "validation_retention_metric"),
                    _number(winner, "validation_plasticity"),
                )
            ),
            "winner_validation_scale": winner.get(
                "validation_scale", "pp1_a1_v2"
            ),
            "stream_b_pool_size": int(decision["chosen_pool_size"]),
            "isolation": V2_ISOLATION,
            "edge_extension_applied": bool(
                plan["methods"][method].get("new_configs")
            ),
            "edge_extension_outcomes": edge_outcomes,
            "still_at_edge": bool(still),
            "still_at_edge_details": still,
            "tried_configs": tried,
        }
    if WINNER_ARTIFACT.exists():
        existing = load_json(WINNER_ARTIFACT)
        if existing != artifact:
            raise RuntimeError(
                "the frozen winner artifact differs from the current selection"
            )
    else:
        atomic_write_json(WINNER_ARTIFACT, artifact)
    digest = sha256_file(WINNER_ARTIFACT)
    update_freeze_record(
        "winner_artifact",
        {"path": str(WINNER_ARTIFACT), "sha256": digest},
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan-extension", "freeze-winners"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with post_guard():
        if args.action == "plan-extension":
            build_edge_plan()
        else:
            freeze_winners()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
