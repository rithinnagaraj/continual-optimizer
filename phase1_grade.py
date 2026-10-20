"""Grade the P1-A7 Phase-1 primary gate and its conditional CAP re-tune."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from experiment_e2 import read_csv
from experiment_e4a import FULL_SEEDS, atomic_write_json, utc_now
from experiment_stream_b import (
    ARMS as STREAM_B_UNCONDITIONAL_ARMS,
    EWC_DISCLOSURE,
    REPLAY_LABEL,
    end_retention_by_seed,
)
from phase1_arms import (
    CONDITIONAL_RETUNE_ARM,
    FINAL_STREAM_AMENDMENT,
    P1_A3_DISCLOSURE,
    P1_A4_DISCLOSURE,
    P1_A5_DISCLOSURE,
    P1_A6_DISCLOSURE,
    P1_A7_DISCLOSURE,
    P1_A7_FAILURE_BRANCH_STATEMENT,
    P1_A7_FREEZE_STATEMENT,
    P1_FULL_SEEDS,
    RETENTION_FOCUSED_LABEL,
    RETENTION_FOCUSED_SUFFIX,
    TUNED_METHODS,
    l2_descope_guard,
    load_conditional_retune_request,
    load_tuned_configs,
    resolved_config_for_arm,
    validate_p1_a7_artifact,
    validation_metric_for_endpoint,
)


STREAM_A_START_STEP = 500_001
STREAM_A_END_STEP = 1_000_000
PLASTICITY_TOLERANCE = 0.02
RETENTION_MARGIN = 0.02
EWC_TOLERANCE = 0.01
REPLAY_TOLERANCE = 0.01

PRIMARY_STREAM_A_ARM = {
    "adamw": "adamw",
    "adam_replay": "adam_replay",
    "upgd": "upgd",
    "ewc_oracle": "ewc_oracle",
    "l2_init": "l2_init",
}


def stream_a_unconditional_arms(
    tuned_configs: Mapping[str, Any],
) -> tuple[str, ...]:
    arms = [
        "adamw",
        "cap_leak",
        "cap_ratchet",
        "cap_release_10000",
        "adam_replay",
        "upgd",
        "ewc_oracle",
    ]
    release_arm = str(tuned_configs["stream_a_release_arm"])
    if release_arm not in arms:
        arms.append(release_arm)
    return tuple(arms)


def validate_stream_a_append_completion(
    task_rows: Sequence[Mapping[str, Any]],
    tuned_configs: Mapping[str, Any],
    *,
    include_conditional_retune: bool,
) -> None:
    arms = list(stream_a_unconditional_arms(tuned_configs))
    if include_conditional_retune:
        arms.append(CONDITIONAL_RETUNE_ARM)
    expected = {
        (arm, int(seed), task)
        for arm in arms
        for seed in P1_FULL_SEEDS
        for task in range(1, 201)
    }
    actual = {
        (
            str(row["arm"]),
            int(float(row["seed"])),
            int(float(row["task"])),
        )
        for row in task_rows
    }
    if len(task_rows) != len(expected) or actual != expected:
        raise RuntimeError(
            "Stream-A P1-A7 coverage is incomplete, duplicated, or contains "
            "an arm outside the frozen roster"
        )


def validate_stream_b_completion(
    pool_rows: Sequence[Mapping[str, Any]],
    *,
    include_conditional_retune: bool,
) -> None:
    arms = list(STREAM_B_UNCONDITIONAL_ARMS)
    if include_conditional_retune:
        arms.append(CONDITIONAL_RETUNE_ARM)
    expected = {
        (arm, int(seed), task, pool_index)
        for arm in arms
        for seed in P1_FULL_SEEDS
        for task in range(1, 201)
        for pool_index in range(10)
    }
    actual = {
        (
            str(row["arm"]),
            int(float(row["seed"])),
            int(float(row["task"])),
            int(float(row["pool_index"])),
        )
        for row in pool_rows
    }
    if len(pool_rows) != len(expected) or actual != expected:
        raise RuntimeError(
            "Stream-B P1-A7 coverage is incomplete, duplicated, or contains "
            "an arm outside the frozen roster"
        )


def stream_a_online_by_seed(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str]
) -> dict[str, dict[int, float]]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        step = int(float(row["step"]))
        if arm in arms and STREAM_A_START_STEP <= step <= STREAM_A_END_STEP:
            grouped[(arm, int(float(row["seed"])))].append(
                (step, float(row["batch_accuracy"]))
            )
    expected_steps = set(range(500_100, STREAM_A_END_STEP + 1, 100))
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (arm, seed), values in grouped.items():
        observed_steps = {step for step, _accuracy in values}
        if len(values) != len(expected_steps) or observed_steps != expected_steps:
            raise RuntimeError(
                f"Stream-A clause requires the exact 5,000-row logged grid "
                f"for {arm}/{seed}; found {len(values)}"
            )
        result[arm][seed] = mean(accuracy for _step, accuracy in values)
    return dict(result)


def _five_seed_mean(values: Mapping[int, float], label: str) -> float:
    if set(values) != set(P1_FULL_SEEDS):
        raise RuntimeError(f"{label} does not have the full five-seed set")
    return mean(values[seed] for seed in P1_FULL_SEEDS)


def validation_flexibility(
    tuned_configs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in TUNED_METHODS:
        entry = tuned_configs["methods"][method]
        p = entry["primary"]
        r = entry["retention_focused"]
        p_plasticity = float(p["plasticity_metric"])
        r_plasticity = float(r["plasticity_metric"])
        p_retention = validation_metric_for_endpoint(
            tuned_configs, method, "primary"
        )
        r_retention = validation_metric_for_endpoint(
            tuned_configs, method, "retention_focused"
        )
        result[method] = {
            "P_config": dict(p["config"]),
            "R_config": dict(r["config"]),
            "P_plasticity": p_plasticity,
            "P_retention": p_retention,
            "R_plasticity": r_plasticity,
            "R_retention": r_retention,
            "delta_plasticity": r_plasticity - p_plasticity,
            "delta_retention": r_retention - p_retention,
            "endpoints_coincide": bool(entry["endpoints_coincide"]),
            "scale": "validation-scale",
        }
    return result


def grade_phase1(
    *,
    stream_a_rows: Sequence[Mapping[str, Any]],
    stream_b_pool_rows: Sequence[Mapping[str, Any]],
    tuned_configs: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the binding P1-A7 grading order to complete final-stream rows."""
    release_a_arm = str(tuned_configs["stream_a_release_arm"])
    required_a = (
        release_a_arm,
        "adamw",
        "upgd",
        "cap_leak",
        CONDITIONAL_RETUNE_ARM,
    )
    online_by_seed = stream_a_online_by_seed(stream_a_rows, required_a)
    for arm in (release_a_arm, "adamw", "upgd", "cap_leak"):
        if arm not in online_by_seed:
            raise RuntimeError(f"missing Stream-A grading arm: {arm}")
    online = {
        arm: _five_seed_mean(values, f"Stream-A {arm}")
        for arm, values in online_by_seed.items()
    }

    retention_by_seed = end_retention_by_seed(stream_b_pool_rows)
    for arm in STREAM_B_UNCONDITIONAL_ARMS:
        if arm not in retention_by_seed:
            raise RuntimeError(f"missing Stream-B grading arm: {arm}")
    retention = {
        arm: _five_seed_mean(values, f"Stream-B {arm}")
        for arm, values in retention_by_seed.items()
    }

    guard = l2_descope_guard(tuned_configs)
    replay_absent = (
        retention["adam_replay"] <= retention["adamw"] + REPLAY_TOLERANCE
    )
    ewc_suspect = (
        retention["ewc_oracle"] <= retention["adamw"] + EWC_TOLERANCE
    )

    retune_a_present = CONDITIONAL_RETUNE_ARM in online_by_seed
    retune_b_present = CONDITIONAL_RETUNE_ARM in retention_by_seed
    if retune_a_present != retune_b_present:
        raise RuntimeError(
            "conditional CAP re-tune exists on only one final stream"
        )

    primary_result: dict[str, Any] = {
        "verdict": "NOT-EVALUATED",
        "reason": None,
    }
    retune_result: dict[str, Any] | None = None
    if not guard["passed"]:
        primary_result["reason"] = "L2INIT-DESCOPE-INVALID"
        if retune_a_present:
            raise RuntimeError(
                "conditional re-tune data exists while the L2 guard blocks "
                "primary grading"
            )
        overall = "L2INIT-DESCOPE-INVALID"
    elif replay_absent:
        primary_result["reason"] = "REPLAY-CONTROL-ABSENT"
        if retune_a_present:
            raise RuntimeError(
                "conditional re-tune data exists while REPLAY-CONTROL-ABSENT "
                "blocks primary grading"
            )
        overall = "NOT-GRADED"
    else:
        plasticity_opponents = {
            arm: online[arm] for arm in ("adamw", "upgd", "cap_leak")
        }
        plasticity_best_arm = max(
            plasticity_opponents, key=plasticity_opponents.get
        )
        plasticity_threshold = (
            plasticity_opponents[plasticity_best_arm] - PLASTICITY_TOLERANCE
        )
        retention_opponents = {
            arm: retention[arm] for arm in ("adamw", "upgd")
        }
        retention_best_arm = max(
            retention_opponents, key=retention_opponents.get
        )
        retention_threshold = (
            retention_opponents[retention_best_arm] + RETENTION_MARGIN
        )
        primary_plasticity_pass = (
            online[release_a_arm] >= plasticity_threshold
        )
        primary_retention_pass = (
            retention["cap_release"] >= retention_threshold
        )
        primary_pass = primary_plasticity_pass and primary_retention_pass
        primary_result = {
            "plasticity": {
                "cap_release": online[release_a_arm],
                "best_opponent_arm": plasticity_best_arm,
                "best_opponent": plasticity_opponents[plasticity_best_arm],
                "threshold": plasticity_threshold,
                "passes": primary_plasticity_pass,
            },
            "retention": {
                "cap_release": retention["cap_release"],
                "best_opponent_arm": retention_best_arm,
                "best_opponent": retention_opponents[retention_best_arm],
                "threshold": retention_threshold,
                "passes": primary_retention_pass,
            },
            "verdict": "PASS" if primary_pass else "FAIL",
            "reason": None,
        }
        if primary_pass:
            if retune_a_present:
                raise RuntimeError(
                    "conditional re-tune data exists even though the primary "
                    "passed"
                )
            overall = "PASS"
        elif not retune_a_present:
            overall = "CONDITIONAL-RETUNE-REQUIRED"
        else:
            retune_a = _five_seed_mean(
                online_by_seed[CONDITIONAL_RETUNE_ARM],
                f"Stream-A {CONDITIONAL_RETUNE_ARM}",
            )
            retune_b = _five_seed_mean(
                retention_by_seed[CONDITIONAL_RETUNE_ARM],
                f"Stream-B {CONDITIONAL_RETUNE_ARM}",
            )
            retune_result = {
                "arm": CONDITIONAL_RETUNE_ARM,
                "label": RETENTION_FOCUSED_LABEL,
                "stream_a_online": retune_a,
                "stream_b_end_retention": retune_b,
                "plasticity_threshold": plasticity_threshold,
                "retention_threshold": retention_threshold,
                "plasticity_passes": retune_a >= plasticity_threshold,
                "retention_passes": retune_b >= retention_threshold,
            }
            retune_result["passes"] = bool(
                retune_result["plasticity_passes"]
                and retune_result["retention_passes"]
            )
            overall = (
                "PASS-AFTER-PREDECLARED-RETUNE"
                if retune_result["passes"]
                else "FAIL"
            )

    upgd_plasticity_gap = online[release_a_arm] - online["upgd"]
    upgd_retention_gap = retention["cap_release"] - retention["upgd"]
    if upgd_plasticity_gap >= -PLASTICITY_TOLERANCE and upgd_retention_gap >= 0.02:
        upgd_headline = "matches plasticity, adds memory"
    elif upgd_plasticity_gap < -0.05:
        upgd_headline = "plasticity problem"
    elif upgd_retention_gap < 0:
        upgd_headline = "thesis problem"
    else:
        upgd_headline = "mixed"

    return {
        "seeds": list(P1_FULL_SEEDS),
        "stream_a_release_arm": release_a_arm,
        "stream_a_online_by_seed": online_by_seed,
        "stream_a_online": online,
        "stream_b_end_retention_by_seed": retention_by_seed,
        "stream_b_end_retention": retention,
        "controls": {
            "l2_init_descope_guard": guard,
            "replay_control_absent": replay_absent,
            "replay_verdict": (
                "REPLAY-CONTROL-ABSENT" if replay_absent else "PASS"
            ),
            "ewc_integration_suspect": ewc_suspect,
            "ewc_verdict": (
                "EWC-INTEGRATION-SUSPECT" if ewc_suspect else "PASS"
            ),
        },
        "kill_gate": {
            "primary": primary_result,
            "conditional_retune": retune_result,
            "conditional_retune_required": (
                overall == "CONDITIONAL-RETUNE-REQUIRED"
            ),
            "verdict": overall,
        },
        "cap_policy_ablation": {
            arm: retention[arm]
            for arm in ("cap_leak", "cap_ratchet", "cap_release")
        },
        "validation_flexibility": validation_flexibility(tuned_configs),
        "headlines": {
            "retention_graded": guard["passed"] and not replay_absent,
            "upgd": {
                "plasticity_gap": upgd_plasticity_gap,
                "retention_gap": upgd_retention_gap,
                "interpretation": upgd_headline,
            },
            "ewc_oracle": {
                "retention_gap": retention["cap_release"]
                - retention["ewc_oracle"],
                "matches_boundary_privileged": (
                    None
                    if ewc_suspect
                    else retention["cap_release"] - retention["ewc_oracle"]
                    >= -EWC_TOLERANCE
                ),
            },
            "adam_replay": {
                "retention_gap": retention["cap_release"]
                - retention["adam_replay"],
                "storage_free_matches_storage": (
                    retention["cap_release"] - retention["adam_replay"]
                    >= -REPLAY_TOLERANCE
                ),
            },
        },
    }


def conditional_retune_request(
    *,
    grade: Mapping[str, Any],
    tuned_config_path: Path,
    tuned_configs: Mapping[str, Any],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if grade["kill_gate"]["verdict"] != "CONDITIONAL-RETUNE-REQUIRED":
        raise ValueError("the primary result does not authorize a re-tune")
    primary = grade["kill_gate"]["primary"]
    if primary["verdict"] != "FAIL":
        raise ValueError("conditional branch requires a primary FAIL")
    return {
        "schema_version": 1,
        "amendment": FINAL_STREAM_AMENDMENT,
        "created_at_utc": (
            existing.get("created_at_utc", utc_now()) if existing else utc_now()
        ),
        "status": "CONDITIONAL-RETUNE-REQUIRED",
        "primary_kill_gate_verdict": "FAIL",
        "failed_clauses": [
            clause
            for clause in ("plasticity", "retention")
            if not bool(primary[clause]["passes"])
        ],
        "arm": CONDITIONAL_RETUNE_ARM,
        "label": RETENTION_FOCUSED_LABEL,
        "config": resolved_config_for_arm(
            CONDITIONAL_RETUNE_ARM, tuned_configs
        ),
        "seeds": list(P1_FULL_SEEDS),
        "tuned_config_path": str(tuned_config_path),
        "tuned_config_sha256": hashlib.sha256(
            tuned_config_path.read_bytes()
        ).hexdigest(),
        "failure_branch_statement": P1_A7_FAILURE_BRANCH_STATEMENT,
    }


def build_report(grade: Mapping[str, Any]) -> str:
    controls = grade["controls"]
    kill = grade["kill_gate"]
    primary = kill["primary"]
    retention = grade["stream_b_end_retention"]
    lines = [
        "# Phase 1 report",
        "",
        "## Protocol header",
        "",
        (
            "Forced deviations: "
            f"{P1_A3_DISCLOSURE} {P1_A4_DISCLOSURE} {P1_A5_DISCLOSURE} "
            f"{P1_A6_DISCLOSURE} {P1_A7_DISCLOSURE}"
        ),
        f"Seed set run: {tuple(grade['seeds'])}.",
        "Full preregistered five-seed grade: True.",
        (
            "Settings: MLP-300; Stream A permutations 100000+t; Stream B "
            "ten-permutation recurrence pool and seed-555 schedule; batch 128; "
            "all configurations read from phase1_tuned_configs.json."
        ),
        P1_A7_FREEZE_STATEMENT,
        EWC_DISCLOSURE,
        f"adam_replay: {REPLAY_LABEL}.",
        (
            "E4b showed that the endless stream places no value on retention; "
            "Stream B is therefore the load-bearing retention test, with all "
            "pre-registered thresholds unchanged."
        ),
        "",
        "## Binding validity order",
        "",
    ]
    guard = controls["l2_init_descope_guard"]
    lines.extend(
        [
            (
                "- l2_init guard: l2_init="
                f"{guard['l2_init_primary_validation_retention']:.7f}; "
                f"max(adamw={guard['opponents']['adamw']:.7f}, "
                f"upgd={guard['opponents']['upgd']:.7f})="
                f"{guard['binding_value']:.7f}; verdict **{guard['verdict']}**."
            ),
            (
                "The predeclared validation-scale L2 descope guard passed."
                if guard["passed"]
                else "L2INIT-DESCOPE-INVALID."
            ),
            (
                "- Replay control: adam_replay="
                f"{retention['adam_replay']:.6f}; adamw+0.01="
                f"{retention['adamw'] + REPLAY_TOLERANCE:.6f}; verdict "
                f"**{controls['replay_verdict']}**."
            ),
            (
                "- EWC integration: ewc_oracle="
                f"{retention['ewc_oracle']:.6f}; adamw+0.01="
                f"{retention['adamw'] + EWC_TOLERANCE:.6f}; verdict "
                f"**{controls['ewc_verdict']}**."
            ),
            "",
            "## Primary kill gate",
            "",
        ]
    )
    if primary["verdict"] == "NOT-EVALUATED":
        lines.append(
            "- Primary kill gate was not evaluated because "
            f"**{primary['reason']}** blocks grading."
        )
    else:
        lines.extend(
            [
                (
                    "- Clause (i): primary CAP="
                    f"{primary['plasticity']['cap_release']:.6f}; best "
                    f"opponent {primary['plasticity']['best_opponent_arm']}="
                    f"{primary['plasticity']['best_opponent']:.6f}; threshold "
                    f">={primary['plasticity']['threshold']:.6f}; passes="
                    f"{primary['plasticity']['passes']}."
                ),
                (
                    "- Clause (ii): primary CAP="
                    f"{primary['retention']['cap_release']:.6f}; best "
                    f"opponent {primary['retention']['best_opponent_arm']}="
                    f"{primary['retention']['best_opponent']:.6f}; threshold "
                    f">={primary['retention']['threshold']:.6f}; passes="
                    f"{primary['retention']['passes']}."
                ),
                f"- Primary verdict: **{primary['verdict']}**.",
            ]
        )
    lines.append(f"- Phase-1 verdict: **{kill['verdict']}**.")
    if kill["conditional_retune_required"]:
        lines.extend(
            [
                (
                    "The valid primary failed. The frozen conditional CAP "
                    "endpoint is now authorized on both streams; no other "
                    "configuration may enter this branch."
                ),
                (
                    "Run both stream programs with `--conditional-retune "
                    "--resume`, then rerun `phase1_grade.py`."
                ),
            ]
        )
    elif kill["conditional_retune"] is not None:
        retune = kill["conditional_retune"]
        lines.append(
            f"- Conditional re-tune: {retune['arm']} "
            f"({RETENTION_FOCUSED_LABEL}); Stream-A="
            f"{retune['stream_a_online']:.6f} "
            f"(passes={retune['plasticity_passes']}); Stream-B="
            f"{retune['stream_b_end_retention']:.6f} "
            f"(passes={retune['retention_passes']}); combined="
            f"{retune['passes']}."
        )

    lines.extend(["", "## Headline comparisons", ""])
    if not grade["headlines"]["retention_graded"]:
        lines.append(
            "Retention headline comparisons are withheld because a binding "
            "validity control blocks grading."
        )
    else:
        upgd = grade["headlines"]["upgd"]
        ewc = grade["headlines"]["ewc_oracle"]
        replay = grade["headlines"]["adam_replay"]
        lines.extend(
            [
                (
                    "- UPGD: Stream-A CAP-UPGD gap="
                    f"{upgd['plasticity_gap']:+.6f} (required >=-0.020000); "
                    "Stream-B CAP-UPGD gap="
                    f"{upgd['retention_gap']:+.6f} (required >=+0.020000); "
                    f"interpretation: **{upgd['interpretation']}**."
                ),
                (
                    "- EWC oracle: Stream-B CAP-EWC gap="
                    f"{ewc['retention_gap']:+.6f}; boundary-privileged "
                    f"match within 0.01={ewc['matches_boundary_privileged']}; "
                    f"integration verdict={controls['ewc_verdict']}."
                ),
                (
                    "- Adam replay: Stream-B CAP-replay gap="
                    f"{replay['retention_gap']:+.6f}; storage-free match "
                    "within 0.01="
                    f"{replay['storage_free_matches_storage']}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## CAP retention-policy ablation (descriptive)",
            "",
            "| Policy | Stream-B end retention |",
            "|---|---:|",
        ]
    )
    for arm, value in grade["cap_policy_ablation"].items():
        lines.append(f"| {arm} | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Validation-scale hyperparameter flexibility (descriptive only)",
            "",
            (
                "At validation scale, P1-A5 tests whether CAP offers greater "
                "hyperparameter-controlled plasticity–retention flexibility."
            ),
            "No final-stream flexibility claim is made.",
            "",
            (
                "| Method / R designation | P plasticity | P retention | "
                "R plasticity | R retention | Δ plasticity | Δ retention | "
                "Coincide |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for method in TUNED_METHODS:
        item = grade["validation_flexibility"][method]
        designation = (
            "coincident (no sensitivity endpoint)"
            if item["endpoints_coincide"]
            else f"{method}{RETENTION_FOCUSED_SUFFIX} "
            f"({RETENTION_FOCUSED_LABEL}; validation-scale)"
        )
        lines.append(
            f"| {method}: {designation} | {item['P_plasticity']:.6f} | "
            f"{item['P_retention']:.6f} | {item['R_plasticity']:.6f} | "
            f"{item['R_retention']:.6f} | "
            f"{item['delta_plasticity']:+.6f} | "
            f"{item['delta_retention']:+.6f} | "
            f"{'yes' if item['endpoints_coincide'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "All flexibility comparisons are endpoint-matched and use "
                "only the completed recurrent-validation campaign. See "
                "`flexibility_plot.svg`."
            ),
            "",
            (
                "The Stream-A ratchet/leak verdicts remain banked in "
                "results/stream_a/report_e4b.md and are not re-graded here."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_flexibility_svg(
    flexibility: Mapping[str, Mapping[str, Any]],
) -> str:
    width, height = 900, 620
    left, right, top, bottom = 90, 40, 45, 80
    points = [
        (float(item[key_x]), float(item[key_y]))
        for item in flexibility.values()
        for key_x, key_y in (
            ("P_plasticity", "P_retention"),
            ("R_plasticity", "R_retention"),
        )
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_pad = max((max(xs) - min(xs)) * 0.08, 0.005)
    y_pad = max((max(ys) - min(ys)) * 0.08, 0.005)
    x0, x1 = min(xs) - x_pad, max(xs) + x_pad
    y0, y1 = min(ys) - y_pad, max(ys) + y_pad

    def px(value: float) -> float:
        return left + (value - x0) / (x1 - x0) * (width - left - right)

    def py(value: float) -> float:
        return height - bottom - (value - y0) / (y1 - y0) * (
            height - top - bottom
        )

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#111"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height-bottom}" stroke="#111"/>'
        ),
        (
            f'<text x="{width/2}" y="{height-22}" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">'
            "Validation plasticity</text>"
        ),
        (
            f'<text x="22" y="{height/2}" text-anchor="middle" '
            f'transform="rotate(-90 22 {height/2})" '
            'font-family="sans-serif" font-size="16">'
            "Validation retention</text>"
        ),
        (
            '<text x="450" y="25" text-anchor="middle" '
            'font-family="sans-serif" font-size="18" font-weight="bold">'
            "P1-A7 validation-scale endpoint flexibility</text>"
        ),
    ]
    for tick in range(6):
        xv = x0 + (x1 - x0) * tick / 5
        yv = y0 + (y1 - y0) * tick / 5
        svg.append(
            f'<text x="{px(xv):.1f}" y="{height-bottom+22}" '
            'text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{xv:.3f}</text>'
        )
        svg.append(
            f'<text x="{left-10}" y="{py(yv)+4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{yv:.3f}</text>'
        )
    for index, method in enumerate(TUNED_METHODS):
        item = flexibility[method]
        color = colors[index % len(colors)]
        xp = px(float(item["P_plasticity"]))
        yp = py(float(item["P_retention"]))
        xr = px(float(item["R_plasticity"]))
        yr = py(float(item["R_retention"]))
        svg.extend(
            [
                (
                    f'<line x1="{xp:.1f}" y1="{yp:.1f}" x2="{xr:.1f}" '
                    f'y2="{yr:.1f}" stroke="{color}" stroke-width="2"/>'
                ),
                f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="5" fill="{color}"/>',
                (
                    f'<rect x="{xr-5:.1f}" y="{yr-5:.1f}" width="10" '
                    f'height="10" fill="white" stroke="{color}" '
                    'stroke-width="2"/>'
                ),
                (
                    f'<text x="{xr+8:.1f}" y="{yr-7:.1f}" '
                    f'font-family="sans-serif" font-size="12" '
                    f'fill="{color}">{method}</text>'
                ),
            ]
        )
    svg.append(
        '<text x="500" y="585" font-family="sans-serif" font-size="12">'
        "● P endpoint   □ R endpoint (validation-scale sensitivity)</text>"
    )
    svg.append("</svg>")
    return "\n".join(svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream-a-scalars",
        type=Path,
        default=Path("results/stream_a/stream_scalars.csv"),
    )
    parser.add_argument(
        "--stream-a-tasks",
        type=Path,
        default=Path("results/stream_a/stream_tasks.csv"),
    )
    parser.add_argument(
        "--stream-b-pool",
        type=Path,
        default=Path("results/phase1/stream_b/streamb_pool.csv"),
    )
    parser.add_argument(
        "--tuned-config-path",
        type=Path,
        default=Path("results/phase1/phase1_tuned_configs.json"),
    )
    parser.add_argument(
        "--conditional-retune-request",
        type=Path,
        default=Path("results/phase1/conditional_retune_request.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned_configs = load_tuned_configs(args.tuned_config_path)
    validate_p1_a7_artifact(tuned_configs)
    stream_a_rows = read_csv(args.stream_a_scalars)
    stream_a_tasks = read_csv(args.stream_a_tasks)
    stream_b_pool = read_csv(args.stream_b_pool)

    conditional_a = any(
        str(row["arm"]) == CONDITIONAL_RETUNE_ARM for row in stream_a_tasks
    )
    conditional_b = any(
        str(row["arm"]) == CONDITIONAL_RETUNE_ARM for row in stream_b_pool
    )
    if conditional_a != conditional_b:
        raise RuntimeError(
            "conditional CAP data exists on only one final stream"
        )
    conditional_present = conditional_a and conditional_b
    if conditional_present:
        load_conditional_retune_request(
            args.conditional_retune_request,
            tuned_config_path=args.tuned_config_path,
            tuned_configs=tuned_configs,
        )

    validate_stream_a_append_completion(
        stream_a_tasks,
        tuned_configs,
        include_conditional_retune=conditional_present,
    )
    validate_stream_b_completion(
        stream_b_pool,
        include_conditional_retune=conditional_present,
    )
    grade = grade_phase1(
        stream_a_rows=stream_a_rows,
        stream_b_pool_rows=stream_b_pool,
        tuned_configs=tuned_configs,
    )

    if grade["kill_gate"]["conditional_retune_required"]:
        existing_request = (
            json.loads(args.conditional_retune_request.read_text(encoding="utf-8"))
            if args.conditional_retune_request.exists()
            else None
        )
        request = conditional_retune_request(
            grade=grade,
            tuned_config_path=args.tuned_config_path,
            tuned_configs=tuned_configs,
            existing=existing_request,
        )
        atomic_write_json(args.conditional_retune_request, request)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(grade)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "flexibility_plot.svg").write_text(
        build_flexibility_svg(grade["validation_flexibility"]),
        encoding="utf-8",
    )
    atomic_write_json(
        args.output_dir / "grade.json",
        {
            "schema_version": 3,
            "amendment": FINAL_STREAM_AMENDMENT,
            "created_at_utc": utc_now(),
            "tuned_config_path": str(args.tuned_config_path),
            "tuned_config_sha256": hashlib.sha256(
                args.tuned_config_path.read_bytes()
            ).hexdigest(),
            "conditional_retune_request_path": str(
                args.conditional_retune_request
            ),
            **grade,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
