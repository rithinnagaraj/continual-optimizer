"""Run the frozen Post-P1 Stream A/B CAP-v2 ablation.

Mandatory pre-launch gate::

    python -m unittest tests.test_cap_v2_postp1

The runner reuses the authoritative Post-P1 stream engines and injects only the
registered CAP optimizer variant plus detached mechanism diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import experiment_e4b as stream_a
import experiment_stream_b as stream_b
from experiment_e2 import read_csv, write_csv
from post_phase1_common import block_stream_b_schedule
from post_phase1_streams import StreamRuntime
from paper_run_integrity import require_complete_cell
from cap_v2_postp1_common import (
    DEFAULT_OUTPUT_ROOT,
    DIAGNOSTIC_FIELDS,
    MODES,
    POST_P1_POOL_SIZE,
    DiagnosticRecorder,
    add_mode_arguments,
    add_seed_arguments,
    arm_name,
    assert_authoritative_protocol,
    build_run_config,
    capture_aux_rng_state,
    cell_rows,
    checkpoint_identity,
    diagnostic_cell_rows,
    lock_run_config,
    optimizer_factory,
    resolve_seeds,
    resolve_variant,
    restore_aux_rng_state,
    trim_cell_rows,
    trim_diagnostic_rows,
    validate_checkpoint_identity,
)


RUNNER_PATH = Path(__file__).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=("a", "b", "both"), required=True)
    add_mode_arguments(parser)
    add_seed_arguments(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    try:
        args.variant = resolve_variant(args)
        args.resolved_seeds = resolve_seeds(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def stream_directory(
    output_root: Path,
    variant: str,
    stream: str,
) -> Path:
    if variant not in MODES:
        raise ValueError(f"unknown CAP-v2 variant: {variant!r}")
    if stream not in {"a", "b"}:
        raise ValueError(f"unknown stream: {stream!r}")
    return output_root / f"v25_{variant}" / f"stream_{stream}"


def stream_paths(root: Path, stream: str) -> dict[str, Path]:
    if stream == "a":
        return {
            "scalars": root / "stream_scalars.csv",
            "tasks": root / "stream_tasks.csv",
            "diagnostics": root / "cap_v2_diagnostics.csv",
            "config": root / "run_config.json",
            "checkpoints": root / "checkpoints",
        }
    return {
        "scalars": root / "streamb_scalars.csv",
        "tasks": root / "streamb_tasks.csv",
        "pool": root / "streamb_pool.csv",
        "diagnostics": root / "cap_v2_diagnostics.csv",
        "config": root / "run_config.json",
        "checkpoints": root / "checkpoints",
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def _latest_checkpoint(
    checkpoint_dir: Path, arm: str, seed: int, stream: str
) -> Path | None:
    finder = stream_a.latest_checkpoint if stream == "a" else stream_b.latest_checkpoint
    return finder(checkpoint_dir, arm, seed)


def _has_partial_cell(
    *,
    stream: str,
    arm: str,
    variant: str,
    seed: int,
    scalar_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
) -> bool:
    return bool(
        cell_rows(scalar_rows, arm, seed)
        or cell_rows(task_rows, arm, seed)
        or cell_rows(pool_rows, arm, seed)
        or diagnostic_cell_rows(diagnostic_rows, variant, stream, seed)
        or _latest_checkpoint(checkpoint_dir, arm, seed, stream)
    )


def _cell_extra_payload(
    *,
    stream: str,
    variant: str,
    seed: int,
    arm: str,
    identity: Mapping[str, Any],
    recorder: DiagnosticRecorder,
    scalar_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    step: int,
) -> dict[str, Any]:
    task = ((step - 1) // stream_a.STEPS_PER_TASK) + 1
    local_step = ((step - 1) % stream_a.STEPS_PER_TASK) + 1
    counts = {
        "scalars": len(cell_rows(scalar_rows, arm, seed)),
        "tasks": len(cell_rows(task_rows, arm, seed)),
    }
    counts["diagnostics"] = len(
        diagnostic_cell_rows(diagnostic_rows, variant, stream, seed)
    )
    if stream == "b":
        counts["pool"] = len(cell_rows(pool_rows, arm, seed))
    return {
        "cap_v2_postp1": {
            **dict(identity),
            "global_step": step,
            "task_index": task,
            "local_step": local_step,
            "loader_progress": step,
            "cell_row_counts": counts,
            "diagnostic_state": recorder.state_dict(),
            **capture_aux_rng_state(),
        }
    }


def run_cell(
    *,
    stream: str,
    variant: str,
    seed: int,
    runtime: StreamRuntime,
    paths: Mapping[str, Path],
    resume: bool,
    cap_overrides: Mapping[str, Any] | None = None,
    arm_override: str | None = None,
    experiment_context: Mapping[str, Any] | None = None,
    factory_override=None,
    record_diagnostics: bool = True,
) -> None:
    arm = arm_name(variant) if arm_override is None else arm_override
    scalar_rows = _load_rows(paths["scalars"])
    task_rows = _load_rows(paths["tasks"])
    pool_rows = _load_rows(paths["pool"]) if stream == "b" else []
    diagnostic_rows = _load_rows(paths["diagnostics"])

    partial = _has_partial_cell(
        stream=stream,
        arm=arm,
        variant=variant,
        seed=seed,
        scalar_rows=scalar_rows,
        task_rows=task_rows,
        pool_rows=pool_rows,
        diagnostic_rows=diagnostic_rows,
        checkpoint_dir=paths["checkpoints"],
    )
    checkpoint_file = _latest_checkpoint(paths["checkpoints"], arm, seed, stream)
    if partial and not resume:
        raise RuntimeError(
            f"partial cell exists; rerun with --resume: {variant} {stream} seed={seed}"
        )
    if partial and resume and checkpoint_file is None:
        raise RuntimeError(
            f"partial cell has no resumable checkpoint: {variant} {stream} seed={seed}"
        )

    recorder = DiagnosticRecorder(
        variant=variant,
        stream=stream,
        seed=seed,
        arm=arm,
        rows=diagnostic_rows,
        path=paths["diagnostics"],
    )
    identity_holder: dict[str, Mapping[str, Any]] = {}
    base_factory = optimizer_factory(variant, cap_overrides) if factory_override is None else factory_override

    def factory(model):
        identity_holder["value"] = checkpoint_identity(
            stream=stream,
            variant=variant,
            seed=seed,
            model=model,
            cap_overrides=cap_overrides,
            experiment_context=experiment_context,
        )
        optimizer, hooks = base_factory(model)
        return optimizer, hooks

    def validate(checkpoint: Mapping[str, Any]) -> None:
        expected = identity_holder.get("value")
        if expected is None:
            raise RuntimeError("optimizer factory did not establish checkpoint identity")
        metadata = validate_checkpoint_identity(checkpoint, expected)
        step = int(checkpoint.get("step", -1))
        if int(metadata["global_step"]) != step:
            raise RuntimeError("checkpoint global-step identity mismatch")
        expected_task = ((step - 1) // stream_a.STEPS_PER_TASK) + 1
        expected_local = ((step - 1) % stream_a.STEPS_PER_TASK) + 1
        if (
            int(metadata["task_index"]) != expected_task
            or int(metadata["local_step"]) != expected_local
            or int(metadata["loader_progress"]) != step
        ):
            raise RuntimeError("checkpoint task/local-step identity mismatch")
        engine = stream_a if stream == "a" else stream_b
        if step == engine.TOTAL_STEPS:
            require_complete_cell(
                scalar_rows, task_rows, pool_rows, arm=arm, seed=seed,
                task_count=engine.TASK_COUNT, total_steps=engine.TOTAL_STEPS,
                scalar_every=engine.SCALAR_LOG_EVERY,
                pool_size=POST_P1_POOL_SIZE if stream == "b" else 0,
                committed_counts=metadata["cell_row_counts"],
            )
            if len(diagnostic_cell_rows(diagnostic_rows, variant, stream, seed)) != metadata["cell_row_counts"]["diagnostics"]:
                raise RuntimeError("Terminal checkpoint has incomplete diagnostic output")

    def loaded(checkpoint: Mapping[str, Any]) -> None:
        metadata = checkpoint["cap_v2_postp1"]
        restore_aux_rng_state(checkpoint)
        recorder.load_state_dict(metadata["diagnostic_state"])

    def trim_a(
        checkpoint: Mapping[str, Any],
        loaded_scalars: list[dict],
        loaded_tasks: list[dict],
    ) -> None:
        metadata = validate_checkpoint_identity(
            checkpoint, identity_holder["value"]
        )
        counts = metadata["cell_row_counts"]
        trim_cell_rows(loaded_scalars, arm, seed, int(counts["scalars"]))
        trim_cell_rows(loaded_tasks, arm, seed, int(counts["tasks"]))
        trim_diagnostic_rows(
            diagnostic_rows,
            variant,
            stream,
            seed,
            int(counts["diagnostics"]),
        )
        write_csv(paths["diagnostics"], diagnostic_rows, DIAGNOSTIC_FIELDS)

    def trim_b(
        checkpoint: Mapping[str, Any],
        loaded_scalars: list[dict],
        loaded_tasks: list[dict],
        loaded_pool: list[dict],
    ) -> None:
        trim_a(checkpoint, loaded_scalars, loaded_tasks)
        counts = checkpoint["cap_v2_postp1"]["cell_row_counts"]
        trim_cell_rows(loaded_pool, arm, seed, int(counts["pool"]))

    def step_callback(model, optimizer, step: int, task: int, local_step: int) -> None:
        if record_diagnostics:
            recorder.record(model, optimizer, step, task, local_step)

    def checkpoint_extra(model, optimizer, step: int) -> Mapping[str, Any]:
        del model, optimizer
        return _cell_extra_payload(
            stream=stream,
            variant=variant,
            seed=seed,
            arm=arm,
            identity=identity_holder["value"],
            recorder=recorder,
            scalar_rows=scalar_rows,
            task_rows=task_rows,
            pool_rows=pool_rows,
            diagnostic_rows=diagnostic_rows,
            step=step,
        )

    common = {
        "arm": arm,
        "seed": seed,
        "scalar_rows": scalar_rows,
        "task_rows": task_rows,
        "scalar_path": paths["scalars"],
        "task_path": paths["tasks"],
        "checkpoint_dir": paths["checkpoints"],
        "device": runtime.device,
        "num_workers": runtime.num_workers,
        "resume": bool(partial and resume),
        "optimizer_factory": factory,
        "checkpoint_validator": validate,
        "checkpoint_loaded_callback": loaded,
        "step_callback": step_callback,
        "checkpoint_extra_factory": checkpoint_extra,
    }
    if stream == "a":
        stream_a.run_cell(
            **common,
            mnist_train=runtime.mnist_train,
            test_loader=runtime.test_loader_a,
            permutations_cpu=runtime.permutations_a,
            tuned_configs={},
            checkpoint_trim_callback=trim_a,
        )
    else:
        stream_b.run_cell(
            **common,
            tuned_configs={},
            mnist_train=runtime.mnist_train,
            test_loader=runtime.test_loader_b,
            permutations_cpu=runtime.permutations_b,
            schedule=runtime.schedule_b,
            pool_rows=pool_rows,
            pool_path=paths["pool"],
            pool_size=POST_P1_POOL_SIZE,
            checkpoint_trim_callback=trim_b,
        )


def run(args: argparse.Namespace) -> None:
    assert_authoritative_protocol()
    if block_stream_b_schedule(POST_P1_POOL_SIZE) != block_stream_b_schedule(40):
        raise RuntimeError("authoritative Stream-B pool-40 schedule changed")
    device = torch.device(args.device)
    selected_streams = ("a", "b") if args.stream == "both" else (args.stream,)
    runtime = StreamRuntime(
        args.data_dir, device, args.num_workers, POST_P1_POOL_SIZE
    )
    for stream in selected_streams:
        root = stream_directory(args.output_dir, args.variant, stream)
        root.mkdir(parents=True, exist_ok=True)
        paths = stream_paths(root, stream)
        proposed = build_run_config(
            stream=stream,
            variant=args.variant,
            device=device,
            runner_path=RUNNER_PATH,
        )
        lock_run_config(
            paths["config"],
            proposed,
            requested_seeds=args.resolved_seeds,
            resume=args.resume,
            device=device,
        )
        for seed in args.resolved_seeds:
            run_cell(
                stream=stream,
                variant=args.variant,
                seed=seed,
                runtime=runtime,
                paths=paths,
                resume=args.resume,
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
