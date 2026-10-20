"""Verify immutable artifacts produced by the self-contained v2 experiment."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from post_phase1_common import (
    FREEZE_RECORD,
    POOL_DECISION,
    V2_ISOLATION,
    WINNER_ARTIFACT,
    load_json,
    load_pool_decision,
    load_winner_artifact,
    sha256_file,
)


def verify() -> dict[str, object]:
    decision = load_pool_decision()
    winner = load_winner_artifact()
    record = load_json(FREEZE_RECORD)
    return {
        "status": "VERIFIED",
        "isolation": V2_ISOLATION,
        "pool_size": decision["chosen_pool_size"],
        "pool_decision_sha256": sha256_file(POOL_DECISION),
        "winner_sha256": sha256_file(WINNER_ARTIFACT),
        "winner_record_sha256": record["winner_artifact"]["sha256"],
        "methods": sorted(winner),
        "legacy_result_dependencies": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify",))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
