import json
from pathlib import Path

VERSION = "cifar-mechanisms-final-v1"
SEEDS = (42, 123, 451, 789, 2026)
CONFIG_DIR = Path(__file__).parent / "configs"

def config(name):
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())

CAP = config("final_cap")
STREAMS = config("streams")
MECHANISMS = config("mechanisms")
