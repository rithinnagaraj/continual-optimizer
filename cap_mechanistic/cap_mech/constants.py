import json
from pathlib import Path

VERSION = "cap-mechanisms-final-v1"
SEEDS = (42, 123, 451, 789, 2026)
EXPECTED = {42: (.903093, .906064), 123: (.901381, .902385),
            451: (.902134, .911359), 789: (.901280, .909474),
            2026: (.905459, .907551)}
TOLERANCE = 5e-4
CONFIG_DIR = Path(__file__).parent / "configs"

def config(name):
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())

CAP = config("final_cap")
STREAMS = config("streams")
MECHANISMS = config("mechanisms")
