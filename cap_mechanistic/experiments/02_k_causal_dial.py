"""Convenience wrapper for the standalone CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cap_mech.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["m2-k-dial", *sys.argv[1:]]))
