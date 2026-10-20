import ast
import importlib.abc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

BANNED = ("cap_adam", "model", "experiment_e", "phase1_", "experiment_stream_b", "patched_e1", "extra_e1", "post_phase1", "cap_release_final", "cap_v2_postp1")

def banned(name):
    root = name.split(".")[0]
    return any(root == p or root.startswith(p+"_") or (p.endswith("_") or p == "experiment_e") and root.startswith(p) for p in BANNED)

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if banned(fullname):
            raise ImportError(f"Forbidden legacy import: {fullname}")

def test_static_no_legacy_imports():
    package = Path(__file__).resolve().parents[1]
    for path in (package/"cap_mech").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not any(banned(n.name) for n in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not banned(node.module or ""), path

@pytest.mark.skipif(os.environ.get("CAP_MECH_ISOLATED") == "1", reason="Outer isolation test already copied the package")
def test_copied_package_runs_tests_without_old_root(tmp_path):
    source = Path(__file__).resolve().parents[1]
    copy = tmp_path/"standalone"
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "results", "validation", "*.egg-info"))
    env = os.environ.copy()
    # Only the copied package and installed dependencies remain on the import path.
    env.pop("PYTHONPATH", None)
    env["CAP_MECH_ISOLATED"] = "1"
    script = """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path.cwd()/'tests'))
from test_isolation import BlockLegacy
sys.meta_path.insert(0, BlockLegacy())
import pytest
raise SystemExit(pytest.main(['-q', 'tests', '-k', 'not copied_package']))
"""
    # Dependencies injected by the validation host are copied as dependencies,
    # never retained via an old project PYTHONPATH.
    for name in ("pandas", "dateutil", "pytz", "tzdata", "six"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        location = Path(module.__file__).resolve()
        if source.parent in location.parents:
            target = copy/(location.parent.name if location.name == "__init__.py" else location.name)
            if location.name == "__init__.py":
                shutil.copytree(location.parent, target)
            else:
                shutil.copy2(location, target)
            # Windows binary wheels ship adjacent DLL directories.
            for libs in location.parent.parent.glob(name+".libs"):
                shutil.copytree(libs, copy/libs.name, dirs_exist_ok=True)
    result = subprocess.run([sys.executable, "-c", script], cwd=copy, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0, result.stdout
