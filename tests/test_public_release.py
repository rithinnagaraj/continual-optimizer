"""Exercise portability and release hygiene without importing scientific dependencies."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from run_experiments import ROOT, STAGES, plan
from tools.audit_release import audit, manifest_paths


class PublicReleaseTests(unittest.TestCase):
    def test_all_stages_have_local_entrypoints(self):
        for stage in STAGES:
            for cwd, command in plan(stage):
                self.assertTrue(cwd.resolve().is_relative_to(ROOT))
                if command[0] == "-m":
                    self.assertTrue((cwd / (command[1].replace(".", "/") + ".py")).is_file())
                else:
                    self.assertTrue((cwd / command[0]).is_file())

    def test_mechanistic_smoke_keeps_m5_all_seeds(self):
        jobs = plan("mechanistic-mnist", smoke=True)
        self.assertEqual(len(jobs), 8)
        for _, command in jobs:
            self.assertIn("--smoke", command)
            self.assertNotIn("--seeds", command)
            self.assertEqual("--allow-unverified-reference" in command, command[2] != "reproduce")

    def test_paper_mnist_routes_through_bounded_validation(self):
        jobs=plan("mechanistic-mnist-paper",resume=True)
        self.assertEqual([command[0] for _,command in jobs],
                         ["tools/reference_diagnostics.py", "cap_mechanistic/diagnostics/run_validated_mechanisms.py"])
        self.assertTrue(all("--resume" in command for _,command in jobs))

    def test_launcher_flags_are_declared_by_each_entrypoint(self):
        for stage in STAGES:
            for cwd, command in plan(stage, resume=True):
                if command[0] == "-m":
                    source = cwd / (command[1].replace(".", "/") + ".py")
                    arguments = command[2:]
                else:
                    source = cwd / command[0]
                    arguments = command[1:]
                tree = ast.parse(source.read_text(encoding="utf-8-sig"))
                declared = {
                    argument.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                }
                for argument in arguments:
                    if argument.startswith("--"):
                        self.assertIn(argument, declared, f"{stage}: {source.name}: {argument}")

    def test_distribution_audit(self):
        result = audit()
        self.assertTrue(result["passed"], result["errors"])

    def test_copied_release_plans_without_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "source-copy"
            copy.mkdir()
            for relative in [*manifest_paths(), "MANIFEST.sha256"]:
                destination = copy / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for command in (["run_experiments.py", "--list"],
                            ["run_experiments.py", *STAGES, "--dry-run"],
                            ["tools/audit_release.py", "--strict"],
                            ["tools/package_release.py"]):
                completed = subprocess.run([sys.executable, *command], cwd=copy, env=env,
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with zipfile.ZipFile(copy / "dist/anonymous-source.zip") as archive:
                expected = {"anonymous-source/" + relative for relative in [*manifest_paths(), "MANIFEST.sha256"]}
                self.assertEqual(set(archive.namelist()), expected)
                self.assertTrue(all(entry.date_time == (2000, 1, 1, 0, 0, 0) for entry in archive.infolist()))


if __name__ == "__main__":
    unittest.main()
