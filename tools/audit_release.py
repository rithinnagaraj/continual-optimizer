"""Verify the source allowlist, syntax, local dependencies and metadata."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = {"torch", "torchvision", "numpy", "pandas", "matplotlib", "pytest", "PIL", "transformers"}
OMITTED_VENDOR_IMPORTS = ("core",)
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "personal-home": re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\s\"']+|/(?:home|Users)/[A-Za-z0-9_.-]+)", re.I),
    "absolute-drive": re.compile(r"(?<![A-Za-z])[A-Za-z]:\\[A-Za-z]"),
    "credential": re.compile(r"\b(?:ghp_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})\b"),
    "identity-collector": re.compile(r"(?:socket\.gethostname|(?:getpass|os)\.getuser|os\.getlogin)\s*\("),
}
GENERATED_PARTS = {"__pycache__", ".pytest_cache", ".venv", "data", "results", "dist", "validation_output"}


def manifest_paths(root=ROOT):
    entries = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = Path(relative)
        if separator != "  " or not re.fullmatch(r"[a-f0-9]{64}", digest) or path.is_absolute() or ".." in path.parts or relative in entries:
            raise ValueError("Invalid distribution manifest entry")
        entries[relative] = digest
    return entries


def audit(root=ROOT, *, forbidden=(), strict=False):
    errors = []
    entries = manifest_paths(root)
    modules = {p.stem for p in root.glob("*.py")}
    modules.update({"cap_mech", "cifar_mech", "compare_reference", "reference_model", "reference_optimizer",
                    "run_validated_mechanisms", "audit_release", "test_isolation", "tests", "tools", "run_experiments"})
    parsed = 0
    for relative, expected in entries.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"Missing or escaping file: {relative}")
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            errors.append(f"Manifest mismatch: {relative}")
        try:
            content = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary file: {relative}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                errors.append(f"Anonymity pattern {label}: {relative}")
        for term in forbidden:
            if term.casefold() in content.casefold() or term.casefold() in relative.casefold():
                errors.append(f"Forbidden identity term: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(content)
            except ValueError as error:
                errors.append(f"Invalid JSON: {relative}: {error}")
        if path.suffix == ".py":
            try:
                tree = ast.parse(content, filename=relative)
                parsed += 1
            except SyntaxError as error:
                errors.append(f"Invalid Python: {relative}:{error.lineno}")
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    top = name.split(".")[0]
                    if top in sys.stdlib_module_names or top in modules or top in EXTERNAL:
                        continue
                    if relative.startswith("external/upgd/") and top in OMITTED_VENDOR_IMPORTS:
                        continue  # Unexecuted, pinned upstream provenance files.
                    errors.append(f"Undeclared import {name}: {relative}")
    # Verify the third-party and alternative-validation pins without importing torch.
    constants = {}
    for node in ast.parse((root / "phase1_arms.py").read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    pins = {
        "external/upgd/core/optim/weight_upgd/first_order.py": constants["UPGD_SOURCE_SHA256"],
        "external/upgd/experiments/input_permuted_mnist.py": constants["UPGD_DESIGNATION_SHA256"],
        "external/upgd/experiments/statistics_input_permuted_mnist.py": constants["UPGD_TASK_CONFIG_SHA256"],
    }
    launcher = root / "cap_mechanistic/diagnostics/run_validated_mechanisms.py"
    diagnostic_constants = {}
    for node in ast.parse(launcher.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"SOURCE", "PINS"}:
                diagnostic_constants[node.targets[0].id] = ast.literal_eval(node.value)
    for name, expected in diagnostic_constants["PINS"].items():
        pins["cap_mechanistic/diagnostics/" + name] = expected
    for relative, expected in pins.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            errors.append(f"Pinned source changed: {relative}")
    core = root / "cap_mechanistic/cap_mech"
    digest = hashlib.sha256()
    for path in sorted([*core.rglob("*.py"), *core.rglob("*.json")]):
        digest.update(path.relative_to(core).as_posix().encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != diagnostic_constants["SOURCE"]:
        errors.append("MNIST core differs from the alternative validation source pin")
    expected_files = set(entries) | {"MANIFEST.sha256"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"Symlink: {relative.as_posix()}")
        if not path.is_file() or relative.as_posix() in expected_files:
            continue
        if strict or not (set(relative.parts) & GENERATED_PARTS or path.suffix in {".pyc", ".log", ".pt", ".pth"}):
            errors.append(f"Unexpected file: {relative.as_posix()}")
    return {"passed": not errors, "files_checked": len(entries), "python_files_parsed": parsed, "errors": sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid", action="append", default=[], help="Private identity string; never saved in the report")
    parser.add_argument("--strict", action="store_true", help="Also reject every unlisted generated file")
    args = parser.parse_args()
    result = audit(forbidden=args.forbid, strict=args.strict)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
