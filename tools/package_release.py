"""Create an anonymous source archive using only the checked distribution manifest."""
from pathlib import Path
import argparse
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_release import ROOT, audit, manifest_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "anonymous-source.zip")
    parser.add_argument("--forbid", action="append", default=[])
    args = parser.parse_args()
    result = audit(forbidden=args.forbid)
    if not result["passed"]:
        for error in result["errors"]:
            print(error, file=sys.stderr)
        return 1
    files = sorted([*manifest_paths(), "MANIFEST.sha256"])
    target = args.output.resolve()
    if target == ROOT / "MANIFEST.sha256" or target in {(ROOT / name).resolve() for name in files}:
        parser.error("Output must not overwrite a source file")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            info = zipfile.ZipInfo("anonymous-source/" + relative, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    print(f"Created {target.name}: {len(files)} source files; fixed timestamps; no host metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
