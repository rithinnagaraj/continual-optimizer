# Source distribution

`MANIFEST.sha256` lists the source files included in the distribution and their
SHA-256 checksums. Project code, configuration, and local dependencies are
bundled together. Datasets, trained checkpoints, historical logs, environments,
and Git metadata are excluded. No parent checkout or existing result tree is
needed to start a run.

## Integrity checks

```bash
python tools/audit_release.py
python tools/audit_release.py --strict
python -m unittest tests.test_public_release -v
```

The auditor checks manifest hashes, Python and JSON syntax, declared top-level
imports, source pins, symlinks, escaping paths, and unexpected files. Normal mode
allows generated caches, data, and results. Strict mode rejects every unlisted
file; use it on a clean source copy.

The manifest is an explicit allowlist. After an intentional source edit, check
the change and regenerate its checksum before packaging. Generated artifacts
should not be added. Distribution tests exercise a manifest-only copy in a
temporary directory with inherited `PYTHONPATH` removed.

## Building an archive

```bash
python tools/package_release.py
```

This verifies the manifest and writes `dist/anonymous-source.zip`. The archive
contains the listed files and the manifest under `anonymous-source/`, with fixed
timestamps and normalized permissions. Generated output, caches, environments,
logs, and `.git` remain excluded even when present in the working directory.
The command does not publish or upload files.

## Metadata and privacy

The shared final-run configuration writer records the literal `anonymous`
instead of collecting a hostname. Source checks scan for personal home paths,
absolute drive paths, email addresses, common credential formats, and identity
collection calls. Add `--forbid PRIVATE_STRING` repeatedly to check additional
strings; values are not saved or echoed in reports. These pattern checks cannot
determine whether a research topic or algorithm is recognizable.

New run logs can contain paths, software versions, timestamps, and device
descriptions. They are excluded from the source archive; inspect their content
and metadata separately before sharing results. Required third-party attribution
is preserved; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
