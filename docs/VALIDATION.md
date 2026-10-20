# Validation

The repository separates source integrity, implementation checks, and scientific
reference comparisons. A passing source audit or synthetic smoke run does not
establish reproduction of published numerical results.

## Source checks

```bash
python tools/audit_release.py
python -m unittest tests.test_public_release -v
```

The auditor parses distributed Python and JSON, verifies manifest checksums and
source pins, checks declared imports, scans identifying metadata, and reports
unexpected files. Distribution tests check launcher entry points and flags,
command planning from a manifest-only copy, removal of inherited `PYTHONPATH`,
and archive membership with fixed timestamps. See
[SOURCE_DISTRIBUTION.md](SOURCE_DISTRIBUTION.md) for packaging details.

## Numerical and protocol tests

From the repository root:

```bash
python -m pytest -q
python run_experiments.py mechanistic-mnist --smoke --device cpu
python run_experiments.py mechanistic-cifar --smoke --device cpu
cd cap_mechanistic
python -m pytest -q tests diagnostics/test_validation_launcher.py
cd ../cap_mechanistic_cifar
python -m pytest -q tests test_lr_selection.py
cd ..
```

The root tests cover optimizer equations and state, historical variants,
reporting, configuration commits, and checkpoint recovery. Each mechanistic
package has separate tests for its optimizer, interventions, sampler, metrics,
and continuation. Run the packages separately to avoid test-module name clashes.
Smoke runs use synthetic data and reduced horizons; their reports remain
unverified. The MNIST smoke pipeline retains all five seeds for M5's nested
validation. CIFAR smoke runs both architectures with seed 42.

## Scientific reference comparisons

The strict MNIST route compares each seed and metric against historical targets
with absolute tolerance `5e-4`. Failed comparisons remain failures. The bounded
route instead checks exact same-environment agreement over 1,000 real-data
updates per seed/stream and 16 continuation updates from each of 45 checkpoints.
It preserves the historical gate and conservative report labels; it does not
prove full-trajectory or intervention equivalence.

CIFAR admission checks complete reference banks, finite metrics, frozen protocol
and source identities, and checkpoint hashes. It does not compare CIFAR accuracy
to MNIST targets or impose an accuracy-quality threshold. See
[MECHANISTIC.md](MECHANISTIC.md) for both admission policies.

## Execution record

Current source/configuration, source-pin, manifest, metadata, and archive-payload
checks are static. Numerical suites, synthetic smoke runs, and full experiment
campaigns have not been rerun for this source. The recorded runtime limitation
and environment setup are described in [ENVIRONMENT.md](ENVIRONMENT.md).

[VALIDATION_SUMMARY.json](../VALIDATION_SUMMARY.json) retains an earlier test
record: 41 passing source/reporting tests and five PyTorch-dependent skips.
Those results predate the current implementation and do not validate it. The
summary separates that historical record from current execution status.
