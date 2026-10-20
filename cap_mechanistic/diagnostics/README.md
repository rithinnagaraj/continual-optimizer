# Bounded numerical validation

These diagnostics compare the mechanistic implementation with bundled optimizer
and model snapshots. SHA-256 pins identify the exact reference sources; no parent
project imports are needed.

`compare_reference.py` compares the final finite-T_m optimizer with the historical
snapshot on identical gradients and integrated synthetic inputs. From the repo
root, after reference outputs exist:

```bash
python cap_mechanistic/diagnostics/compare_reference.py --device cuda --steps 128 --output-dir results/mechanistic_mnist
```

`run_validated_mechanisms.py` implements the separate bounded same-environment
policy. It requires a complete, non-smoke five-seed reference bank, all checkpoint
hashes, matching source/configuration and the existing MNIST raw files:

```bash
python cap_mechanistic/diagnostics/run_validated_mechanisms.py --device cuda --data-dir data --output-dir results/mechanistic_mnist --validate-only
python cap_mechanistic/diagnostics/run_validated_mechanisms.py --device cuda --data-dir data --output-dir results/mechanistic_mnist --resume
```

The checks compare 1,000 real-data updates per seed/stream and 16 updates from
each of 45 checkpoints. A mismatch stops execution before mechanisms. A successful
certificate is reused only when provenance still matches. The second command
runs M1–M6 and analysis after validation, using the explicit unverified
reference option. Historical endpoint failures remain recorded, and automated
claim labels remain conservative. This certifies bounded agreement, not full-run
equivalence or agreement of every intervention. M6's infinite-T_m intervention
holds bound importance unchanged, as specified in its packaged protocol.

The reference can be trained from scratch. Its historical endpoint targets are
defined in `cap_mech/constants.py`; existing result logs are not required.
