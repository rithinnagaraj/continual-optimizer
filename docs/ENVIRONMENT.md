# Environment and numerical reproducibility

The recorded scientific environment used Python 3.10.20 and PyTorch
2.11.0+cu130. A complete historical package freeze is unavailable.
`requirements.txt` specifies minimum dependencies, rather than an exact tested
environment lock. Install a compatible PyTorch/torchvision pair for the target
CPU or CUDA installation before installing the remaining requirements.

The benchmark and development environment was NVIDIA L4; MNIST mechanisms and
the CIFAR learning-rate pilot used NVIDIA A30. The full CIFAR mechanism GPU
model was not separately recorded. End-to-end wall times were not recorded.
The README gives update budgets where available.

## Checking an environment

From the repository root in a working scientific environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiments.py mechanistic-mnist --smoke --device cpu
python run_experiments.py mechanistic-cifar --smoke --device cpu
cd cap_mechanistic
python -m pytest -q tests diagnostics/test_validation_launcher.py
cd ../cap_mechanistic_cifar
python -m pytest -q tests test_lr_selection.py
cd ..
python -c "from pathlib import Path; Path('results/validation').mkdir(parents=True, exist_ok=True)"
python -m pip freeze > results/validation/requirements.lock.txt
```

Keep the environment freeze with its test outputs. Synthetic smoke checks
exercise the implementation; agreement with published metrics requires complete
scientific runs and the corresponding reference checks. Compare the fixed
benchmark and development metrics using the README's table and figure commands.

## Execution status

The current source has received static source/configuration and archive checks.
Numerical optimizer tests, synthetic smoke pipelines, and full experiment
campaigns have not been rerun for this source. The recorded local PyTorch load
attempt failed with Windows Application Control error 4551; the available
standard-library Python environment had no PyTorch installed. These checks
therefore supply no current numerical pass result.

Exact continuation requires consistent software, device, and thread settings.
Cross-version or cross-device numerical equivalence is not assumed. Historical
endpoint discrepancies remain visible in reference reports. Bounded implementation
validation preserves those gate outcomes and does not establish full-trajectory
equivalence. See [validation](VALIDATION.md) and
[mechanistic protocols](MECHANISTIC.md) for their distinct scopes.
