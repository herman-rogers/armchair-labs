# Local NVIDIA tree research

The optional tree adapter is
[`experiments/future_player_lab/gpu_backends.py`](../../experiments/future_player_lab/gpu_backends.py).
It provides NVIDIA RAPIDS cuML random forests, XGBoost CUDA histogram boosting,
CatBoost GPU boosting, and explicit CPU controls. The application and CPU tests do
not import CUDA or require an NVIDIA installation.

## Environment

The tested machine has an RTX 4090 (compute capability 8.9, 24 GB), NVIDIA driver
610.57.04, and a driver supporting CUDA 13.3. Driver support alone does not imply
that a CUDA toolkit is installed. The optional dependency group installs the
matching CUDA 13.3 runtime/compiler libraries into the virtual environment.

From the repository root:

```sh
uv sync --python 3.12 --extra shared-data --extra research --extra research-gpu
uv run --no-sync python -m experiments.future_player_lab.gpu_backends \
  --repeats 3 --output data/research/gpu_backend_smoke.json
uv run --no-sync python -m experiments.future_player_lab.gpu_backends \
  --families xgboost catboost random_forest --device cpu --repeats 3 \
  --output data/research/cpu_backend_smoke.json
```

Use `--no-sync` for subsequent commands, or retain all desired `--extra` flags;
an ordinary `uv sync` intentionally removes optional packages. Use only
`--extra research` on a CPU machine. The RAPIDS extra is platform-marked for
Linux x86-64 and Python below 3.15; unsupported machines cannot execute its GPU
backend even if a package manager skips those dependencies.

The lock records XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0, and cuML 26.8.0.
RAPIDS 26.8 requires Numba below 0.65, whose compatible NumPy range is below 2.5.
The base NumPy minimum was therefore relaxed from 2.5.2 to 2.2; the resolver selects
a compatible version when the GPU extra is enabled. Dependencies remain resolved
together, without bypassing dependency checks.

NVIDIA's [installation requirements](https://docs.rapids.ai/install/) require
compute capability 7.0 or newer and driver 580.65.06 or newer for CUDA 13. The
RAPIDS wheels must match the toolkit major version and need NVRTC. NVIDIA's
[cuML installation page](https://developer.nvidia.com/topics/ai/data-science/cuda-x-data-science-libraries/cuml)
describes the official pip packages. Current
[XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/)
uses `tree_method="hist"` with `device="cuda"`; the tested release is pinned
in the lock. [CatBoost GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu)
uses `task_type="GPU"` and `devices` and documents GPU nondeterminism.

## Estimator contract

```python
from experiments.future_player_lab.gpu_backends import build_estimator

model, execution = build_estimator(
    "cuml_rf",
    {"n_estimators": 400, "max_depth": 12, "n_bins": 128},
    seed=42,
    device="cuda:0",
    threads=4,
)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
# Persist execution AFTER fit: effective_device and device_verified are updated.
```

| Family | Device | Notes |
| --- | --- | --- |
| `cuml_rf` | CUDA | NVIDIA cuML histogram random forest; finite float32 inputs; sample weights supported in pinned 26.8 release |
| `xgboost` | CPU or CUDA | Histogram booster; actual fitted device inspected to detect CPU fallback |
| `catboost` | CPU or CUDA | Native GPU setting; GPU training is not bitwise deterministic |
| `random_forest` | CPU | scikit-learn random forest; distinct algorithm from cuML |
| `extra_trees` | CPU | scikit-learn extra trees |
| `hist` | CPU | scikit-learn histogram gradient booster |
| `lightgbm` | CPU | Portable prebuilt LightGBM; CUDA builds are outside this environment |

`build_estimator` returns a serializable wrapper and a mutable execution record.
Arrays are converted to float32; predictions return NumPy arrays. The native model
is available through `model.estimator`, and native properties/methods such as
`feature_importances_` or `save_model` are forwarded when implemented. Save fitted
preprocessing with the wrapper to reproduce the feature ordering and transforms.

Explicit GPU requests fail if the driver, device, library, or GPU implementation
is unavailable. They also fail for CPU-only families. There is no automatic
fallback or silent substitution of sklearn for cuML. XGBoost's native fallback is
checked after fitting. The record includes requested/configured/effective device,
parameters, seed, thread limit, library/NumPy/Python versions, driver and GPU
identity, and fit duration. Effective device remains unset until fitting succeeds.
Threads are bounded for both native estimator settings and BLAS/OpenMP pools.

Parameters use native names. CatBoost additionally accepts `n_estimators`,
`max_depth`, and `max_bin` as aliases for `iterations`, `depth`, and `border_count`.
Device/seed/thread controls are supplied through explicit arguments to prevent
disagreement with the execution record. cuML defaults to one CUDA stream for seeded
repeatability; `n_streams` can explicitly request a different number and is recorded
with the native parameters. CatBoost GPU results have floating-point nondeterminism.

Run one GPU fit at a time initially. Fit imputers, scaling, variance filters,
unsupervised representations, and feature selectors using training rows only;
cuML explicitly rejects missing/infinite values instead of learning an imputation
from the evaluation rows. GPU speedups must be measured on the actual RB matrices:
the synthetic smoke command establishes working execution, not research quality
or representative throughput.

Portable contract tests:

```sh
uv run --no-sync python -m pytest experiments/future_player_lab/tests/test_gpu_backends.py
```

After restoring the canonical raw release and building the verified raw-input
cache, benchmark the actual RB season matrix:

```sh
uv run --no-sync python -m experiments.future_player_lab.gpu_lab.benchmark \
  --output experiments/future_player_lab/runs/gpu_benchmark_001 \
  --target rushing_yards --year 2025 --repeats 3
```

The benchmark compares identical XGBoost settings at 1/2/4/8 CPU threads and on
CUDA, plus CatBoost and cuML GPU anchor recipes. Every configuration starts a fresh
Python process and performs three fits, separating its first fit from warm fits.
The report records preprocessing cost, fit/predict/construction time, RMSE/MAE,
prediction differences, process memory, GPU memory readings, and joblib model
roundtrip checks. Other GPU workloads remain untouched and can affect timings;
GPU memory readings include them. This is one representative target, not the
complete seven-stat experiment. Inputs and feature names are preserved alongside
the report with a checksum.

On the tested 4090, the initial 2025 rushing-yard benchmark used 3,673 training
rows, 170 evaluation rows and 5,499 transformed inputs. The median of two warm
160-tree XGBoost fits was 1.41 seconds on CUDA versus 2.05 seconds with four CPU
threads. cuML random forest took 27.18 seconds with one stream and 11.75 seconds
with four streams; all three four-stream predictions matched the one-stream
predictions exactly in this fixture. Broader campaigns can explicitly choose
`n_streams=4` while retaining the seed and settings in their artifacts. This
observed repeatability is not a promise of identical results across platforms.
An unrelated GPU workload was active during these timings.

The initial run's `resource.getrusage` memory readings include the parent process's
pre-exec history on Linux. Treat those values as lifetime bounds, not isolated
model memory requirements. The benchmark now records `/proc/self/status` `VmHWM`
and current RSS separately, which reset with the child process's executable.
