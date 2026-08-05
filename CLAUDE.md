# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SIDDA (SInkhorn Dynamic Domain Adaptation) is the reference implementation for the paper *SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks* (arXiv:2501.14048). It trains an image classifier with an optional unsupervised domain-adaptation loss based on Sinkhorn divergences, and dynamically tunes the DA regularization/loss-weighting during training instead of requiring manual hyperparameter search.

This repo is a **JAX / Flax NNX / Optax / OTT-JAX rewrite** of the paper's original PyTorch implementation, covering the plain **CNN pipeline only** — the paper's escnn-based equivariant (ENN/D4) model was intentionally dropped in the rewrite, not ported.

There is no build system, package manifest, linter config, or test suite in this repo — it is a small collection of standalone training/eval scripts run directly with `python`.

## Environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Requires Python 3.10+ (developed/tested on 3.12). All scripts assume they are run from `src/scripts/` (they import sibling modules like `from dataset import dataset_dict` and `from models import model_dict` with no package prefix). JAX runs CPU-only here — no CUDA GPU, and `jax-metal` (Apple's experimental Metal backend) pins older, conflicting `jax`/`jaxlib` versions, so it is deliberately not installed.

## Downloading data

```bash
cd src/scripts
python download_data.py --all   # or --dataset <shapes|astro_objects|mnist_m|gz_evo|mrssc2>
```

Fetches and extracts the 5 datasets from Zenodo record 15215272 into `src/scripts/data/<name>/` (gitignored, ~7.2GB total). Each dataset's internal layout differs — inspect before writing new glue code rather than assuming a common structure:
- `shapes`, `astro_objects`, `mnist_m`: `<dir>/{train,test}/{x,y}_{train,test}[_<shift>].npy`, source domain unsuffixed, target domain suffixed (`_noise`, or `_PSF` for an alternate mnist_m shift not used by default).
- `gz_evo`: same `train`/`test` split structure, but source/target are suffixed `_sdss`/`_desi` (real SDSS-vs-DESI telescope shift) instead of clean-vs-noise. `x_train_*.npy` are ~7.7GB each as `float64` (32,000 images/domain) — `dataset.py` downcasts to `float32` on load specifically to keep this dataset's memory footprint manageable on a 16GB machine.
- `mrssc2`: flat layout, no `train`/`test` subdirectories — `source_X_train.npy`, `source_y_train.npy`, `target_X_train.npy`, `target_y_train.npy`, `source_X_test.npy`, `target_X_test.npy`, etc. Images are natively 256×256 (not 100×100 like the other RGB datasets); `augment.py` resizes them down to the model's expected 100×100 input.

## Running training and evaluation

Training and testing are both driven by YAML config files (see `src/scripts/example_yaml_train_CE.yaml` and `example_yaml_train_SIDDA.yaml` for the schema).

```bash
cd src/scripts

# Cross-entropy-only training (no domain adaptation)
python train_CE.py --config example_yaml_train_CE.yaml

# SIDDA training (source classification + Sinkhorn domain adaptation)
python train_SIDDA.py --config example_yaml_train_SIDDA.yaml

# Evaluate all checkpoints in a run directory
python test.py \
  --model_path "/path/to/directory/containing/model" \
  --x_test_path "/path/to/test/images" \
  --y_test_path "/path/to/test/labels" \
  --output_name "name for metrics files" \
  --model_name "cnn" \
  --dataset "shapes|astro_objects|mnist_m|gz_evo|mrssc2"

# Same as above, plus calibration (ECE, Brier score, calibrated confusion matrices)
python test_calibration.py --model_path ... --x_test_path ... --y_test_path ... --output_name ... --model_name ... --dataset ...
```

Key config fields (see the example YAMLs): `model` (`cnn` — the only supported value post-rewrite), `dataset` (key into `dataset_dict`/`model_dict`), `train_data.{input_path,output_path}` (source domain) and, for SIDDA only, `train_data.{target_input_path,target_output_path}` (target domain; labels unused during training), `parameters.warmup` (epochs of CE-only training before the DA loss kicks in), `parameters.{lr,weight_decay,val_size,batch_size,epochs,early_stopping,report_interval,lr_decay,milestones}`, `seed`, and `save_dir`.

Training output directory naming: `<save_dir><model>_DA_<timestr>` for SIDDA runs, `<save_dir><model>_<timestr>` for CE-only runs. Each run directory gets a `config.yaml` (input config plus final metrics), a `losses/` subdirectory (per-epoch and per-step `.npy` arrays, loss/JS-distance/blur/eta plots), and checkpoint **directories** (Orbax format, not flat files) named `best_model_val_acc/`, and for SIDDA also `best_model_total_val_loss/`, `best_model_classification_loss/`, `best_model_DA_loss/`, plus `final_model/`.

There is no automated test suite — verify changes by running the training/eval scripts end-to-end on a small dataset (`shapes` is fastest — 33.6MB, ~15-25s/epoch on CPU), or by loading `src/notebooks/*.ipynb`.

## Architecture

- **`src/scripts/dataset.py`** — `NpyImageDataset`: loads a `.npy` image array (and optionally a `.npy` label array), applies a transform, returns `(img, label)` — or just `img` when `target_domain=True`. `Subset` wraps a parent dataset + an index array + its own independent transform (deliberately does **not** mutate the parent's `.transform` in place — see the transform-aliasing note below). `split_dataset` builds train/val `Subset`s via `np.random.default_rng(seed).permutation`. `NumpyLoader` is a minimal batch iterator (shuffles per-epoch, yields stacked `np.ndarray` batches) replacing `torch.utils.data.DataLoader`. `dataset_dict`/`classes_dict` map dataset-name strings to the class/class-label tuple.
- **`src/scripts/augment.py`** — single source of truth for per-dataset data augmentation, replacing torchvision transforms. Uses `scipy.ndimage` (not PIL) for rotation/translation, since these datasets are pre-generated `float64` arrays with values outside `[0, 255]` in places (e.g. shapes' noise-shifted target domain runs up to ~7.0), not uint8 photographs. `get_transform(dataset_name, train, rng, input_size=None)` returns a per-item callable: training gets random rotation (±180°) + translation (±10%) + h/v flips (p=0.3 each) + resize + normalize(mean=0.5, std=0.5); eval/test gets only resize + normalize.
- **`src/scripts/models.py`** — `CNN(nnx.Module)`: 3 conv blocks (`nnx.Conv`→`nnx.BatchNorm`→`nnx.relu`→`nnx.max_pool`→`nnx.Dropout(0.2)`, kernel sizes 5/3/3, channels 8/16/32), flatten, `nnx.Linear(→256)`→`nnx.LayerNorm`→**latent_space**→`nnx.Linear(→num_classes)`→**logits**. `__call__(self, x, *, train)` returns `(latent_space, logits)` — the 256-d latent vector is what the Sinkhorn DA loss operates on during training and what gets dumped to `latent_vectors/` during testing. Input is **NHWC** (JAX/Flax convention), not NCHW. The flatten size feeding `fc1` is computed dynamically via a real forward pass through the conv/pool stack at construction time (not hand-derived — VALID-padding maxpools on odd intermediate sizes don't divide evenly, e.g. 100→50→25→**12**, not 12.5). Per-dataset factories (`cnn_shapes`, `cnn_mnistm`, etc.) take an `nnx.Rngs` and fix `num_channels`/`num_classes`/`input_size`; `model_dict[dataset_name]["cnn"](rngs)` is how every script instantiates a model. **Conv layers explicitly override `kernel_init`/`bias_init`** to replicate PyTorch's default `nn.Conv2d` init (`Uniform(-1/√fan_in, 1/√fan_in)` for both weight and bias) — Flax's own defaults differ meaningfully (≈2x larger kernel std, zero bias) and measurably changes trained feature scale/separation between domains, which cascades into the SIDDA loss's dynamic blur/weighting. **`BatchNorm` explicitly sets `momentum=0.9`** — Flax's `momentum` is the *retention* weight on the old running stat (opposite convention from PyTorch, where it's the weight on the *new* batch stat), so replicating PyTorch's default `momentum=0.1` requires Flax `momentum=1-0.1=0.9`, not Flax's own default of `0.99`.
- **`src/scripts/sidda_losses.py`** — `HalfSqEuclidean(costs.SqEuclidean)`: OTT-JAX's built-in `SqEuclidean` cost omits geomloss's `p=2` cost convention's `1/2` factor (`C(x,y) = 0.5*‖x−y‖²`); this custom cost (re-registered as its own pytree node class) makes `epsilon = blur**2` literally correct, matching `geomloss.SamplesLoss("sinkhorn", p=2, blur=b, ...)` exactly — verified numerically (value to ~1e-5, gradient to ~1e-4 via finite differences) against both a hand-rolled log-domain Sinkhorn reference and geomloss itself. `dynamic_sinkhorn_divergence(source_features, target_features)` computes the dynamic blur (`0.05 * max_pairwise_L2_distance`, floored at 0.01, `stop_gradient`'d — matching the original's `.detach()`) and the resulting debiased Sinkhorn divergence via `ott.tools.sinkhorn_divergence.sinkhorn_divergence`. `jensen_shannon_distance`/`_divergence`/`kl_divergence` are a direct `jnp` port of the paper's diagnostic (operates on raw latent feature vectors, not proper probability distributions — an inherited quirk from the original, kept as-is).
- **`src/scripts/checkpointing.py`** — Orbax-based save/restore for `nnx.Module`s. `save_checkpoint` saves the **full** `nnx.split(model)` state (both `nnx.Param` *and* `nnx.BatchStat` — dropping BatchNorm running stats would silently break eval-mode inference after restore). `load_models` allowlist-filters run-directory entries against `KNOWN_CHECKPOINT_NAMES` before attempting to restore them (every Orbax checkpoint is a directory, so a naive "any subdirectory" scan would also try to restore the run's `losses/`/`metrics/`/`confusion_matrix/`/etc. output dirs and crash).
- **`src/scripts/train_CE.py`** — standard supervised training loop (cross-entropy only), used as a baseline and also conceptually corresponds to the paper's warmup phase. Single `nnx.Optimizer` (`optax.chain(clip_by_global_norm(10.0), adamw(lr_schedule, weight_decay=...))`), `lr_schedule` built via `optax.piecewise_constant_schedule` from the config's epoch-based `milestones`/`lr_decay`.
- **`src/scripts/train_SIDDA.py`** — the core DA training loop, and the highest-risk file for subtle math drift when modifying. For `epoch < warmup`: identical to `train_CE.py` (`train_step_ce`). After warmup (`train_step_sidda`): concatenates source+target batches into one forward pass, splits features/logits back apart, computes source cross-entropy + the dynamic Sinkhorn DA loss (via `sidda_losses.py`), and combines them via **learnable log-variance loss weighting**: `EtaParams(nnx.Module)` holds `eta_1`, `eta_2` as `nnx.Param`s, trained via a **second, separate `nnx.Optimizer`** (same lr-schedule/weight_decay as the model's, but *no* gradient clipping — matching the original's `clip_grad_norm_(model.parameters(), ...)` being called with only model params, never `eta_1`/`eta_2`). Two independent optimizers stepped in lockstep is mathematically equivalent to the original PyTorch code's single `AdamW` with `eta_1`/`eta_2` added via `add_param_group`. `loss = (1/(2*eta_1²))*CE + (1/(2*eta_2²))*DA_loss + log(|eta_1|*|eta_2|)`; after computing gradients but **before** applying the optimizer step, `eta_1`/`eta_2` are clamped (`eta_1 ≥ 1e-3`, `eta_2 ≥ 0.25*eta_1`) — this ordering (clamp-before-step, not after) matters and is copied exactly from the original, which leaves the *freshly-updated* eta values unclamped until the *next* iteration's clamp call, a looser/lagging enforcement than clamping immediately post-update. Tracks Jensen-Shannon distance as a diagnostic. Saves four separate "best" checkpoints (by val loss, val accuracy, classification loss, DA loss — see the exact per-checkpoint save conditions in the function body, transcribed carefully from the original since they're easy to get subtly wrong, e.g. only 3 of the 4 are gated on `epoch >= warmup`). The early-stopping `no_improvement_count` is never reset to 0 on improvement (only ever incremented) — a known, pre-existing bug inherited unfixed from the original PyTorch code; harmless as long as `early_stopping` is set ≥ total epochs.
- **`src/scripts/test.py`** — `checkpointing.load_models` + a plain (non-jitted) inference loop calling `model(x, train=False)`; sklearn `classification_report`/`confusion_matrix`/heatmap-PNG plumbing is framework-agnostic and unchanged from a from-scratch design.
- **`src/scripts/test_calibration.py`** — same evaluation flow, plus post-hoc calibration: `sklearn.calibration.CalibratedClassifierCV` (sigmoid/Platt scaling over `LogisticRegression`) fit on the extracted latent features, reports Expected Calibration Error and Brier score alongside the calibrated classification report/confusion matrix. Imports `load_models` from `checkpointing.py`, not from `test.py`.
- **`src/scripts/download_data.py`** — fetches/extracts the 5 Zenodo datasets (see "Downloading data" above).
- **`src/notebooks/`** — exploratory notebooks per dataset (`shapes`, `astronomical_objects`, `mnistm`, `GZ_evo`), unmodified from the original PyTorch repo. May reference the old torch-based API.
- **`src/paper_notebooks/`** — notebooks reproducing paper figures (isomaps, JS-distance, eta evolution), unmodified from the original PyTorch repo. May reference the old torch-based API.

### Bugs found and fixed during the JAX rewrite (worth knowing if diffing against the original PyTorch code)

1. **Transform-aliasing** (`split_dataset` in the original PyTorch `train_CE.py`/`train_SIDDA.py`): `torch.utils.data.random_split` returns two `Subset`s sharing the *same* underlying dataset object; assigning `train_subset.dataset.transform = train_transform` then `val_subset.dataset.transform = val_transform` meant **both ended up using `val_transform`** — training-time augmentation never actually ran in the original. Fixed here: `Subset` owns its own transform reference, verified via a regression test that `train_transform(img) != val_transform(img)` on the same source image.
2. **mrssc2 resize inconsistency**: the original `test.py` resized mrssc2 to 256×256 while `train_CE.py`/`train_SIDDA.py`/`test_calibration.py` all resized it to 100×100 (matching the model's expected input size) — a real bug that would break the FC layer's expected flatten size at test time. Fixed by centralizing all per-dataset transform construction in `augment.get_transform`, used identically by every script.
3. **BatchNorm momentum**, **eta clamp-ordering**, **conv weight/bias init** — see the relevant bullets above; these are framework-porting discrepancies (Flax defaults silently differing from PyTorch defaults/semantics) rather than logic bugs in the original, but were significant enough to visibly affect SIDDA training dynamics (a converging vs. degrading accuracy trajectory) before being found and fixed.

New datasets are added by: adding entries to `dataset_dict`/`classes_dict` in `dataset.py`, a `_INPUT_SIZES` entry plus per-dataset factory function(s) in `models.py`/`augment.py`, and a download spec in `download_data.py`'s `DATASETS` dict.
