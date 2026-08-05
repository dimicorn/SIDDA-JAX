# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SIDDA (SInkhorn Dynamic Domain Adaptation) is the reference implementation for the paper *SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks* (arXiv:2501.14048). It trains an image classifier with an optional unsupervised domain-adaptation loss based on Sinkhorn divergences, and dynamically tunes the DA regularization/loss-weighting during training instead of requiring manual hyperparameter search.

This repo is a **JAX / Flax NNX / Optax / OTT-JAX rewrite** of the paper's original PyTorch implementation, covering the plain **CNN pipeline only** — the paper's escnn-based equivariant (ENN/D4) model was intentionally dropped in the rewrite, not ported.

There is no build system, package manifest, linter config, or test suite in this repo — it is a small collection of standalone training/eval scripts run directly with `python`.

## Environment

```bash
uv sync
source .venv/bin/activate
```

Dependencies are pinned in `uv.lock` (resolved from `pyproject.toml`); `requirements.txt` is kept as a plain-pip fallback (`uv pip install -r requirements.txt` / `pip install -r requirements.txt`) but `uv.lock` is the source of truth — update both together if dependencies change.

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

Key config fields (see the example YAMLs): `model` (`cnn` — the only supported value post-rewrite), `dataset` (key into `dataset_dict`/`model_dict`), `train_data.{input_path,output_path}` (source domain) and, for SIDDA only, `train_data.{target_input_path,target_output_path}` (target domain; labels unused during training), `parameters.warmup` (epochs of CE-only training before the DA loss kicks in), `parameters.{lr,weight_decay,val_size,batch_size,epochs,early_stopping,report_interval,lr_decay,milestones}`, `parameters.augment` (optional, **default `False`** — see "Reproducing the paper's results" below for why), `seed`, and `save_dir`.

Training output directory naming: `<save_dir><model>_DA_<timestr>` for SIDDA runs, `<save_dir><model>_<timestr>` for CE-only runs. Each run directory gets a `config.yaml` (input config plus final metrics), a `losses/` subdirectory (per-epoch and per-step `.npy` arrays, loss/JS-distance/blur/eta plots), and checkpoint **directories** (Orbax format, not flat files) named `best_model_val_acc/`, and for SIDDA also `best_model_total_val_loss/`, `best_model_classification_loss/`, `best_model_DA_loss/`, plus `final_model/`.

There is no automated test suite — verify changes by running the training/eval scripts end-to-end on a small dataset (`shapes` is fastest — 33.6MB, ~15-25s/epoch on CPU), or by loading `src/notebooks/*.ipynb`.

## Architecture

- **`src/scripts/dataset.py`** — `NpyImageDataset`: loads a `.npy` image array (and optionally a `.npy` label array), applies a transform, returns `(img, label)` — or just `img` when `target_domain=True`. `Subset` wraps a parent dataset + an index array + its own independent transform (deliberately does **not** mutate the parent's `.transform` in place — see the transform-aliasing note below). `split_dataset` builds train/val `Subset`s via `np.random.default_rng(seed).permutation`. `NumpyLoader` is a minimal batch iterator (shuffles per-epoch, yields stacked `np.ndarray` batches) replacing `torch.utils.data.DataLoader`. `dataset_dict`/`classes_dict` map dataset-name strings to the class/class-label tuple.
- **`src/scripts/augment.py`** — single source of truth for per-dataset data augmentation, replacing torchvision transforms. Uses `scipy.ndimage` (not PIL) for rotation/translation, since these datasets are pre-generated `float64` arrays with values outside `[0, 255]` in places (e.g. shapes' noise-shifted target domain runs up to ~7.0), not uint8 photographs. `get_transform(dataset_name, train, rng, input_size=None)` returns a per-item callable: `train=True` gets random rotation (±180°) + translation (±10%) + h/v flips (p=0.3 each) + resize + normalize(mean=0.5, std=0.5); `train=False` gets only resize + normalize. **`train_CE.py`/`train_SIDDA.py` do not call this with `train=True` by default** — gated behind `parameters.augment` (default `False`); see "Reproducing the paper's results" below for why.
- **`src/scripts/models.py`** — `CNN(nnx.Module)`: 3 conv blocks (`nnx.Conv`→`nnx.BatchNorm`→`nnx.relu`→`nnx.max_pool`→`nnx.Dropout(0.2)`, kernel sizes 5/3/3, channels 8/16/32), flatten, `nnx.Linear(→256)`→`nnx.LayerNorm`→**latent_space**→`nnx.Linear(→num_classes)`→**logits**. `__call__(self, x, *, train)` returns `(latent_space, logits)` — the 256-d latent vector is what the Sinkhorn DA loss operates on during training and what gets dumped to `latent_vectors/` during testing. Input is **NHWC** (JAX/Flax convention), not NCHW. The flatten size feeding `fc1` is computed dynamically via a real forward pass through the conv/pool stack at construction time (not hand-derived — VALID-padding maxpools on odd intermediate sizes don't divide evenly, e.g. 100→50→25→**12**, not 12.5). Per-dataset factories (`cnn_shapes`, `cnn_mnistm`, etc.) take an `nnx.Rngs` and fix `num_channels`/`num_classes`/`input_size`; `model_dict[dataset_name]["cnn"](rngs)` is how every script instantiates a model. **Conv layers explicitly override `kernel_init`/`bias_init`** to replicate PyTorch's default `nn.Conv2d` init (`Uniform(-1/√fan_in, 1/√fan_in)` for both weight and bias) — Flax's own defaults differ meaningfully (≈2x larger kernel std, zero bias) and measurably changes trained feature scale/separation between domains, which cascades into the SIDDA loss's dynamic blur/weighting. **`BatchNorm` explicitly sets `momentum=0.9`** — Flax's `momentum` is the *retention* weight on the old running stat (opposite convention from PyTorch, where it's the weight on the *new* batch stat), so replicating PyTorch's default `momentum=0.1` requires Flax `momentum=1-0.1=0.9`, not Flax's own default of `0.99`.
- **`src/scripts/sidda_losses.py`** — `HalfSqEuclidean(costs.SqEuclidean)`: OTT-JAX's built-in `SqEuclidean` cost omits geomloss's `p=2` cost convention's `1/2` factor (`C(x,y) = 0.5*‖x−y‖²`); this custom cost (re-registered as its own pytree node class) makes `epsilon = blur**2` literally correct, matching `geomloss.SamplesLoss("sinkhorn", p=2, blur=b, ...)` exactly — verified numerically (value to ~1e-5, gradient to ~1e-4 via finite differences) against both a hand-rolled log-domain Sinkhorn reference and geomloss itself. `dynamic_sinkhorn_divergence(source_features, target_features)` computes the dynamic blur (`0.05 * max_pairwise_L2_distance`, floored at 0.01, `stop_gradient`'d — matching the original's `.detach()`) and the resulting debiased Sinkhorn divergence via `ott.tools.sinkhorn_divergence.sinkhorn_divergence`. `jensen_shannon_distance`/`_divergence`/`kl_divergence` are a direct `jnp` port of the paper's diagnostic (operates on raw latent feature vectors, not proper probability distributions — an inherited quirk from the original, kept as-is).
- **`src/scripts/checkpointing.py`** — Orbax-based save/restore for `nnx.Module`s. `save_checkpoint` saves the **full** `nnx.split(model)` state (both `nnx.Param` *and* `nnx.BatchStat` — dropping BatchNorm running stats would silently break eval-mode inference after restore). `load_models` allowlist-filters run-directory entries against `KNOWN_CHECKPOINT_NAMES` before attempting to restore them (every Orbax checkpoint is a directory, so a naive "any subdirectory" scan would also try to restore the run's `losses/`/`metrics/`/`confusion_matrix/`/etc. output dirs and crash).
- **`src/scripts/train_CE.py`** — standard supervised training loop (cross-entropy only), used as a baseline and also conceptually corresponds to the paper's warmup phase. Single `nnx.Optimizer` (`optax.chain(clip_by_global_norm(10.0), adamw(lr_schedule, weight_decay=...))`), `lr_schedule` built via `optax.piecewise_constant_schedule` from the config's epoch-based `milestones`/`lr_decay`.
- **`src/scripts/train_SIDDA.py`** — the core DA training loop, and the highest-risk file for subtle math drift when modifying. For `epoch < warmup`: identical to `train_CE.py` (`train_step_ce`). After warmup (`train_step_sidda`): concatenates source+target batches into one forward pass, splits features/logits back apart, computes source cross-entropy + the dynamic Sinkhorn DA loss (via `sidda_losses.py`), and combines them via **learnable log-variance loss weighting**: `EtaParams(nnx.Module)` holds `eta_1`, `eta_2` as `nnx.Param`s, trained via a **second, separate `nnx.Optimizer`** (same weight_decay as the model's, but *no* gradient clipping — matching the original's `clip_grad_norm_(model.parameters(), ...)` being called with only model params, never `eta_1`/`eta_2`). Two independent optimizers stepped in lockstep is mathematically equivalent to the original PyTorch code's single `AdamW` with `eta_1`/`eta_2` added via `add_param_group` *only if their LR schedules are kept in sync* — they are not the same schedule object. optax evaluates a schedule against the calling optimizer's own internal step counter, and `eta_optimizer.update()` is never called during warmup (`train_step_ce` never touches it), so its counter starts at 0 when SIDDA begins rather than at the true global step `warmup*steps_per_epoch`. The original avoids this because a single shared optimizer + `MultiStepLR` decays every param group's `lr` off one true-epoch-indexed counter, regardless of when a group joined via `add_param_group`. Fixed here by giving eta its own schedule, `eta_lr_schedule = lambda count: lr_schedule(count + warmup*steps_per_epoch)`, so its local counter maps back onto the same global step the unshifted schedule expects. `loss = (1/(2*eta_1²))*CE + (1/(2*eta_2²))*DA_loss + log(|eta_1|*|eta_2|)`; after computing gradients but **before** applying the optimizer step, `eta_1`/`eta_2` are clamped (`eta_1 ≥ 1e-3`, `eta_2 ≥ 0.25*eta_1`) — this ordering (clamp-before-step, not after) matters and is copied exactly from the original, which leaves the *freshly-updated* eta values unclamped until the *next* iteration's clamp call, a looser/lagging enforcement than clamping immediately post-update. Tracks Jensen-Shannon distance as a diagnostic. Saves four separate "best" checkpoints (by val loss, val accuracy, classification loss, DA loss — see the exact per-checkpoint save conditions in the function body, transcribed carefully from the original since they're easy to get subtly wrong, e.g. only 3 of the 4 are gated on `epoch >= warmup`). The early-stopping `no_improvement_count` is never reset to 0 on improvement (only ever incremented) — a known, pre-existing bug inherited unfixed from the original PyTorch code; harmless as long as `early_stopping` is set ≥ total epochs.
- **`src/scripts/test.py`** — `checkpointing.load_models` + a plain (non-jitted) inference loop calling `model(x, train=False)`; sklearn `classification_report`/`confusion_matrix`/heatmap-PNG plumbing is framework-agnostic and unchanged from a from-scratch design.
- **`src/scripts/test_calibration.py`** — same evaluation flow, plus post-hoc calibration: `sklearn.calibration.CalibratedClassifierCV` (sigmoid/Platt scaling over `LogisticRegression`) fit on the extracted latent features, reports Expected Calibration Error and Brier score alongside the calibrated classification report/confusion matrix. Imports `load_models` from `checkpointing.py`, not from `test.py`.
- **`src/scripts/download_data.py`** — fetches/extracts the 5 Zenodo datasets (see "Downloading data" above).
- **`src/notebooks/`** — per-dataset data-*generation* notebooks (`shapes`, `astronomical_objects`, `mnistm`, `GZ_evo`), unmodified from the original PyTorch repo. These have no torch/escnn dependency at all (pure `deepbench`/numpy/sklearn/HuggingFace `datasets`) — they document how the raw Zenodo dataset arrays were originally produced, not model code, so there was nothing to port. Kept as-is.
- `src/paper_notebooks/` (removed) — the original repo's 4 paper-figure-reproduction notebooks (isomaps, JS-distance vs. ENN group order, eta/σ evolution). Where they imported `torch` it was dead code (zero actual `torch.*` calls, or divergence helper functions defined but never invoked — the real plotting logic was always just matplotlib over precomputed `.npy` arrays), so there was no substantive torch code to reimplement in JAX. The actual blocker to usability was that all 4 depend on external precomputed artifacts (isomap latents, JS-distance sweeps, eta/σ evolution dumps) from a separate Zenodo upload never referenced or downloaded elsewhere in this repo, and 3 of the 4 specifically compare the D1/D2/D4/D8 ENN group orders — a model family this port doesn't include. Hardcoded personal machine paths (a local TeX install, `/Users/snehpandya/Projects/...` save paths) confirmed these were one-off scripts rather than reusable code, so they were removed rather than kept in a permanently non-functional state.

### Bugs found and fixed during the JAX rewrite (worth knowing if diffing against the original PyTorch code)

1. **Transform-aliasing** (`split_dataset` in the original PyTorch `train_CE.py`/`train_SIDDA.py`): `torch.utils.data.random_split` returns two `Subset`s sharing the *same* underlying dataset object; assigning `train_subset.dataset.transform = train_transform` then `val_subset.dataset.transform = val_transform` meant **both ended up using `val_transform`** — training-time augmentation never actually ran in the original. Fixed here in the sense that `Subset` now owns its own transform reference (verified via a regression test that `train_transform(img) != val_transform(img)` on the same source image) and augmentation genuinely works when requested — **but the original's real-world published results were, in all likelihood, produced with this bug active, i.e. with no training-time augmentation at all**. See "Reproducing the paper's results" below: replaying the paper's stated augmentation recipe for real measurably breaks reproduction, so `parameters.augment` defaults to `False`, matching what the original's code actually did rather than what it says it did.
2. **mrssc2 resize inconsistency**: the original `test.py` resized mrssc2 to 256×256 while `train_CE.py`/`train_SIDDA.py`/`test_calibration.py` all resized it to 100×100 (matching the model's expected input size) — a real bug that would break the FC layer's expected flatten size at test time. Fixed by centralizing all per-dataset transform construction in `augment.get_transform`, used identically by every script.
3. **BatchNorm momentum**, **eta clamp-ordering**, **conv weight/bias init** — see the relevant bullets above; these are framework-porting discrepancies (Flax defaults silently differing from PyTorch defaults/semantics) rather than logic bugs in the original, but were significant enough to visibly affect SIDDA training dynamics (a converging vs. degrading accuracy trajectory) before being found and fixed.
4. **eta-optimizer LR-schedule desync** — see the `train_SIDDA.py` bullet above for the mechanism. Fixed by giving `eta_optimizer` its own schedule shifted by `warmup*steps_per_epoch`. Doesn't fire until the first post-warmup LR milestone, so it's unrelated to the collapse/reproduction issues below, but is a real deviation from the original's actual single-shared-optimizer behavior worth keeping fixed regardless.

New datasets are added by: adding entries to `dataset_dict`/`classes_dict` in `dataset.py`, a `_INPUT_SIZES` entry plus per-dataset factory function(s) in `models.py`/`augment.py`, and a download spec in `download_data.py`'s `DATASETS` dict.

### Reproducing the paper's results

The paper's Appendix A (Table 7) gives per-dataset training hyperparameters that differ substantially from any example config previously in this repo (which mixed up the D4/ENN model's shorter warmup with the CNN's). All values below are AdamW, batch size 128, lr `1e-2`→`1e-4` via two 0.1× decays applied at epochs `⌊T/3⌋` and `⌊2T/3⌋` (`T` = total epochs; the paper states the decay is "applied twice sequentially" and gives the `⌊T/3⌋` formula for the first one — the second milestone is inferred, not quoted verbatim):

| Dataset | Warmup (CNN-DA) | Total epochs |
|---|---|---|
| Shapes | 10 | 50 |
| Astro. objects | 10 | 50 |
| MNIST-M | 30 | 100 |
| Galaxy Zoo Evo | 30 | 100 |
| MRSSC2 | 30 | 100 |

**Augmentation must be off to reproduce the paper**, for two independent reasons found by direct experiment (not just the transform-aliasing bug above making this the original's *de facto* behavior):
- **mnist_m**: digit class labels are not invariant to ±180° rotation (a `6` rotated ~180° looks like a `9`), so the augmentation silently mislabels a chunk of training data. CE-only source accuracy: 65.7% (augmented, 100 epochs) vs. **94.97%** (unaugmented — paper reports 95.64%).
- **shapes**: rotation-*invariant* labels (a rotated square is still a square), yet the augmentation recipe (±180° rotation + ±10% translation) at `lr=1e-2` still destabilizes plain CE training (oscillates 34-97% all run, ends at 84.50%) vs. clean, monotonic convergence to 100.00% unaugmented. The instability shows up right at LR-decay milestones and, with augmentation, never resolves.

Verification status as of this investigation (paper-exact hyperparameters + `augment: false` except astro_objects, single seed) — see `RESULTS.md` for the full table. **Note**: an earlier version of this section compared gz_evo and mrssc2 against incorrect reference numbers that don't appear anywhere in the paper (95.27/86.54 and 98.29/64.18); the correct Table 1 values are used below and in `RESULTS.md`.
- **mnist_m, astro_objects, gz_evo**: all three closely reproduce the paper on both source and target (e.g. mnist_m SIDDA `final_model` 94.4%/77.6% vs. paper 95.31%/76.24%; gz_evo SIDDA `final_model` 79.9%/78.3% vs. paper 81.57%/77.54%), and no longer show the catastrophic warmup→DA collapse seen with augmentation on for mnist_m (61.5%/36.1%, worse than not doing DA at all).
- **shapes, mrssc2**: source-domain collapse is fixed (shapes SIDDA stays ≥99% throughout, vs. crashing to 34% mid-training with augmentation on), and the *best-case* numbers from either dataset land close to the paper — but **no single checkpoint reproduces both source and target simultaneously**. Shapes' `best_model_val_acc` gets 64.47% target (flat vs. CE's 65.20%) while `final_model` drops to 36.53% (worse than CE, source still 99.3%). MRSSC2's `final_model` target (37.16%) nearly exactly matches the paper's CNN-DA target (36.80%), but its source (49.31%) is far below the paper's 71.27%, while `best_model_val_acc`'s source (74.90%) is close to the paper but its target (23.70%) is not. This is a known, open, unresolved gap in DA training stability specifically for these two datasets — not yet root-caused.

Don't assume `augment: false` alone is a complete fix for the DA mechanism — it resolves the catastrophic source-accuracy collapse and gets mnist_m very close to the paper, but shapes shows the underlying SIDDA training dynamics can still fail to produce a real target-domain improvement even once source-domain training is stable.
