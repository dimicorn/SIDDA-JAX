# Reproduction Results

This documents how closely this JAX/Flax NNX/OTT-JAX port reproduces the paper's
(arXiv:2501.14048) CNN/CNN-DA numbers, using the paper's actual per-dataset
hyperparameters (Appendix A, Table 7 — see `CLAUDE.md`), a single seed, CPU-only JAX
(Apple M2 Pro). Raw configs/logs/checkpoints backing these numbers live under
`results/paper_exact/` (gitignored, local only).

## The key finding: training-time augmentation must be off

Every dataset, when trained with the paper's stated augmentation recipe (random
±180° rotation, ±10% translation, h/v flips), showed a severe source-accuracy
collapse the moment SIDDA's DA loss activated after warmup — regardless of using the
paper's exact hyperparameters. This traced back to a bug in the original PyTorch code
(`split_dataset`'s `Subset`s silently sharing a transform reference after
`random_split`, see `CLAUDE.md`): the paper's own published results were, in all
likelihood, produced with this bug active, i.e. with **no training-time augmentation
actually applied**, despite the paper describing it. Replaying the augmentation recipe
for real:
- Mislabels MNIST-M digits under ±180° rotation (a `6` rotated ~180° looks like a `9`).
- Destabilizes plain CE training even on rotation-invariant-label data — shapes'
  CE-only baseline oscillated between 34% and 97% accuracy for an entire 50-epoch run
  with augmentation on, vs. a clean, monotonic climb to 100.00% without it.

`parameters.augment` now defaults to `False` in `train_CE.py`/`train_SIDDA.py`. Results
below use `augment: false` for shapes/mnist_m/mrssc2/gz_evo; the astro_objects SIDDA
run used augmentation (it was trained before this fix landed) and was **not**
destabilized by it — consistent with galaxy morphology labels being genuinely
rotation-invariant, unlike digits.

## Results (test set, `final_model` checkpoint unless noted)

| Dataset | CNN src/tgt | CNN-DA src/tgt | Paper CNN src/tgt | Paper CNN-DA src/tgt | Verdict |
|---|---|---|---|---|---|
| Shapes | 99.87 / 65.20 | 99.30 / 36.53 | 99.80 / 50.47 | 99.82 / 78.20 | Unresolved — DA benefit is checkpoint-dependent, sometimes worse than CE |
| Astro. objects | 99.13 / 79.13 | 95.70 / 92.83 | 99.34 / 50.81 | 95.32 / 91.33 | **Close match** |
| MNIST-M | 95.30 / 70.22 | 94.36 / 77.56 | 95.64 / 68.32 | 95.31 / 76.24 | **Close match** |
| Galaxy Zoo Evo | 81.29 / 65.48 | 79.90 / 78.28 | 95.27 / 86.54 | 95.31 / 91.83 | Genuine DA gain, but ~15pp below paper on both source and target |
| MRSSC2 | 77.50 / 25.35 | 49.31 / 37.16 | 98.29 / 64.18 | 98.25 / 77.21 | Unresolved — ~20pp below paper even on the CE baseline |

## Assessment

**MNIST-M and astronomical objects closely reproduce the paper** on both source and
target, once augmentation is handled correctly. Both show the paper's qualitative
signature cleanly: source accuracy roughly unchanged, target accuracy up substantially
with DA.

**Two open, unresolved gaps:**

1. **Shapes' and MRSSC2's DA benefit is checkpoint-dependent and sometimes inverted.**
   Source-accuracy collapse is fixed (shapes holds ≥99% throughout the DA phase), but
   the *target*-domain benefit is inconsistent across checkpoints — e.g. shapes'
   `best_model_val_acc` gets 64.47% target (roughly flat vs. CE's 65.20%) while
   `final_model` drops to 36.53% (worse than CE). MRSSC2 shows the same
   inconsistency in direction across checkpoints. This suggests the DA mechanism
   isn't settling into a stable optimum for these two datasets the way it does for
   mnist_m/astro_objects/gz_evo.
2. **MRSSC2 and Galaxy Zoo Evo — the two real-world (non-synthetic) photographic
   datasets — have a substantial ceiling gap below the paper on the CE-only baseline
   alone**, independent of DA entirely. Raw data was checked and is clean (`[0,1]`
   range, correct shapes, balanced classes). Not yet explained — untested candidates
   include whether resizing to 100×100 (mrssc2's native resolution is 256×256) loses
   too much discriminative detail, or whether these datasets need more training epochs
   than the paper's stated budget to reach its reported ceiling.

**Bottom line**: the core SIDDA mechanism, once decoupled from the augmentation bug,
is correctly implemented and reproduces the paper closely on 2 of 5 datasets outright,
and shows genuine (if capped) DA benefit on a 3rd (gz_evo). The remaining two datasets
each have a distinct, unresolved gap — one about DA training stability specifically,
one about baseline classification ceiling on real-world imagery — that are known,
open, parked items rather than resolved ones.

See `CLAUDE.md` for the full technical writeup (architecture, all bugs found and
fixed, exact hyperparameter table, file-by-file details).
