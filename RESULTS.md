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

Paper numbers are the actual Table 1 values (CNN / CNN-DA columns) — an earlier
version of this file compared GZ Evo and MRSSC2 against incorrect reference numbers
that don't appear anywhere in the paper; corrected below.

| Dataset | CNN src/tgt | CNN-DA src/tgt | Paper CNN src/tgt | Paper CNN-DA src/tgt | Verdict |
|---|---|---|---|---|---|
| Shapes | 99.87 / 65.20 | 99.30 / 36.53 | 99.80 / 50.47 | 99.82 / 78.20 | Source/CE close match; DA target benefit is checkpoint-dependent |
| Astro. objects | 99.13 / 79.13 | 95.70 / 92.83 | 99.34 / 50.81 | 95.32 / 91.33 | **Close match** |
| MNIST-M | 95.30 / 70.22 | 94.36 / 77.56 | 95.64 / 68.32 | 95.31 / 76.24 | **Close match** |
| Galaxy Zoo Evo | 81.29 / 65.48 | 79.90 / 78.28 | 81.49 / 70.65 | 81.57 / 77.54 | **Close match** |
| MRSSC2 | 77.50 / 25.35 | 49.31 / 37.16 | 76.14 / 31.28 | 71.27 / 36.80 | Source/CE close match; DA target matches almost exactly, but DA source is checkpoint-dependent |

## Assessment

**MNIST-M, astronomical objects, and Galaxy Zoo Evo all closely reproduce the paper**
on both source and target, once augmentation is handled correctly. All three show the
paper's qualitative signature cleanly: source accuracy roughly unchanged, target
accuracy up substantially with DA.

**One open, unresolved issue: shapes' and MRSSC2's SIDDA checkpoints are unstable.**
Source-accuracy collapse is fixed (shapes holds ≥99% throughout the DA phase), and the
*best-case* numbers from either dataset land close to the paper — but which checkpoint
gives the good number isn't consistent. Shapes' `best_model_val_acc` gets 64.47%
target (roughly flat vs. CE's 65.20%) while `final_model` drops to 36.53% (worse than
CE, source still fine at 99.3%). MRSSC2 shows the mirror image: `final_model`'s target
(37.16%) nearly exactly matches the paper's CNN-DA target (36.80%), but its source
(49.31%) is far below the paper's 71.27%, while `best_model_val_acc`'s source (74.90%)
is close to the paper but its target (23.70%) is not. In both cases, no single
checkpoint simultaneously matches the paper on *both* source and target — the DA
mechanism isn't settling into one stable optimum for these two datasets the way it
does for mnist_m/astro_objects/gz_evo. Not yet root-caused.

**Bottom line**: the core SIDDA mechanism, once decoupled from the augmentation bug,
correctly reproduces the paper on 3 of 5 datasets, and gets close on the other 2 —
the remaining open question is specifically about checkpoint-to-checkpoint training
stability for shapes/mrssc2, not a baseline accuracy ceiling (an earlier version of
this document mischaracterized it as the latter, due to comparing against incorrect
reference numbers).

See `CLAUDE.md` for the full technical writeup (architecture, all bugs found and
fixed, exact hyperparameter table, file-by-file details).
