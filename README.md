# SIDDA-JAX

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A JAX / Flax NNX / OTT-JAX reimplementation of [deepskies/SIDDA](https://github.com/deepskies/SIDDA), the reference PyTorch implementation for:

> Pandya, S., Patel, P., Nord, B.D., Walmsley, M., Ciprijanovic, A. **SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks**. *Machine Learning: Science and Technology* (2025). [arXiv:2501.14048](https://arxiv.org/abs/2501.14048)

SIDDA trains an image classifier with an optional Sinkhorn-divergence domain adaptation loss, dynamically tuning the DA regularization and loss weighting during training instead of requiring manual hyperparameter search. This port covers the plain CNN pipeline only — the original's escnn-based equivariant (ENN) model was not ported.

## Install

```bash
uv sync
source .venv/bin/activate
```

## Data

```bash
cd src/scripts
python download_data.py --all   # or --dataset <shapes|astro_objects|mnist_m|gz_evo|mrssc2>
```

Datasets: [Zenodo record 15215272](https://zenodo.org/records/15215272).

## Train / evaluate

```bash
cd src/scripts
python train_CE.py --config example_yaml_train_CE.yaml       # cross-entropy baseline
python train_SIDDA.py --config example_yaml_train_SIDDA.yaml # + domain adaptation

python test.py --model_path <run_dir> --x_test_path <path> --y_test_path <path> \
  --output_name <name> --model_name cnn --dataset <shapes|astro_objects|mnist_m|gz_evo|mrssc2>
```

`test_calibration.py` runs the same evaluation plus post-hoc calibration (ECE, Brier score).

## Reproducing the paper's numbers

Two things that aren't obvious from the paper text alone:
- Per-dataset warmup/epoch/LR-milestone values differ (paper Appendix A, Table 7); the bundled example configs use the shapes values.
- Training-time augmentation defaults to **off** (`parameters.augment: false`). The original code has a bug that meant its published results were, in practice, trained without augmentation despite the paper describing it — turning augmentation on for real measurably hurts reproduction here (e.g. it mislabels MNIST-M digits under ±180° rotation, since a `6` rotated ~180° looks like a `9`).

See `RESULTS.md` for per-dataset reproduction numbers and open issues, and `CLAUDE.md` for the full technical writeup.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{Pandya_2025,
   title={SIDDA: SInkhorn Dynamic Domain Adaptation for image classification with equivariant neural networks},
   ISSN={2632-2153},
   url={http://dx.doi.org/10.1088/2632-2153/adf701},
   DOI={10.1088/2632-2153/adf701},
   journal={Machine Learning: Science and Technology},
   publisher={IOP Publishing},
   author={Pandya, Sneh and Patel, Purvik and Nord, Brian D and Walmsley, Mike and Ciprijanovic, Aleksandra},
   year={2025},
   month=aug }
```
