# SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/deepskies/SIDDA/blob/main/LICENSE)  
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

![Pipeline Diagram](plots/pipeline.png)

## Overview

**SInkhorn Dynamic Domain Adaptation (SIDDA)** supplements the experiments presented in *[2501.14048](https://arxiv.org/abs/2501.14048), SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks*. 

SIDDA introduces a **semi-supervised, automatic domain adaptation method** that leverages Sinkhorn divergences to dynamically adjust the regularization in the optimal transport plan and the weighting between classification and domain adaptation loss terms during training. 

### Key Features:
- **Minimal hyperparameter tuning**: SIDDA utilizes information from the NN latent space geometry to dynamically adjust the OT plan during training. Loss coefficients are trainable parameters, bypassing the need for tuning loss terms when training with domain adaptation.
- **Extensive validation**: Tested on synthetic and real-world datasets, including:
  - Synthetic shapes and astronomical objects generated with [DeepBench](https://github.com/deepskies/DeepBench).
  - The [MNIST-M](https://paperswithcode.com/dataset/mnist-m) dataset.
  - The [Galaxy Zoo Evo](https://huggingface.co/collections/mwalmsley/galaxy-zoo-evo-66532c6c258f5fad31f31880) dataset.
  - The MRSSC2 SAR/optical remote-sensing dataset.
- This is a **JAX/Flax NNX/OTT-JAX rewrite** of the original PyTorch implementation, covering the plain CNN pipeline only (the escnn-based equivariant model from the paper is not included here).
- **Minimal Computational overhead**: SIDDA is written using JAX, [Flax NNX](https://flax.readthedocs.io/), [Optax](https://optax.readthedocs.io/), and [OTT-JAX](https://ott-jax.readthedocs.io/) for an efficient, differentiable implementation of Sinkhorn divergences.

### Data Availability
All datasets used in this project are available on the [Zenodo record](https://zenodo.org/records/15215272) (DOI 10.5281/zenodo.15215272).

---

## Installation

Requires Python 3.10+. Set up the environment and install dependencies with (e.g. via [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

To download the datasets from Zenodo:

```bash
cd src/scripts
python download_data.py --all   # or --dataset <shapes|astro_objects|mnist_m|gz_evo|mrssc2>
```

## Code Structure

The repository is organized into the following components:

- **Dataset Handling**:  
  `src/scripts/dataset.py`  
  Contains dataset classes for loading and preprocessing all datasets used in the experiments.

- **Model Definitions**:  
  `src/scripts/models.py`  
  A Flax NNX CNN model (equivariant/ENN support was dropped in the JAX rewrite).

- **Training Scripts**:  
  - `src/scripts/train_CE.py`  
    Standard training with cross-entropy loss only.
  - `src/scripts/train_SIDDA.py`  
    Implementation of the SIDDA training algorithm.

- **Testing Scripts**:  
  - `src/scripts/test.py`  
    Standard model evaluation script.
  - `src/scripts/test_calibration.py`  
    Script for evaluating model calibration.

- **Configuration Management**:  
  Training and testing are managed via YAML configuration files.  
  An example configuration file for typical training is provided at:  
  `src/scripts/example_yaml_train_CE.yaml`, while an example yaml for SIDDA is provided at `src/scripts/example_yaml_train_SIDDA.yaml`. To train a model, run 

  ```bash
  python train_SIDDA.py --config example_yaml_train_SIDDA.yaml
  ```

After training, the training results are dumped into a directory <save_dir> which can be specified in the yaml file. The outputted directory has the following naming convention: `<savedir_model_(DA)_timestr>`. The directory includes the best-epoch model, final model, loss curve(s) data, $\sigma_\ell$ values, JS distances, and a config.yaml file with numerical specifics (best epoch, best loss, etc.) saved. 

To test the model, run

```bash
python test.py \
--model_path "/path/to/directory/containing/model" \
--x_test_path "/path/to/test/images" \
--y_test_path "/path/to/test/labels" \
--output_name "name for metrics files" \
--model_name "cnn"
```

The calibration testing script takes all the same arguments as above.

The test script will save:
  - a sklearn classification report for all saved models in the directory (`/dir/metrics`)
  - source and target domain latent vectors for each model on the whole test set (`/dir/latent_vectors`). This can later be used to plot isomaps for the models.
  - model predictions for each model over the whole test set (`dir/y_pred`)
  - confusion matrices for each model over the whole test set (`dir/confusion_matrix`)

The calibration test script will further save:
  - calibrated confusion matrices (`dir/confusion_matrix`)
  - calibrated probabilities on the whole test set (`dir/calibrated_probs`)
  - Expected calibration error (ECE) and Brier scores (`dir/metrics`)

## Notebooks

- **Exploratory Data Analysis**
  - `src/notebooks/astronomical_objects.ipynb` 
  - `src/notebooks/shapes.ipynb`
  - `src/notebooks/GZ_evo.ipynb`
  - `src/notebooks/mnistm.ipynb`

  These notebooks walk through the data generation procedure for simulated datasets (shapes and astronomical objects), inducing covariate shifts (for shapes, astronomical objects, and MNIST-M), and properly loading the galaxy evo dataset.

- **Paper Plots**
  - `src/paper_notebooks/plotting_isomaps.ipynb`
  - `src/paper_notebooks/plotting_js_distances.ipynb`

These notebooks can be used to reproduce Figures 4 and 5 in the paper. The data can be found on our Zenodo page.

### Code Authors

- Sneh Pandya

## Citation

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
